if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import copy
import json
import random
import threading
import time

import dill
import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from termcolor import cprint

from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    return copy.deepcopy(x)


class TrainManiFlowRoboCasaWorkspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = ()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: ManiFlowTransformerImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: ManiFlowTransformerImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )
        self.global_step = 0
        self.epoch = 0

    # ── Main training loop ──────────────────────────────────────────────────

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        RUN_ROLLOUT = not cfg.training.debug
        RUN_VALIDATION = True

        if cfg.training.debug:
            cfg.training.num_epochs = 10
            cfg.training.max_train_steps = 5
            cfg.training.max_val_steps = 2
            cfg.training.rollout_every = 2
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Resume
        if cfg.training.resume:
            ckpt = self.get_checkpoint_path()
            if ckpt.is_file():
                cprint(f"Resuming from {ckpt}", "cyan")
                self.load_checkpoint(path=ckpt)

        # Dataset
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_loader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        val_dataset = dataset.get_validation_dataset()
        val_loader = DataLoader(val_dataset, **cfg.val_dataloader)

        cprint(f"Dataset: {dataset.__class__.__name__}", "red")
        cprint(
            f"Train samples: {len(dataset)}  Val samples: {len(val_dataset)}", "red"
        )

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_loader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # EMA
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # Env runner (only for rollout evaluation)
        env_runner: BaseRunner = None
        if RUN_ROLLOUT:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner, output_dir=self.output_dir
            )

        cfg.logging.name = str(cfg.task.name)
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        for _ in range(cfg.training.num_epochs):
            step_log = {}
            train_losses = []
            n_skipped = 0  # この epoch でスキップした NaN/Inf バッチ数

            with tqdm.tqdm(
                train_loader,
                desc=f"Train epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    raw_loss, loss_dict = self.model.compute_loss(
                        batch, self.ema_model, epoch=self.epoch, num_epochs=cfg.training.num_epochs
                    )
                    if not torch.isfinite(raw_loss):
                        n_skipped += 1  # NaN/Inf batch をスキップ（カウントして可視化）
                        continue
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        grad_clip_norm = cfg.training.get("grad_clip_norm", 1.0)
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=grad_clip_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    if cfg.training.use_ema:
                        ema.step(self.model)

                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    step_log.update(loss_dict)

                    is_last = batch_idx == len(train_loader) - 1
                    if not is_last:
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (
                        cfg.training.max_train_steps - 1
                    ):
                        break

            step_log["train_loss"] = float(np.mean(train_losses))
            step_log["n_skipped_batches"] = n_skipped
            if n_skipped > 0:
                print(
                    f"[epoch {self.epoch}] NaN/Inf でスキップした train バッチ: "
                    f"{n_skipped} / {len(train_loader)}",
                    flush=True,
                )

            # Policy used for eval / sampling
            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()

            # Rollout (skip epoch 0: model is untrained, rollout would just waste time)
            if (
                RUN_ROLLOUT
                and env_runner is not None
                and self.epoch > 0
                and (self.epoch % cfg.training.rollout_every) == 0
            ):
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)
            elif self.epoch == 0:
                step_log.update(
                    {
                        "test_mean_score": 0.0,
                        "mean_success_rates": 0.0,
                        "SR_test_L3": 0.0,
                        "SR_test_L5": 0.0,
                    }
                )

            # Validation
            if RUN_VALIDATION and (self.epoch % cfg.training.val_every) == 0:
                with torch.no_grad():
                    val_losses = []
                    with tqdm.tqdm(
                        val_loader,
                        desc=f"Val epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as vepoch:
                        for batch_idx, batch in enumerate(vepoch):
                            batch = dict_apply(
                                batch, lambda x: x.to(device, non_blocking=True)
                            )
                            loss, _ = self.model.compute_loss(
                                batch, self.ema_model, epoch=self.epoch, num_epochs=cfg.training.num_epochs
                            )
                            val_losses.append(loss.item())
                            if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                cfg.training.max_val_steps - 1
                            ):
                                break
                    if val_losses:
                        # 生（nanmean 前）の NaN 数を可視化してからマスクする
                        n_val_nan = int(np.sum(~np.isfinite(val_losses)))
                        step_log["n_val_nan_batches"] = n_val_nan
                        if n_val_nan > 0:
                            print(
                                f"[epoch {self.epoch}] NaN/Inf になった val バッチ: "
                                f"{n_val_nan} / {len(val_losses)}",
                                flush=True,
                            )
                        # nanmean: 万が一 NaN が混入した場合も残りの値で平均を取る
                        step_log["val_loss"] = float(np.nanmean(val_losses))

            # Sample action MSE on training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    b = dict_apply(
                        train_sampling_batch, lambda x: x.to(device, non_blocking=True)
                    )
                    pred = policy.predict_action(b["obs"])["action_pred"]
                    mse = torch.nn.functional.mse_loss(pred, b["action"])
                    step_log["train_action_mse_error"] = mse.item()

            if step_log.get("test_mean_score") is None:
                step_log["test_mean_score"] = -step_log.get("train_loss", 0.0)

            # Checkpoint
            if cfg.checkpoint.save_ckpt:
                # latest.ckpt はホストの突発的な OOM 等で学習が落ちた際に
                # resume=true で直前の epoch から再開できるよう毎 epoch 上書き保存する
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    metric = {k.replace("/", "_"): v for k, v in step_log.items()}
                    try:
                        ckpt_path = topk.get_ckpt_path(metric)
                        if ckpt_path is not None:
                            self.save_checkpoint(path=ckpt_path)
                    except Exception as e:
                        cprint(f"Checkpoint error: {e}", "red")

            policy.train()
            # エポック要約を無条件に出力（wandb 無効時でも NaN 状況を追跡できるように）
            print(
                f"[epoch {self.epoch}] "
                f"train_loss={step_log.get('train_loss', float('nan')):.4f} "
                f"val_loss={step_log.get('val_loss', float('nan')):.4f} "
                f"n_skipped={step_log.get('n_skipped_batches', 0)} "
                f"n_val_nan={step_log.get('n_val_nan_batches', 0)}",
                flush=True,
            )
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1

    # ── Evaluation only ─────────────────────────────────────────────────────

    def eval(self, mode: str = "best", eval_dir_tag: str = ""):
        cfg = copy.deepcopy(self.cfg)
        ckpt = self.get_checkpoint_path(
            tag=mode, monitor_key=cfg.checkpoint.topk.monitor_key
        )
        if ckpt.is_file():
            cprint(f"Loading {mode} checkpoint: {ckpt}", "magenta")
            self.load_checkpoint(path=ckpt)

        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        policy = self.ema_model if cfg.training.use_ema else self.model
        policy.eval()
        policy.cuda()

        for n_steps in cfg.get("eval_inference_steps", [10]):
            policy.num_inference_steps = n_steps

            subdir = f"steps{n_steps}" + (f"_{eval_dir_tag}" if eval_dir_tag else "")
            eval_dir = os.path.join(
                self.output_dir,
                f"eval_results/{self.epoch}/{subdir}",
            )
            os.makedirs(eval_dir, exist_ok=True)
            video_dir = os.path.join(eval_dir, "videos")

            runner_log = env_runner.run(policy, video_dir=video_dir)

            metrics = {k: v for k, v in runner_log.items() if isinstance(v, (int, float))}
            with open(os.path.join(eval_dir, f"metrics_{mode}.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            cprint(f"Eval results saved to {eval_dir}", "magenta")
            for k, v in metrics.items():
                cprint(f"  {k}: {v:.4f}", "magenta")

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    @property
    def output_dir(self):
        if self._output_dir is None:
            return HydraConfig.get().runtime.output_dir
        return self._output_dir

    def save_checkpoint(self, path=None, tag="latest", use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in self.exclude_keys:
                    payload["state_dicts"][key] = (
                        _to_cpu(value.state_dict()) if use_thread else value.state_dict()
                    )
            elif key in self.include_keys:
                payload["pickles"][key] = dill.dumps(value)

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest", monitor_key="val_loss"):
        import pathlib

        ckpt_dir = pathlib.Path(self.output_dir) / "checkpoints"
        if tag == "latest":
            return ckpt_dir / "latest.ckpt"
        if tag == "best":
            best, best_score = None, float("inf") if "loss" in monitor_key else -float("inf")
            for p in ckpt_dir.glob("*.ckpt"):
                if "latest" in p.name:
                    continue
                try:
                    score = float(p.stem.split(f"{monitor_key}=")[1])
                    if ("loss" in monitor_key and score < best_score) or (
                        "loss" not in monitor_key and score > best_score
                    ):
                        best, best_score = p, score
                except (IndexError, ValueError):
                    pass
            if best is None:
                raise ValueError(f"No checkpoint found with key {monitor_key}")
            return best
        raise NotImplementedError(f"Unknown tag: {tag}")

    def load_payload(self, payload, exclude_keys=None, include_keys=None):
        exclude_keys = exclude_keys or ()
        include_keys = include_keys or list(payload["pickles"].keys())
        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(self, path=None, tag="latest", **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, **kwargs)

        # use_consistency / sample_target_t_mode change sample_ode()'s inference
        # behavior but are plain python attributes set at construction time, not
        # part of the state_dict. If the invocation that loads this checkpoint
        # (e.g. a later `eval_maniflow_robocasa_workspace` run in the same output
        # dir) doesn't repeat the exact policy.* overrides used at training time,
        # self.model would silently be instantiated with the wrong flag and
        # sample_ode would query the model out of its training distribution.
        # Restore these from the checkpoint's own recorded training cfg so eval
        # always matches how the model was actually trained.
        ckpt_cfg = payload.get("cfg", None)
        if ckpt_cfg is not None:
            for attr in ("use_consistency", "sample_target_t_mode"):
                trained_value = ckpt_cfg.policy.get(attr, None)
                if trained_value is None:
                    continue
                for model_attr in ("model", "ema_model"):
                    policy = getattr(self, model_attr, None)
                    if policy is None:
                        continue
                    current_value = getattr(policy, attr, None)
                    if current_value != trained_value:
                        cprint(
                            f"[load_checkpoint] {model_attr}.{attr}: overriding "
                            f"{current_value} -> {trained_value} (from checkpoint's training cfg)",
                            "yellow",
                        )
                    setattr(policy, attr, trained_value)

        return payload


import pathlib


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "config"),
)
def main(cfg):
    workspace = TrainManiFlowRoboCasaWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
