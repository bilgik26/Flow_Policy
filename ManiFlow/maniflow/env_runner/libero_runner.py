"""
Evaluation runner for the LIBERO benchmark using a Flow_Policy image policy.

Structurally mirrors maniflow/env_runner/robocasa_runner.py (sliding
n_obs_steps window, policy queried every num_open_loop_steps, success
statistics collection), but the rollout protocol itself follows
openvla-oft's experiments/robot/libero/run_libero_eval.py:
  - one LiberoEnv is created per task and reused across all trials of that
    task (LIBERO varies the episode via `init_state_idx`, not by recreating
    the env with different physical parameters like RoboCasa's layout/style),
  - success is read directly from the env's own `done`/`info["success"]`
    (LIBERO's BDDL goal check), no extra checker needed,
  - by default evaluates every task in the suite and reports both per-task
    and overall (mean-of-all-episodes) success rates.
"""

import os
from collections import deque
from typing import List, Optional

import imageio
import numpy as np
import torch
import tqdm
import wandb
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.common.logger_util import LargestKRecorder
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.policy.base_policy import BasePolicy


class LiberoRunner(BaseRunner):
    """
    Parameters
    ----------
    output_dir : str
    task_suite_name : str
        One of ``"libero_spatial"``, ``"libero_object"``, ``"libero_goal"``,
        ``"libero_10"``, ``"libero_90"``.
    task_ids : list[int] or None
        Subset of task indices to evaluate. ``None`` → every task in the
        suite (the standard LIBERO benchmark protocol). Use a small subset
        (e.g. ``[0]``) for fast periodic rollouts during training, and
        ``None`` with a higher ``eval_episodes_per_task`` for the final
        full-suite evaluation.
    eval_episodes_per_task : int
        Number of rollout episodes (trials) per task. The official LIBERO
        benchmark uses 50; a smaller number is fine for quick checks.
    n_obs_steps : int
        Observation history window (must match policy.n_obs_steps).
    n_action_steps : int
        Actions predicted per policy call.
    num_open_loop_steps : int
        Steps executed before querying the policy again. openvla-oft uses 8
        (== its action-chunk size) as the default for LIBERO.
    image_size : int
        Rendered image resolution (both agentview and wrist camera).
    num_steps_wait : int
        Dummy no-op steps executed after reset to let objects settle
        physically (see LiberoEnv.reset).
    max_episode_steps : int or None
        Episode time limit. ``None`` → LiberoEnv.TASK_MAX_STEPS[task_suite_name].
    fps : int
        Video frames per second for wandb/local logging.
    tqdm_interval_sec : float
        tqdm refresh interval.
    """

    def __init__(
        self,
        output_dir: str,
        task_suite_name: str,
        task_ids: Optional[List[int]] = None,
        eval_episodes_per_task: int = 20,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_open_loop_steps: int = 8,
        image_size: int = 256,
        num_steps_wait: int = 10,
        max_episode_steps: int = None,
        fps: int = 20,
        tqdm_interval_sec: float = 5.0,
    ):
        super().__init__(output_dir)
        self.task_suite_name = task_suite_name
        self.task_ids = task_ids
        self.eval_episodes_per_task = eval_episodes_per_task
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_open_loop_steps = num_open_loop_steps
        self.image_size = image_size
        self.num_steps_wait = num_steps_wait
        self.max_episode_steps = max_episode_steps
        self.fps = fps
        self.tqdm_interval_sec = tqdm_interval_sec

        self._logger3 = LargestKRecorder(K=3)
        self._logger5 = LargestKRecorder(K=5)

    # ── Public interface ────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, policy: BasePolicy, save_video: bool = True, video_dir: str = None) -> dict:
        from libero.libero import benchmark as libero_benchmark

        from maniflow.env.libero import LiberoEnv

        device = policy.device
        dtype = policy.dtype

        task_suite = libero_benchmark.get_benchmark_dict()[self.task_suite_name]()
        task_ids = self.task_ids if self.task_ids is not None else list(range(task_suite.n_tasks))

        all_success = []
        log_data = {"task_suite_name": self.task_suite_name}

        for task_id in task_ids:
            env = LiberoEnv(
                task_suite_name=self.task_suite_name,
                task_id=task_id,
                image_size=self.image_size,
                num_steps_wait=self.num_steps_wait,
                seed=0,
            )
            max_steps = self.max_episode_steps or env.max_episode_steps
            task_success = []

            for ep_idx in tqdm.tqdm(
                range(self.eval_episodes_per_task),
                desc=f"Eval {self.task_suite_name}/task{task_id:02d}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            ):
                obs_buf = deque(maxlen=self.n_obs_steps)
                frames = []

                obs = env.reset(init_state_idx=ep_idx)
                obs_buf.append(obs)

                action_queue = []
                success = False

                for _ in range(max_steps):
                    if save_video:
                        frame = (obs["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
                        frames.append(frame)

                    if len(action_queue) == 0:
                        stacked = self._stack_obs(obs_buf)
                        obs_dict = dict_apply(
                            stacked,
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=dtype),
                        )
                        obs_dict["task_name"] = [env.task_description]
                        result = policy.predict_action(obs_dict)
                        actions_np = result["action"].squeeze(0).cpu().numpy()
                        action_queue = list(actions_np[: self.num_open_loop_steps])

                    action = action_queue.pop(0)
                    obs, _, done, info = env.step(action)
                    obs_buf.append(obs)

                    if info.get("success", False):
                        success = True
                        break
                    if done:
                        break

                task_success.append(float(success))
                all_success.append(float(success))

                if save_video and len(frames) > 0:
                    vid = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (N, C, H, W)
                    try:
                        log_data[f"sim_video_eval_task{task_id:02d}_{ep_idx}"] = wandb.Video(
                            vid, fps=self.fps, format="mp4"
                        )
                    except Exception:
                        pass  # wandb disabled または moviepy 未インストール時はスキップ

                    if video_dir is not None:
                        os.makedirs(video_dir, exist_ok=True)
                        result_tag = "success" if success else "fail"
                        video_path = os.path.join(
                            video_dir,
                            f"task{task_id:02d}_ep{ep_idx:03d}_{result_tag}.mp4",
                        )
                        try:
                            with imageio.get_writer(video_path, fps=self.fps, format="mp4") as writer:
                                for frame in frames:  # (H, W, C) uint8
                                    writer.append_data(frame)
                        except Exception as e:
                            cprint(f"  [WARNING] Failed to save video {video_path}: {e}", "red")

                torch.cuda.empty_cache()

            env.close()
            task_sr = float(np.mean(task_success)) if task_success else 0.0
            log_data[f"SR_task{task_id:02d}"] = task_sr
            cprint(
                f"[{self.task_suite_name}] task {task_id:02d} ({env.task_description}): "
                f"SR={task_sr:.3f}",
                "cyan",
            )

        mean_sr = float(np.mean(all_success)) if all_success else 0.0
        self._logger3.record(mean_sr)
        self._logger5.record(mean_sr)

        log_data.update(
            {
                "mean_success_rates": mean_sr,
                "test_mean_score": mean_sr,
                "SR_test_L3": self._logger3.average_of_largest_K(),
                "SR_test_L5": self._logger5.average_of_largest_K(),
            }
        )
        cprint(f"[{self.task_suite_name}] overall mean success rate: {mean_sr:.3f}", "green")
        return log_data

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _stack_obs(self, obs_buf: deque) -> dict:
        """Stack obs history into (n_obs_steps, *) arrays, padding as needed."""
        obs_list = list(obs_buf)
        n = self.n_obs_steps

        result = {}
        for key in obs_list[-1].keys():
            arr = np.stack([o[key] for o in obs_list], axis=0)
            if len(obs_list) < n:
                pad = np.repeat(arr[:1], n - len(obs_list), axis=0)
                arr = np.concatenate([pad, arr], axis=0)
            result[key] = arr[-n:]  # (n_obs_steps, *)
        return result
