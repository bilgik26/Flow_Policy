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
    suite_names : list[str] or None
        Multi-suite "seen"/"unseen" rollout mode (used by libero_all4) — one
        entry per LIBERO task suite, e.g. ``["libero_spatial", "libero_object",
        "libero_goal", "libero_10"]``. For each suite, task_ids are split
        deterministically (via ``maniflow.common.libero_task_split``, seeded
        by ``task_split_seed`` below, independent of ``training.seed``) into:
          - ``val_tasks_per_suite`` whole tasks held out entirely from
            training -- rolled out for the "unseen" metric
            (``unseen_mean_success_rate_<suite>``), using
            ``unseen_episodes_per_task`` episodes each.
          - ``seen_tasks_per_suite`` of the remaining (trained-on) tasks --
            rolled out for the "seen" metric
            (``seen_mean_success_rate_<suite>``), using
            ``seen_episodes_per_task`` episodes each.
        This split matches ``LiberoImageDataset``'s own train/val task split
        exactly (same suite name/count/seed in, same task_ids out), so the
        "unseen" tasks here are always precisely the tasks
        ``LiberoImageDataset`` held out of training -- without the dataset
        and this runner needing to communicate at construction time. Takes
        priority over ``eval_tasks``/``task_suite_name`` below when given.
    val_tasks_per_suite : int
        Per suite, how many whole tasks are held out for "unseen" eval. Must
        match ``task.dataset.val_tasks_per_suite`` for the two to agree on
        which tasks are actually unseen. Only used when ``suite_names`` is
        given.
    seen_tasks_per_suite : int
        Per suite, how many of the remaining (trained-on) tasks are probed
        for the "seen" eval. Only used when ``suite_names`` is given.
    task_split_seed : int
        Seed for the task split above, independent of ``training.seed`` --
        must match ``task.dataset.task_split_seed``. Only used when
        ``suite_names`` is given.
    seen_episodes_per_task : int
        Rollout episodes per task for the "seen" metric. Only used when
        ``suite_names`` is given.
    unseen_episodes_per_task : int
        Rollout episodes per task for the "unseen" metric. Only used when
        ``suite_names`` is given.
    eval_tasks : list[dict] or None
        Generic multi-suite rollout spec — one dict per LIBERO task suite to
        evaluate, each with keys:
          - task_suite_name (str): e.g. ``"libero_spatial"``, ``"libero_object"``,
            ``"libero_goal"``, ``"libero_10"``, ``"libero_90"``.
          - task_ids (list[int] or None, optional): subset of task indices to
            evaluate within that suite. ``None`` → every task in the suite
            (the standard LIBERO benchmark protocol).
          - eval_episodes_per_task (int, optional): episodes per task for
            this suite; falls back to the top-level ``eval_episodes_per_task``
            if omitted.
        Lower-level general-purpose alternative to ``suite_names`` above, for
        configs that want explicit control over each suite's task_ids rather
        than the seen/unseen split (e.g. manual CLI overrides). Ignored when
        ``suite_names`` is given; takes priority over the single-suite
        ``task_suite_name``/``task_ids``/``eval_episodes_per_task`` args
        below when given.
    task_suite_name : str or None
        Single-suite convenience form, kept for backward compatibility with
        existing task yaml files that evaluate only one suite. Ignored when
        ``suite_names`` or ``eval_tasks`` is given.
    task_ids : list[int] or None
        Subset of task indices to evaluate. ``None`` → every task in the
        suite (the standard LIBERO benchmark protocol). Use a small subset
        (e.g. ``[0]``) for fast periodic rollouts during training, and
        ``None`` with a higher ``eval_episodes_per_task`` for the final
        full-suite evaluation. Only used when ``suite_names``/``eval_tasks``
        are not given.
    eval_episodes_per_task : int
        Number of rollout episodes (trials) per task. The official LIBERO
        benchmark uses 50; a smaller number is fine for quick checks. Used
        directly in the single-suite form, and as the default for any
        ``eval_tasks`` entry that omits its own ``eval_episodes_per_task``.
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
        suite_names: Optional[List[str]] = None,
        val_tasks_per_suite: int = 2,
        seen_tasks_per_suite: int = 2,
        task_split_seed: int = 0,
        seen_episodes_per_task: int = 10,
        unseen_episodes_per_task: int = 10,
        eval_tasks: Optional[List[dict]] = None,
        task_suite_name: Optional[str] = None,
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
        if suite_names is not None:
            from libero.libero import benchmark as libero_benchmark

            from maniflow.common.libero_task_split import (
                select_seen_eval_tasks,
                split_train_val_tasks,
            )

            self.eval_tasks = []
            for suite in suite_names:
                n_tasks = libero_benchmark.get_benchmark_dict()[suite]().n_tasks
                train_task_ids, val_task_ids = split_train_val_tasks(
                    suite, n_tasks, val_tasks_per_suite, task_split_seed
                )
                seen_eval_task_ids = select_seen_eval_tasks(
                    suite, train_task_ids, seen_tasks_per_suite, task_split_seed
                )
                self.eval_tasks.append(
                    {
                        "task_suite_name": suite,
                        "task_ids": seen_eval_task_ids,
                        "eval_episodes_per_task": seen_episodes_per_task,
                        "group": "seen",
                    }
                )
                self.eval_tasks.append(
                    {
                        "task_suite_name": suite,
                        "task_ids": val_task_ids,
                        "eval_episodes_per_task": unseen_episodes_per_task,
                        "group": "unseen",
                    }
                )
        elif eval_tasks is not None:
            self.eval_tasks = [dict(t) for t in eval_tasks]
        elif task_suite_name is not None:
            self.eval_tasks = [
                {
                    "task_suite_name": task_suite_name,
                    "task_ids": task_ids,
                    "eval_episodes_per_task": eval_episodes_per_task,
                }
            ]
        else:
            raise ValueError(
                "LiberoRunner requires `suite_names` (seen/unseen multi-suite), "
                "`eval_tasks` (generic multi-suite), or `task_suite_name` "
                "(single-suite)."
            )
        # Snapshotted here (before any external mutation of self.eval_tasks) so
        # that distributing rollout work across DDP ranks — each rank keeps only
        # a round-robin slice of self.eval_tasks, see
        # TrainManiFlowLiberoWorkspace.run() — doesn't change whether run()'s
        # wandb keys get suite-prefixed. Without this, a rank left holding a
        # single suite after slicing would drop the "SR_<suite>_taskNN" prefix
        # even though the training run as a whole is evaluating multiple suites.
        self._multi_suite = len(self.eval_tasks) > 1
        self._default_eval_episodes_per_task = eval_episodes_per_task
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

        multi_suite = self._multi_suite
        suite_names = [t["task_suite_name"] for t in self.eval_tasks]

        all_success = []
        # Per seen/unseen group ("seen"/"unseen", see __init__'s suite_names
        # branch) -- only populated when eval_tasks entries actually carry a
        # "group" tag; empty for the plain eval_tasks/task_suite_name modes.
        all_success_by_group: dict = {}
        # "+".join(...) dedupes while preserving order, in case the same suite
        # appears in more than one eval_tasks entry (e.g. two task_ids subsets).
        log_data = {"task_suite_name": "+".join(dict.fromkeys(suite_names))}

        for task_group in self.eval_tasks:
            task_suite_name = task_group["task_suite_name"]
            group_task_ids = task_group.get("task_ids")
            group_episodes = task_group.get(
                "eval_episodes_per_task", self._default_eval_episodes_per_task
            )
            # seen/unseen mode (suite_names given to __init__) tags every
            # entry with "group"; key_prefix then always includes it,
            # regardless of multi_suite. Otherwise only prefix per-suite
            # keys when actually mixing suites, so the plain single-suite
            # case keeps its original wandb metric names (e.g. "SR_task00")
            # unchanged.
            seen_unseen_group = task_group.get("group")
            if seen_unseen_group:
                key_prefix = f"{seen_unseen_group}_{task_suite_name}_"
            else:
                key_prefix = f"{task_suite_name}_" if multi_suite else ""

            task_suite = libero_benchmark.get_benchmark_dict()[task_suite_name]()
            task_ids = group_task_ids if group_task_ids is not None else list(range(task_suite.n_tasks))
            suite_success = []

            for task_id in task_ids:
                env = LiberoEnv(
                    task_suite_name=task_suite_name,
                    task_id=task_id,
                    image_size=self.image_size,
                    num_steps_wait=self.num_steps_wait,
                    seed=0,
                )
                max_steps = self.max_episode_steps or env.max_episode_steps
                task_success = []

                for ep_idx in tqdm.tqdm(
                    range(group_episodes),
                    desc=f"Eval {task_suite_name}/task{task_id:02d}",
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
                    suite_success.append(float(success))
                    all_success.append(float(success))
                    if seen_unseen_group:
                        all_success_by_group.setdefault(seen_unseen_group, []).append(float(success))

                    if save_video and len(frames) > 0:
                        vid = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (N, C, H, W)
                        try:
                            log_data[f"sim_video_eval_{key_prefix}task{task_id:02d}_{ep_idx}"] = wandb.Video(
                                vid, fps=self.fps, format="mp4"
                            )
                        except Exception:
                            pass  # wandb disabled または moviepy 未インストール時はスキップ

                        if video_dir is not None:
                            os.makedirs(video_dir, exist_ok=True)
                            result_tag = "success" if success else "fail"
                            video_path = os.path.join(
                                video_dir,
                                f"{key_prefix}task{task_id:02d}_ep{ep_idx:03d}_{result_tag}.mp4",
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
                log_data[f"SR_{key_prefix}task{task_id:02d}"] = task_sr
                cprint(
                    f"[{task_suite_name}] task {task_id:02d} ({env.task_description}): "
                    f"SR={task_sr:.3f}",
                    "cyan",
                )

            suite_sr = float(np.mean(suite_success)) if suite_success else 0.0
            if seen_unseen_group:
                log_data[f"{seen_unseen_group}_mean_success_rate_{task_suite_name}"] = suite_sr
            elif multi_suite:
                log_data[f"mean_success_rate_{task_suite_name}"] = suite_sr
            cprint(
                f"[{task_suite_name}"
                f"{f'/{seen_unseen_group}' if seen_unseen_group else ''}] "
                f"suite mean success rate: {suite_sr:.3f}",
                "green",
            )

        mean_sr = float(np.mean(all_success)) if all_success else 0.0
        self._logger3.record(mean_sr)
        self._logger5.record(mean_sr)

        log_data.update(
            {
                # Underscore-prefixed: not a wandb metric, just the raw
                # ingredients (total successes / episode count) a caller
                # distributing self.eval_tasks across several LiberoRunner
                # instances (e.g. one per DDP rank) needs to recompute the
                # *training-run-wide* mean_success_rates/SR_test_L3/L5 by
                # combining several partial `run()` results — this instance
                # only ever sees its own eval_tasks slice. See
                # TrainManiFlowLiberoWorkspace.run()'s rollout distribution.
                "_success_sum": float(np.sum(all_success)) if all_success else 0.0,
                "_success_count": len(all_success),
                "mean_success_rates": mean_sr,
                "test_mean_score": mean_sr,
                "SR_test_L3": self._logger3.average_of_largest_K(),
                "SR_test_L5": self._logger5.average_of_largest_K(),
            }
        )
        for group, successes in all_success_by_group.items():
            log_data[f"_success_sum_{group}"] = float(np.sum(successes)) if successes else 0.0
            log_data[f"_success_count_{group}"] = len(successes)
            log_data[f"mean_{group}_success_rates"] = float(np.mean(successes)) if successes else 0.0
        cprint(f"[{log_data['task_suite_name']}] overall mean success rate: {mean_sr:.3f}", "green")
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
