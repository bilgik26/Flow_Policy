"""
Evaluation runner for RoboCasa environments using a Flow_Policy image policy.

Maintains a sliding window of n_obs_steps observations, calls the policy every
num_open_loop_steps, and collects success statistics across eval_episodes.
"""

import numpy as np
import torch
import tqdm
import wandb
from collections import deque
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.common.logger_util import LargestKRecorder
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.policy.base_policy import BasePolicy


class RoboCasaRunner(BaseRunner):
    """
    Parameters
    ----------
    output_dir : str
    task_name : str
        RoboCasa task name (e.g. ``"TurnOffMicrowave"``).
    eval_episodes : int
        Number of rollout episodes.
    n_obs_steps : int
        Observation history window (must match policy.n_obs_steps).
    n_action_steps : int
        Actions predicted per policy call.
    num_open_loop_steps : int
        Steps to execute before querying the policy again.
    camera_name : str
        Camera used for observations.
    image_size : int
        Rendered image resolution.
    obj_instance_split : str
        ``"target"`` (test) or ``"pretrain"`` (train) object split.
    layout_and_style_ids : list of (int, int) or None
        List of (layout_id, style_id) pairs to cycle through.
        ``None`` → random layouts.
    max_episode_steps : int or None
        Episode time limit. ``None`` → use environment default.
    fps : int
        Video frames per second for wandb logging.
    tqdm_interval_sec : float
        tqdm refresh interval.
    """

    def __init__(
        self,
        output_dir: str,
        task_name: str,
        eval_episodes: int = 50,
        n_obs_steps: int = 2,
        n_action_steps: int = 32,
        num_open_loop_steps: int = 16,
        camera_name: str = "robot0_agentview_left",
        image_size: int = 224,
        obj_instance_split: str = "target",
        layout_and_style_ids=None,
        max_episode_steps: int = None,
        fps: int = 20,
        tqdm_interval_sec: float = 5.0,
    ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_open_loop_steps = num_open_loop_steps
        self.camera_name = camera_name
        self.image_size = image_size
        self.obj_instance_split = obj_instance_split
        self.max_episode_steps = max_episode_steps
        self.fps = fps
        self.tqdm_interval_sec = tqdm_interval_sec

        # Test scene cycling: 5 canonical RoboCasa test scenes
        if layout_and_style_ids is None:
            layout_and_style_ids = [(1, 1), (2, 2), (4, 4), (6, 9), (7, 10)]
        self.layout_and_style_ids = layout_and_style_ids

        self._logger3 = LargestKRecorder(K=3)
        self._logger5 = LargestKRecorder(K=5)

    # ── Public interface ────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, policy: BasePolicy, save_video: bool = True) -> dict:
        from maniflow.env.robocasa import RoboCasaEnv

        device = policy.device
        dtype = policy.dtype

        success_list = []
        log_data = {"task_name": self.task_name}

        for ep_idx in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            layout_id, style_id = self.layout_and_style_ids[
                ep_idx % len(self.layout_and_style_ids)
            ]
            env = RoboCasaEnv(
                task_name=self.task_name,
                camera_name=self.camera_name,
                image_size=self.image_size,
                layout_id=layout_id,
                style_id=style_id,
                obj_instance_split=self.obj_instance_split,
                seed=ep_idx,
            )

            max_steps = self.max_episode_steps or env.max_episode_steps
            obs_buf = deque(maxlen=self.n_obs_steps)
            frames = []

            obs = env.reset()
            obs_buf.append(obs)

            action_queue = []
            success = False

            for step in range(max_steps):
                if save_video:
                    # (C, H, W) → (H, W, C) uint8 for recording
                    frame = (obs["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
                    frames.append(frame)

                if len(action_queue) == 0:
                    # Query policy
                    stacked = self._stack_obs(obs_buf)
                    obs_dict = dict_apply(
                        stacked,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=dtype),
                    )
                    obs_dict["task_name"] = [self.task_name]
                    result = policy.predict_action(obs_dict)
                    actions_np = result["action"].squeeze(0).cpu().numpy()
                    # Use only the first num_open_loop_steps actions
                    action_queue = list(
                        actions_np[: self.num_open_loop_steps]
                    )

                action = action_queue.pop(0)
                obs, _, done, info = env.step(action)
                obs_buf.append(obs)

                if info.get("success", False):
                    success = True
                    break
                if done:
                    break

            env.close()
            success_list.append(float(success))

            if save_video and len(frames) > 0:
                vid = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (N, C, H, W)
                try:
                    log_data[f"sim_video_eval_{ep_idx}"] = wandb.Video(
                        vid, fps=self.fps, format="mp4"
                    )
                except Exception:
                    pass  # wandb disabled または moviepy 未インストール時はスキップ

            torch.cuda.empty_cache()
            cprint(
                f"[{self.task_name}] ep {ep_idx}: success={success}", "cyan"
            )

        mean_sr = float(np.mean(success_list))
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
        cprint(f"[{self.task_name}] mean success rate: {mean_sr:.3f}", "green")
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
