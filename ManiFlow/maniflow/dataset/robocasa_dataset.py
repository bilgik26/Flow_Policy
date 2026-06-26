"""
RoboCasa dataset loader for Flow_Policy.

Reads the LeRobot v2.1 format produced by bilgik26/robocasa:
  {dataset_dir}/
    meta/info.json
    meta/episodes.jsonl
    data/chunk-NNN/episode_NNNNNN.parquet
    videos/chunk-NNN/observation.images.{camera}/episode_NNNNNN.mp4

LeRobot action layout (12-dim, reordered from HDF5 by reorder_hdf5_action):
  [0:4]  base_motion       (x, y, θ, mode)
  [4:5]  control_mode
  [5:8]  end_effector_position delta   ← EEF 位置制御
  [8:11] end_effector_rotation delta   ← EEF 姿勢制御
  [11:12] gripper_close

LeRobot observation.state layout (16-dim, reordered from HDF5 by reorder_hdf5_state):
  [0:3]  base_position      (ワールド座標)
  [3:7]  base_rotation      (クォータニオン)
  [7:10] end_effector_position_relative  (台車基準 EEF 位置)
  [10:14] end_effector_rotation_relative (台車基準 EEF 姿勢)
  [14:16] gripper_qpos

→ EEF 制御に整合する agent_pos: state[7:16] = eef_pos_rel(3)+eef_rot_rel(4)+gripper(2)
→ EEF アクション:               action[5:12] = eef_pos(3)+eef_rot(3)+gripper(1)

Returns Flow_Policy's expected format:
  obs:
    image:     (T, C, H, W)  float32 [0, 1]   CHW layout
    agent_pos: (T, 9)        float32   ← state[7:16]
  action:      (T, 7)        float32   ← action[5:12]
"""

import copy
import json
import pathlib
from typing import Dict, List, Optional, Union

import av
import numpy as np
import pandas as pd
import torch
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    get_image_range_normalizer,
)

# LeRobot 形式の action/state スライス（PandaOmron_modality.json 定義に基づく）
ACTION_START = 5   # action[5:12] = EEF delta pos(3) + ori(3) + gripper(1)
ACTION_END = 12
ACTION_DIM = ACTION_END - ACTION_START  # 7

STATE_START = 7    # state[7:16] = eef_pos_relative(3) + eef_rot_relative(4) + gripper_qpos(2)
STATE_END = 16
STATE_DIM = STATE_END - STATE_START  # 9


def _resolve_lerobot_root(path: pathlib.Path) -> pathlib.Path:
    """
    robocasa の download_datasets は <task>/<date>/lerobot/ に展開する。
    dataset_dirs に <task> 直下を渡された場合でも meta/info.json を探して
    実際の LeRobot ルートを返す。
    """
    if (path / "meta" / "info.json").exists():
        return path
    for candidate in sorted(path.glob("*/lerobot")):
        if (candidate / "meta" / "info.json").exists():
            return candidate
    return path  # フォールバック（エラーは _load_dir で検出）


def _decode_video_frames(
    video_path: str,
    frame_indices: List[int],
    image_size: int,
) -> np.ndarray:
    """
    Decode specific frames from an MP4 file with PyAV.

    Returns (N, C, H, W) float32 in [0, 1].
    Frames are flipped vertically (RoboCasa images are upside-down) and
    resized to image_size × image_size.
    """
    unique_sorted = sorted(set(frame_indices))
    target_set = set(unique_sorted)

    decoded: Dict[int, np.ndarray] = {}
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    fi = 0
    for frame in container.decode(stream):
        if fi in target_set:
            img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            if img.shape[0] != image_size or img.shape[1] != image_size:
                frame_pil = frame.to_image().resize(
                    (image_size, image_size), resample=3
                )
                img = np.asarray(frame_pil)
            img = np.flipud(img).copy()                   # flip upside-down
            decoded[fi] = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            if len(decoded) == len(target_set):
                break
        fi += 1
    container.close()

    return np.stack([decoded[i] for i in frame_indices], axis=0)


class _Episode:
    """Scalar data for one episode (kept in RAM; video decoded on demand)."""

    __slots__ = ("actions", "states", "video_path", "length")

    def __init__(self, actions, states, video_path, length):
        self.actions = actions      # (T, ACTION_DIM) float32
        self.states = states        # (T, STATE_DIM)  float32
        self.video_path = video_path
        self.length = length


class RoboCasaImageDataset(BaseDataset):
    """
    Multi-task image dataset for RoboCasa (LeRobot format).

    Parameters
    ----------
    dataset_dirs : str or list[str]
        One or more LeRobot dataset root directories (one per task).
    horizon : int
        Total window length returned per sample (obs + action steps).
    pad_before : int
        Steps padded at the *start* of each episode (repeat first frame).
        Typically ``n_obs_steps - 1``.
    pad_after : int
        Steps padded at the *end* of each episode (repeat last frame).
        Typically ``n_action_steps - 1``.
    seed : int
        RNG seed for reproducible train/val split.
    val_ratio : float
        Fraction of episodes held out for validation.
    camera_name : str
        LeRobot camera feature name (without the ``observation.images.`` prefix).
    max_train_episodes : int or None
        Cap on the number of training episodes (per dataset_dir combined).
    image_size : int
        Target H=W after resize.
    task_name : str or None
        Optional task label for logging.
    """

    def __init__(
        self,
        dataset_dirs: Union[str, List[str]],
        horizon: int = 34,
        pad_before: int = 1,
        pad_after: int = 31,
        seed: int = 42,
        val_ratio: float = 0.02,
        camera_name: str = "robot0_agentview_left",
        max_train_episodes: Optional[int] = None,
        image_size: int = 224,
        task_name: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(dataset_dirs, str):
            dataset_dirs = [dataset_dirs]

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.camera_name = camera_name
        self.image_size = image_size
        self.task_name = task_name or "robocasa"

        # ── Load all episodes ──────────────────────────────────────────────
        all_episodes: List[_Episode] = []
        for ds_dir in dataset_dirs:
            ds_dir = _resolve_lerobot_root(pathlib.Path(ds_dir))
            cprint(f"Loading RoboCasaDataset from {ds_dir}", "green")
            all_episodes.extend(self._load_dir(ds_dir, camera_name))
        cprint(f"Total episodes: {len(all_episodes)}", "green")

        # ── Train / val split ──────────────────────────────────────────────
        n = len(all_episodes)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_set = set(perm[:n_val].tolist())

        train_idx = [i for i in range(n) if i not in val_set]
        if max_train_episodes is not None:
            train_idx = train_idx[:max_train_episodes]

        self.train_episodes: List[_Episode] = [all_episodes[i] for i in train_idx]
        self.val_episodes: List[_Episode] = [all_episodes[i] for i in sorted(val_set)]

        # ── Flat sample index ──────────────────────────────────────────────
        self._train_samples = self._build_index(self.train_episodes)
        self._val_samples = self._build_index(self.val_episodes)

        # Active split (changed by get_validation_dataset)
        self._samples = self._train_samples
        self._episodes = self.train_episodes

        cprint(
            f"Train: {len(self.train_episodes)} eps / {len(self._train_samples)} samples  |  "
            f"Val: {len(self.val_episodes)} eps / {len(self._val_samples)} samples",
            "yellow",
        )

        # ── Statistics for normalizer ──────────────────────────────────────
        if self.train_episodes:
            self._all_actions = np.concatenate(
                [ep.actions for ep in self.train_episodes], axis=0
            )
            self._all_states = np.concatenate(
                [ep.states for ep in self.train_episodes], axis=0
            )
        else:
            self._all_actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
            self._all_states = np.zeros((1, STATE_DIM), dtype=np.float32)

    # ─── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _load_dir(ds_dir: pathlib.Path, camera_name: str) -> List[_Episode]:
        info = json.loads((ds_dir / "meta" / "info.json").read_text())
        chunks_size = info.get("chunks_size", 1000)

        episodes = []
        with open(ds_dir / "meta" / "episodes.jsonl") as f:
            for line in f:
                meta = json.loads(line.strip())
                ep_idx = meta["episode_index"]
                ep_len = meta["length"]

                chunk = ep_idx // chunks_size
                ch_str = f"chunk-{chunk:03d}"
                ep_str = f"episode_{ep_idx:06d}"

                parquet = ds_dir / "data" / ch_str / f"{ep_str}.parquet"
                video = (
                    ds_dir
                    / "videos"
                    / ch_str
                    / f"observation.images.{camera_name}"
                    / f"{ep_str}.mp4"
                )

                if not parquet.exists() or not video.exists():
                    continue

                df = pd.read_parquet(parquet)

                if "action" not in df.columns:
                    continue
                # action[5:12] = EEF delta pos(3) + ori(3) + gripper(1)
                actions = np.stack(df["action"].values)[
                    :, ACTION_START:ACTION_END
                ].astype(np.float32)

                state_col = (
                    "observation.state" if "observation.state" in df.columns else "state"
                )
                if state_col in df.columns:
                    # state[7:16] = eef_pos_relative(3) + eef_rot_relative(4) + gripper_qpos(2)
                    states = np.stack(df[state_col].values)[
                        :, STATE_START:STATE_END
                    ].astype(np.float32)
                else:
                    states = np.zeros((ep_len, STATE_DIM), dtype=np.float32)

                episodes.append(
                    _Episode(actions, states, str(video), ep_len)
                )
        return episodes

    @staticmethod
    def _build_index(episodes: List[_Episode]) -> List[tuple]:
        """Build list of (ep_list_idx, center_step) pairs."""
        idx = []
        for ep_i, ep in enumerate(episodes):
            for step in range(ep.length):
                idx.append((ep_i, step))
        return idx

    # ─── BaseDataset interface ──────────────────────────────────────────────

    def get_validation_dataset(self) -> "RoboCasaImageDataset":
        val = copy.copy(self)
        val._samples = self._val_samples
        val._episodes = self.val_episodes
        return val

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": self._all_actions, "agent_pos": self._all_states},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._all_actions)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_i, center = self._samples[idx]
        ep = self._episodes[ep_i]
        T = ep.length

        # Window: [center - pad_before, center - pad_before + horizon)
        start = center - self.pad_before
        end = start + self.horizon

        # Clamp frame indices to valid range
        frame_indices = [max(0, min(t, T - 1)) for t in range(start, end)]

        # Decode video frames (T, C, H, W) float32
        images = _decode_video_frames(ep.video_path, frame_indices, self.image_size)

        # Scalar sequences
        actions = np.stack([ep.actions[max(0, min(t, T - 1))] for t in range(start, end)])
        states = np.stack([ep.states[max(0, min(t, T - 1))] for t in range(start, end)])

        data = {
            "obs": {
                "image": images,       # (T, C, H, W) float32
                "agent_pos": states,   # (T, 9)       float32
            },
            "action": actions,         # (T, 7)       float32
        }
        return dict_apply(data, torch.from_numpy)
