"""
LIBERO dataset loader for Flow_Policy.

Reads the LeRobot v2.1 format produced by scripts/convert_libero_to_lerobot.py
(same directory layout convention as maniflow/dataset/robocasa_dataset.py):
  {dataset_dir}/
    meta/info.json
    meta/episodes.jsonl
    data/chunk-NNN/episode_NNNNNN.parquet
    videos/chunk-NNN/observation.images.image/episode_NNNNNN.mp4
    videos/chunk-NNN/observation.images.wrist_image/episode_NNNNNN.mp4

Unlike RoboCasa's LeRobot export (whose action/state are reordered slices of
a 12/16-dim PandaOmron composite-controller vector, see robocasa_dataset.py's
ACTION_START/STATE_START constants), the LIBERO converter stores the native
7-dim OSC_POSE delta-pose+gripper action and the 8-dim eef/gripper proprio
state directly (parquet's "action"/"observation.state" columns already have
exactly ACTION_DIM/STATE_DIM entries) — no index slicing is needed here.

  action: (7,) = eef delta_pos(3) + delta_axis_angle_ori(3) + gripper(1)
  state:  (8,) = eef_pos(3) + axis_angle(eef_quat)(3) + gripper_qpos(2)

Images are stored in the raw (upside-down) orientation the LIBERO/robosuite
renderer produces; a 180° rotation (`img[::-1, ::-1]`) is applied at decode
time to match maniflow/env/libero/libero_wrapper.py's online observation
processing — keep both in sync.

Returns Flow_Policy's expected format:
  obs:
    image:       (T, C, H, W) float32 [0, 1]  CHW layout, agentview camera
    wrist_image: (T, C, H, W) float32 [0, 1]  CHW layout, wrist camera
    agent_pos:   (T, 8)        float32
  action:        (T, 7)        float32
"""

import copy
import json
import pathlib
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import cv2
import decord
import numpy as np
import pandas as pd
import torch
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.model.common.normalizer import LinearNormalizer, get_image_range_normalizer

decord.bridge.set_bridge("native")

ACTION_DIM = 7
STATE_DIM = 8

# Must match the videos/observation.images.{key} subdirectory names written
# by scripts/convert_libero_to_lerobot.py.
IMAGE_KEYS = ("image", "wrist_image")


def _discover_lerobot_dirs(root: pathlib.Path) -> List[pathlib.Path]:
    """
    A LIBERO suite may be exported either as a single flat LeRobot dataset
    (``root/meta/info.json`` exists directly) or as one LeRobot dataset per
    task under ``root`` (``root/<task_name>/meta/info.json``), which is what
    ``scripts/convert_libero_to_lerobot.py`` produces by default so that each
    LIBERO task's episodes stay grouped like RoboCasa's per-task dataset_dirs.
    Returns the list of actual LeRobot dataset roots to load.
    """
    if (root / "meta" / "info.json").exists():
        return [root]
    # root.glob("*/meta/info.json") matches "<root>/<task>/meta/info.json";
    # .parent of that match is the "meta" dir, so we need .parent.parent to
    # get back to "<root>/<task>" (the actual LeRobot dataset root).
    dirs = sorted({p.parent.parent for p in root.glob("*/meta/info.json")})
    if not dirs:
        cprint(f"[WARNING] No LeRobot dataset found under {root}", "red")
    return dirs


# Small per-worker-process LRU cache of decord.VideoReader instances, keyed by
# path. __getitem__ samples are drawn from a globally shuffled index, but a
# shuffled epoch still frequently revisits recently-seen episodes (and
# adjacent horizon windows within the same episode land on the same video),
# so reusing an already-opened reader avoids repeatedly re-parsing the
# container/keyframe index — measured ~2x per-call cost on this dataset's
# videos. Bounded (not unbounded) so per-worker RSS stays predictable; see
# docs/setup_and_train_libero.md's マルチGPU学習 section for why unbounded
# per-worker memory growth is a real hazard on this host (num_workers>1
# already exhausts host RAM via swap thrashing independent of this cache).
_READER_CACHE_SIZE = 16
_reader_cache: "OrderedDict[str, decord.VideoReader]" = OrderedDict()


def _get_video_reader(video_path: str) -> decord.VideoReader:
    reader = _reader_cache.get(video_path)
    if reader is not None:
        _reader_cache.move_to_end(video_path)
        return reader
    reader = decord.VideoReader(video_path, num_threads=2)
    _reader_cache[video_path] = reader
    if len(_reader_cache) > _READER_CACHE_SIZE:
        _reader_cache.popitem(last=False)
    return reader


def _decode_video_frames(
    video_path: str,
    frame_indices: List[int],
    image_size: int,
) -> np.ndarray:
    """
    Decode specific frames from an MP4 file with decord.

    decord.VideoReader.get_batch() seeks directly to the requested frames via
    the container's keyframe index, unlike PyAV's container.decode() which
    only exposes sequential iteration — decoding frames near the end of an
    episode previously meant decoding the whole episode from frame 0 every
    single __getitem__ call. Measured ~2-2.6x faster per call on this
    dataset's ~100-170 frame episodes (with the reader cache above).

    Returns (N, C, H, W) float32 in [0, 1]. Frames are rotated 180° (LIBERO
    images are upside-down) and resized to image_size × image_size.
    """
    unique_sorted = sorted(set(frame_indices))
    reader = _get_video_reader(video_path)
    batch = reader.get_batch(unique_sorted).asnumpy()  # (U, H, W, 3) uint8

    decoded: Dict[int, np.ndarray] = {}
    for fi, img in zip(unique_sorted, batch):
        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        img = img[::-1, ::-1].copy()  # 180° rotate — see module docstring
        decoded[fi] = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    return np.stack([decoded[i] for i in frame_indices], axis=0)


class _Episode:
    """Scalar data for one episode (kept in RAM; video decoded on demand)."""

    __slots__ = ("actions", "states", "video_paths", "length")

    def __init__(self, actions, states, video_paths, length):
        self.actions = actions            # (T, ACTION_DIM) float32
        self.states = states              # (T, STATE_DIM)  float32
        self.video_paths = video_paths    # dict: image_key -> path str
        self.length = length


class LiberoImageDataset(BaseDataset):
    """
    Multi-task image dataset for LIBERO (LeRobot format).

    Parameters
    ----------
    dataset_dirs : str or list[str]
        One or more LeRobot dataset roots. Each entry may itself be a single
        converted episode set, or a directory containing one such set per
        LIBERO task (auto-discovered via ``_discover_lerobot_dirs`` — this is
        how a whole task suite, e.g. libero_spatial's 10 tasks, is combined
        into one multitask dataset without hardcoding LIBERO's verbose
        per-task names in hydra config, unlike RoboCasa's explicit
        ``task_names`` list).
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
    max_train_episodes : int or None
        Cap on the number of training episodes (across all discovered dirs).
    image_size : int
        Target H=W after resize.
    task_name : str or None
        Optional task/suite label for logging.
    use_wrist_image : bool
        If False, the wrist camera video is never decoded and "wrist_image"
        is omitted from the returned obs dict — halves per-sample video
        decode time (both cameras cost about the same). Must match the
        policy's shape_meta (see task/*.yaml's use_wrist_image, applied by
        train_maniflow_libero_workspace.py) or the obs_encoder will look up
        a key this dataset never produces.
    """

    def __init__(
        self,
        dataset_dirs: Union[str, List[str]],
        horizon: int = 10,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        image_size: int = 256,
        task_name: Optional[str] = None,
        use_wrist_image: bool = True,
    ):
        super().__init__()

        if isinstance(dataset_dirs, str):
            dataset_dirs = [dataset_dirs]

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_size = image_size
        self.task_name = task_name or "libero"
        self.use_wrist_image = use_wrist_image

        # ── Load all episodes ──────────────────────────────────────────────
        all_episodes: List[_Episode] = []
        for ds_dir in dataset_dirs:
            for resolved in _discover_lerobot_dirs(pathlib.Path(ds_dir)):
                cprint(f"Loading LiberoDataset from {resolved}", "green")
                all_episodes.extend(self._load_dir(resolved))
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
    def _load_dir(ds_dir: pathlib.Path) -> List[_Episode]:
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
                video_paths = {
                    key: ds_dir / "videos" / ch_str / f"observation.images.{key}" / f"{ep_str}.mp4"
                    for key in IMAGE_KEYS
                }

                if not parquet.exists() or not all(p.exists() for p in video_paths.values()):
                    continue

                df = pd.read_parquet(parquet)
                if "action" not in df.columns:
                    continue
                actions = np.stack(df["action"].values).astype(np.float32)

                state_col = (
                    "observation.state" if "observation.state" in df.columns else "state"
                )
                if state_col in df.columns:
                    states = np.stack(df[state_col].values).astype(np.float32)
                else:
                    states = np.zeros((ep_len, STATE_DIM), dtype=np.float32)

                episodes.append(
                    _Episode(
                        actions,
                        states,
                        {k: str(v) for k, v in video_paths.items()},
                        ep_len,
                    )
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

    def get_validation_dataset(self) -> "LiberoImageDataset":
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
        if self.use_wrist_image:
            normalizer["wrist_image"] = get_image_range_normalizer()
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

        images = _decode_video_frames(ep.video_paths["image"], frame_indices, self.image_size)

        actions = np.stack([ep.actions[max(0, min(t, T - 1))] for t in range(start, end)])
        states = np.stack([ep.states[max(0, min(t, T - 1))] for t in range(start, end)])

        obs = {
            "image": images,             # (T, C, H, W) float32
            "agent_pos": states,          # (T, 8)       float32
        }
        if self.use_wrist_image:
            # wrist camera costs about as much decode time as "image" above —
            # skipped entirely (not just unused) when use_wrist_image=False.
            obs["wrist_image"] = _decode_video_frames(
                ep.video_paths["wrist_image"], frame_indices, self.image_size
            )

        data = {
            "obs": obs,
            "action": actions,                # (T, 7)       float32
        }
        return dict_apply(data, torch.from_numpy)
