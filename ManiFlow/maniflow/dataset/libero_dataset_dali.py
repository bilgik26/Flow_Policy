"""
NVIDIA DALI-based video loader for LiberoImageDataset (opt-in).

maniflow/dataset/libero_dataset.py decodes video on CPU via decord. Measured
on this dataset's GOP=10 re-encoded videos (see scripts/reencode_libero_videos.py),
DALI's hardware NVDEC path decodes the same random per-sample frame windows
~6x faster than decord (3.3 ms/sample vs 19.8 ms/sample, single stream,
256x256 frames) and runs entirely in the training process — no CPU worker
processes, so it also sidesteps the num_workers>1 host-RAM/swap-thrashing
issue documented in docs/setup_and_train_libero.md's マルチGPU学習 section.

Why this is a separate class instead of a LiberoImageDataset drop-in
---------------------------------------------------------------------
DALI's fn.readers.video reads a *static* file_list (one "path label
start_frame" line per sample) top-to-bottom / by shard, unlike a PyTorch
map-style Dataset's arbitrary __getitem__(idx). So LiberoDaliLoader does not
subclass BaseDataset/LiberoImageDataset or plug into torch.utils.data.DataLoader
-- it wraps an already-constructed LiberoImageDataset, reusing its episode
loading, train/val split, normalizer and action/state arrays completely
unchanged, and only replaces the *image decode* step with DALI. It exposes
its own minimal iterable interface (__iter__/__len__/set_epoch) that
train_maniflow_libero_workspace.py uses in place of a torch DataLoader when
cfg.dataloader.use_dali (or cfg.val_dataloader.use_dali) is true.

Padding semantics
------------------
LiberoImageDataset._Episode windows are built by clamping frame indices to
[0, T-1] (repeating the first/last frame near an episode's boundary -- see
LiberoImageDataset.__getitem__). DALI's file_list has no way to express "read
before frame 0" (a negative start_frame there means "Nth-from-end", not
pre-padding), so instead this loader asks DALI to decode only the real
in-bounds sub-range (`real_len` frames starting at `read_start`) and
reconstructs the exact same clamped/padded horizon-length window afterward
by repeating the first/last decoded frame -- see _pad_and_layout(). This was
verified to reproduce LiberoImageDataset's clamp formula exactly for both
leading- and trailing-pad cases.

NVDEC vs decord pixel values
------------------------------
NVDEC's hardware YUV->RGB conversion differs slightly from decord's software
(ffmpeg/libavcodec) path -- measured mean abs diff ~1.7/255, max ~12/255 on
this dataset. This is a standard hardware-decoder discrepancy, not a bug;
checkpoints trained with one decode backend remain usable with the other
(same normalizer, same shape_meta) but are not guaranteed bit-identical.
"""

import math
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
import torch

import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import feed_ndarray

from maniflow.dataset.libero_dataset import LiberoImageDataset


def _build_video_pipe(file_list_path, sequence_length, image_size, batch_size, num_threads, device_id, seed):
    @dali.pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed)
    def _pipe():
        frames, label = fn.readers.video(
            device="gpu",
            file_list=file_list_path,
            file_list_frame_num=True,
            sequence_length=sequence_length,
            random_shuffle=False,  # loader shuffles its own file_list row order (see _shard_indices_for_epoch)
            name="reader",
            pad_sequences=True,  # zero-fills sequences that would read past a video's last frame
        )
        # No-op when the source video already matches image_size (true for
        # scripts/convert_libero_to_lerobot.py / reencode_libero_videos.py
        # output); guards against a resolution mismatch otherwise.
        frames = fn.resize(frames, resize_x=float(image_size), resize_y=float(image_size), device="gpu")
        return frames, label

    pipe = _pipe()
    pipe.build()
    return pipe


class LiberoDaliLoader:
    """
    Iterable, DataLoader-like wrapper that decodes LiberoImageDataset's
    "image" (and optionally "wrist_image") video frames via DALI/NVDEC
    instead of decord, on a single GPU. One instance corresponds to one
    torch.utils.data.DataLoader in train_maniflow_libero_workspace.py's
    run() -- construct one for the train split and (on the main process
    only) one for the val split, same as today.

    Always drops a trailing partial batch (drop_last=True) for simplicity --
    DALI's video reader pads a batch by wrapping into the next epoch's data
    otherwise, which this class does not attempt to reproduce.

    Parameters
    ----------
    dataset : LiberoImageDataset
        Already-constructed dataset (or its `.get_validation_dataset()`);
        this loader reads `.horizon`, `.pad_before`, `.use_wrist_image`,
        `.image_size`, `._episodes`, `._samples` from it directly and does
        not otherwise duplicate LiberoImageDataset's loading/splitting logic.
    batch_size : int
        Per-GPU batch size (same meaning as DataLoader's batch_size under DDP).
    shuffle : bool
        Shuffle sample order each epoch (seeded by `seed + epoch`, see set_epoch).
    num_shards, shard_id : int
        DDP shard count / this rank's shard index. Both ranks compute the same
        seeded shuffle+pad independently (no cross-rank communication needed)
        so every shard has exactly the same number of batches -- required
        because train_maniflow_libero_workspace.py's DDP loop calls
        dist.all_reduce() once per batch, which deadlocks if ranks disagree on
        batch count.
    num_threads : int
        DALI CPU-side thread count (file I/O / container parsing), not decode
        (decode runs on the GPU's NVDEC unit). Small values (2-4) are enough.
    device_id : int
        CUDA device index for both decode and the returned tensors -- pass
        the same local rank used for the training process's GPU.
    """

    def __init__(
        self,
        dataset: LiberoImageDataset,
        batch_size: int,
        shuffle: bool = True,
        num_shards: int = 1,
        shard_id: int = 0,
        num_threads: int = 4,
        device_id: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.horizon = dataset.horizon
        self.pad_before = dataset.pad_before
        self.use_wrist_image = dataset.use_wrist_image
        self.image_size = dataset.image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_shards = max(1, num_shards)
        self.shard_id = shard_id
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed
        self._epoch = 0

        self._records = self._build_records()
        self._tmp_dir = tempfile.mkdtemp(prefix="dali_libero_")

    # ── Sample bookkeeping ──────────────────────────────────────────────────

    def _build_records(self) -> List[dict]:
        records = []
        for ep_i, center in self.dataset._samples:
            ep = self.dataset._episodes[ep_i]
            T = ep.length
            start = center - self.pad_before
            end = start + self.horizon
            read_start = max(0, start)
            pad_before_amt = max(0, -start)
            real_len = min(T, end) - read_start  # always >= 1, see module docstring's padding note
            records.append(
                {
                    "image_path": ep.video_paths["image"],
                    "wrist_path": ep.video_paths.get("wrist_image"),
                    "read_start": read_start,
                    "pad_before_amt": pad_before_amt,
                    "real_len": real_len,
                    "ep_i": ep_i,
                    "center": center,
                }
            )
        return records

    def _shard_indices_for_epoch(self, epoch: int) -> List[int]:
        n = len(self._records)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            perm = torch.randperm(n, generator=g).tolist()
        else:
            perm = list(range(n))
        if len(perm) % self.num_shards != 0:
            pad_n = self.num_shards - (len(perm) % self.num_shards)
            perm = perm + perm[:pad_n]
        shard = perm[self.shard_id :: self.num_shards]
        n_batches = len(shard) // self.batch_size
        return shard[: n_batches * self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._shard_indices_for_epoch(self._epoch)) // self.batch_size

    # ── Iteration ────────────────────────────────────────────────────────────

    def _write_file_list(self, path: str, shard: List[int], key: str) -> None:
        # end_frame must be given explicitly and must not exceed the video's
        # own frame count: without it, fn.readers.video reads from
        # read_start all the way to the end of the file as *multiple*
        # sequence_length-sized chunks (repeating this row's label once per
        # chunk) instead of the single window we want; pad_sequences=True
        # then fills the [real_len, horizon) tail of that single window.
        with open(path, "w") as f:
            for i, rec_i in enumerate(shard):
                r = self._records[rec_i]
                end = r["read_start"] + r["real_len"]
                f.write(f"{r[key]} {i} {r['read_start']} {end}\n")

    def _pad_and_layout(self, frames_gpu: torch.Tensor, shard: List[int], batch_start: int) -> torch.Tensor:
        """
        frames_gpu: (B, F, H, W, C) uint8, F == self.horizon, row j holds
        `real_len` real decoded frames starting at slot 0 followed by
        unwanted filler (see module docstring). Returns (B, F, C, H, W)
        float32 in [0, 1], 180-rotated (matches libero_dataset.py's
        img[::-1, ::-1] convention for these upside-down source videos).
        """
        B = frames_gpu.shape[0]
        out_rows = []
        for j in range(B):
            rec = self._records[shard[batch_start + j]]
            pad_before_amt = rec["pad_before_amt"]
            real_len = rec["real_len"]
            pad_after_amt = self.horizon - pad_before_amt - real_len
            if pad_before_amt == 0 and pad_after_amt == 0:
                out_rows.append(frames_gpu[j, : self.horizon])
                continue
            real = frames_gpu[j, :real_len]
            parts = []
            if pad_before_amt > 0:
                parts.append(real[0:1].expand(pad_before_amt, -1, -1, -1))
            parts.append(real)
            if pad_after_amt > 0:
                parts.append(real[-1:].expand(pad_after_amt, -1, -1, -1))
            out_rows.append(torch.cat(parts, dim=0))
        frames = torch.stack(out_rows, dim=0)  # (B, F, H, W, C) uint8
        frames = frames.flip(dims=(2, 3))  # 180 rotate (H, W)
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()  # -> (B, F, C, H, W)
        return frames.float() / 255.0

    def _gather_action_state(self, shard: List[int], batch_start: int):
        actions_batch = []
        states_batch = []
        for j in range(self.batch_size):
            rec = self._records[shard[batch_start + j]]
            ep = self.dataset._episodes[rec["ep_i"]]
            T = ep.length
            start = rec["center"] - self.pad_before
            end = start + self.horizon
            frame_ts = [max(0, min(t, T - 1)) for t in range(start, end)]
            actions_batch.append(np.stack([ep.actions[t] for t in frame_ts]))
            states_batch.append(np.stack([ep.states[t] for t in frame_ts]))
        actions = torch.from_numpy(np.stack(actions_batch, axis=0))
        states = torch.from_numpy(np.stack(states_batch, axis=0))
        return actions, states

    def __iter__(self):
        shard = self._shard_indices_for_epoch(self._epoch)
        n_batches = len(shard) // self.batch_size

        image_list_path = os.path.join(self._tmp_dir, f"image_shard{self.shard_id}.txt")
        self._write_file_list(image_list_path, shard, "image_path")
        image_pipe = _build_video_pipe(
            image_list_path, self.horizon, self.image_size, self.batch_size,
            self.num_threads, self.device_id, self.seed + self._epoch,
        )

        wrist_pipe = None
        if self.use_wrist_image:
            wrist_list_path = os.path.join(self._tmp_dir, f"wrist_shard{self.shard_id}.txt")
            self._write_file_list(wrist_list_path, shard, "wrist_path")
            wrist_pipe = _build_video_pipe(
                wrist_list_path, self.horizon, self.image_size, self.batch_size,
                self.num_threads, self.device_id, self.seed + self._epoch,
            )

        device = torch.device(f"cuda:{self.device_id}")
        raw_shape = (self.batch_size, self.horizon, self.image_size, self.image_size, 3)
        image_buf = torch.empty(raw_shape, dtype=torch.uint8, device=device)
        wrist_buf = torch.empty(raw_shape, dtype=torch.uint8, device=device) if wrist_pipe else None

        for b in range(n_batches):
            batch_start = b * self.batch_size

            img_frames, img_labels = image_pipe.run()
            feed_ndarray(img_frames.as_tensor(), image_buf)
            labels = img_labels.as_cpu().as_array().reshape(-1)
            assert (labels == np.arange(batch_start, batch_start + self.batch_size)).all(), (
                "DALI image reader row order drifted from the expected file_list order"
            )
            image = self._pad_and_layout(image_buf, shard, batch_start)

            obs: Dict[str, torch.Tensor] = {"image": image}
            if wrist_pipe is not None:
                w_frames, w_labels = wrist_pipe.run()
                feed_ndarray(w_frames.as_tensor(), wrist_buf)
                w_labels_np = w_labels.as_cpu().as_array().reshape(-1)
                assert (w_labels_np == labels).all(), "image/wrist_image row misalignment"
                obs["wrist_image"] = self._pad_and_layout(wrist_buf, shard, batch_start)

            actions, states = self._gather_action_state(shard, batch_start)
            obs["agent_pos"] = states
            # Not tensorized -- mirrors LiberoImageDataset.__getitem__'s
            # "task_name" (see its docstring); language_conditioned policies
            # read this as a (B,)-length list of strings.
            obs["task_name"] = [
                self.dataset._episodes[self._records[shard[batch_start + j]]["ep_i"]].language_instruction
                for j in range(self.batch_size)
            ]
            # Mirrors LiberoImageDataset.__getitem__'s "task_suite_name" --
            # lets a multi-suite mix like libero_all4's val loop bucket
            # per-sample loss by suite even when DALI is used for val_dataloader.
            obs["task_suite_name"] = [
                self.dataset._episodes[self._records[shard[batch_start + j]]["ep_i"]].task_suite_name
                for j in range(self.batch_size)
            ]
            yield {"obs": obs, "action": actions}
