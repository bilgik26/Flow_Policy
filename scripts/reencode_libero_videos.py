#!/usr/bin/env python
"""
Re-encode an already-converted LIBERO LeRobot-format dataset's mp4 videos with
a shorter GOP (keyframe interval), without re-running the LIBERO simulator.

scripts/convert_libero_to_lerobot.py's dominant cost is MuJoCo simulation +
osmesa rendering (~40s/episode), not video encoding (~0.3s/video) — re-running
the full pipeline just to change GOP size would waste hours re-simulating
demos whose frames are already correctly rendered and saved. This script
instead reads each existing mp4 with decord and re-writes it with a new -g
(keyframe interval), reusing the source dataset's parquet/meta files unchanged
(action/state data is unaffected by video re-encoding).

See docs/setup_and_train_libero.md's decord section for why a shorter GOP
speeds up maniflow/dataset/libero_dataset.py's random-access reads.

Usage
-----
    python scripts/reencode_libero_videos.py \\
        --source_dir data/libero/datasets/lerobot/libero_spatial \\
        --target_dir data/libero/datasets/lerobot/libero_spatial_gop10 \\
        --gop_size 10
"""

import argparse
import json
import pathlib
import shutil
import time

import decord
import imageio


def _discover_task_dirs(source_dir: pathlib.Path):
    if (source_dir / "meta" / "info.json").exists():
        return [source_dir]
    return sorted({p.parent.parent for p in source_dir.glob("*/meta/info.json")})


def reencode_dataset(source_dir: pathlib.Path, target_dir: pathlib.Path, gop_size: int):
    task_dirs = _discover_task_dirs(source_dir)
    if not task_dirs:
        raise SystemExit(f"No LeRobot dataset found under {source_dir}")

    n_videos = 0
    t_start = time.time()
    for task_dir in task_dirs:
        rel = task_dir.relative_to(source_dir) if task_dir != source_dir else pathlib.Path(".")
        out_task_dir = target_dir / rel

        # meta/ (json) and data/ (parquet action+state) are unaffected by
        # video re-encoding — copy as-is instead of regenerating.
        for sub in ("meta", "data"):
            src_sub = task_dir / sub
            if src_sub.exists():
                shutil.copytree(src_sub, out_task_dir / sub, dirs_exist_ok=True)

        fps = json.loads((task_dir / "meta" / "info.json").read_text())["fps"]

        video_files = sorted((task_dir / "videos").rglob("*.mp4"))
        for src_video in video_files:
            rel_video = src_video.relative_to(task_dir)
            out_video = out_task_dir / rel_video
            out_video.parent.mkdir(parents=True, exist_ok=True)

            vr = decord.VideoReader(str(src_video))
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            with imageio.get_writer(
                out_video, fps=fps, format="mp4", output_params=["-g", str(gop_size)]
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)

            n_videos += 1
            if n_videos % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  {n_videos}/{len(video_files) * len(task_dirs)} (approx) videos re-encoded "
                      f"({elapsed:.0f}s elapsed)", flush=True)

        print(f"[{task_dir.name}] re-encoded {len(video_files)} videos -> {out_task_dir}", flush=True)

    print(f"Done. {n_videos} videos re-encoded -> {target_dir} ({time.time() - t_start:.0f}s total)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Existing converted LeRobot dataset dir.")
    parser.add_argument("--target_dir", required=True, help="Output dir for the re-encoded copy.")
    parser.add_argument("--gop_size", type=int, default=10, help="ffmpeg -g (max keyframe interval).")
    args = parser.parse_args()
    reencode_dataset(pathlib.Path(args.source_dir), pathlib.Path(args.target_dir), args.gop_size)


if __name__ == "__main__":
    main()
