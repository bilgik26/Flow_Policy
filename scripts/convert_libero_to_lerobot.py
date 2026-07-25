#!/usr/bin/env python
"""
Convert raw LIBERO HDF5 demonstrations into the LeRobot v2.1 format consumed
by maniflow/dataset/libero_dataset.py (parquet actions/states + per-camera
mp4 videos) — the same on-disk convention Flow_Policy already uses for
RoboCasa (see maniflow/dataset/robocasa_dataset.py).

This mirrors openvla-oft's dataset-regeneration approach
(experiments/robot/libero/regenerate_libero_dataset.py): raw LIBERO HDF5 demos
store 128x128 images and don't include the wrist camera at usable resolution,
so each demo is *replayed* in a fresh OffScreenRenderEnv (teleported to the
demo's first sim state, then stepped forward through the demo's own recorded
actions) to (a) re-render images at a configurable resolution and (b) recover
consistent proprioceptive state at every step. No-op actions are filtered out
(is_noop) and only demos that replay to a successful termination are kept —
the same two data-cleaning rules openvla-oft applies.

Output layout (one LeRobot dataset directory per LIBERO task, auto-discovered
by maniflow/dataset/libero_dataset.py::_discover_lerobot_dirs):

  {libero_target_dir}/{task.name}/
    meta/
      info.json
      tasks.jsonl
      episodes.jsonl
    data/chunk-000/episode_NNNNNN.parquet       # columns: action(7), observation.state(8)
    videos/chunk-000/observation.images.image/episode_NNNNNN.mp4        # agentview, RAW orientation
    videos/chunk-000/observation.images.wrist_image/episode_NNNNNN.mp4  # wrist,     RAW orientation

Videos are written in the raw (upside-down) orientation the LIBERO/robosuite
renderer produces — the 180° rotation is applied at *read* time by both
libero_dataset.py's _decode_video_frames and libero_wrapper.py's
_process_obs, so training data and online observations stay consistent.
Keep all three in sync if you ever change this convention.

Usage
-----
    python scripts/convert_libero_to_lerobot.py \\
        --libero_task_suite libero_spatial \\
        --libero_raw_data_dir /path/to/Flow_Policy/libero/libero/libero/datasets/libero_spatial \\
        --libero_target_dir  /path/to/Flow_Policy/data/libero/datasets/lerobot/libero_spatial

Raw HDF5 datasets are obtained via LIBERO's own dataset download utility —
see docs/setup_and_train_libero.md (Step 4) for the exact, verified command
(`benchmark_scripts/download_libero_datasets.py --download-dir ... --use-huggingface`).
"""

import argparse
import json
import math
import pathlib

import h5py
import imageio
import numpy as np
import pandas as pd

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
NUM_STEPS_WAIT = 10  # let objects settle after teleporting to the demo's initial state


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Ported from robosuite.utils.transform_utils.quat2axisangle (xyzw quat)."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * np.arccos(quat[3])) / den).astype(np.float32)


def is_noop(action: np.ndarray, prev_action: np.ndarray = None, threshold: float = 1e-4) -> bool:
    """
    An action is a no-op if the EEF pose delta is (near) zero AND the gripper
    command didn't change from the previous step. Checking both — not just
    delta magnitude — avoids stripping legitimate "hold still while the
    gripper opens/closes" transitions. Ported from openvla-oft's
    regenerate_libero_dataset.py.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def get_task_env(task, image_size: int) -> OffScreenRenderEnv:
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=image_size,
        camera_widths=image_size,
    )
    env.seed(0)
    return env


def replay_demo(env: OffScreenRenderEnv, init_state: np.ndarray, raw_actions: np.ndarray):
    """
    Replay one demo's recorded actions in `env`, starting from `init_state`.

    Returns (success, actions, states, agentview_frames, wrist_frames) where
    `actions`/`states` are the per-kept-step (7,)/(8,) arrays and the frame
    lists hold raw (H, W, 3) uint8 images, or None if the demo produced no
    steps at all.
    """
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    actions, states = [], []
    agentview_frames, wrist_frames = [], []

    prev_action = None
    done = False
    for raw_action in raw_actions:
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if is_noop(raw_action, prev_action):
            continue
        prev_action = raw_action

        agentview_frames.append(obs["agentview_image"])
        wrist_frames.append(obs["robot0_eye_in_hand_image"])
        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
                np.asarray(obs["robot0_gripper_qpos"][:2], dtype=np.float32),
            ]
        )
        states.append(state)
        actions.append(raw_action)

        obs, reward, done, info = env.step(raw_action.tolist())
        if done:
            break

    if len(actions) == 0:
        return False, None, None, None, None
    return done, np.stack(actions), np.stack(states), agentview_frames, wrist_frames


def write_episode(
    task_dir: pathlib.Path,
    ep_idx: int,
    actions: np.ndarray,
    states: np.ndarray,
    agentview_frames,
    wrist_frames,
    fps: int,
    chunks_size: int,
    gop_size: int = None,
):
    chunk = ep_idx // chunks_size
    ch_str = f"chunk-{chunk:03d}"
    ep_str = f"episode_{ep_idx:06d}"

    data_dir = task_dir / "data" / ch_str
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "action": list(actions),
            "observation.state": list(states),
        }
    )
    df.to_parquet(data_dir / f"{ep_str}.parquet")

    # gop_size=None keeps ffmpeg's default GOP (effectively one keyframe per
    # clip at this length/fps), matching the dataset's original encoding.
    # Passing a small value (e.g. close to horizon) trades a bit of file size
    # for much cheaper random-access seeks in maniflow/dataset/libero_dataset.py's
    # decord-based reader — see docs/setup_and_train_libero.md's decord section.
    output_params = ["-g", str(gop_size)] if gop_size else None
    for key, frames in (("image", agentview_frames), ("wrist_image", wrist_frames)):
        video_dir = task_dir / "videos" / ch_str / f"observation.images.{key}"
        video_dir.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            video_dir / f"{ep_str}.mp4", fps=fps, format="mp4", output_params=output_params
        ) as writer:
            for frame in frames:
                writer.append_data(frame)


def convert_task(
    task,
    task_id: int,
    raw_data_dir: pathlib.Path,
    target_dir: pathlib.Path,
    image_size: int,
    fps: int,
    chunks_size: int = 1000,
    max_demos: int = None,
    gop_size: int = None,
):
    hdf5_path = raw_data_dir / f"{task.name}_demo.hdf5"
    if not hdf5_path.exists():
        print(f"[WARNING] Skipping task {task_id} ({task.name}): {hdf5_path} not found", flush=True)
        return 0

    task_dir = target_dir / task.name
    (task_dir / "meta").mkdir(parents=True, exist_ok=True)

    env = get_task_env(task, image_size)
    n_saved = 0
    n_total = 0
    episodes_meta = []

    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                f["data"].keys(), key=lambda k: int(k.replace("demo_", ""))
            )
            if max_demos is not None:
                demo_keys = demo_keys[:max_demos]
            n_total = len(demo_keys)
            for demo_i, demo_key in enumerate(demo_keys):
                grp = f[f"data/{demo_key}"]
                orig_states = grp["states"][()]
                orig_actions = grp["actions"][()]

                success, actions, states, agentview_frames, wrist_frames = replay_demo(
                    env, orig_states[0], orig_actions
                )
                if not success:
                    print(f"  [{task.name}] {demo_key} ({demo_i + 1}/{n_total}): replay FAILED, skipped", flush=True)
                    continue

                write_episode(
                    task_dir, n_saved, actions, states,
                    agentview_frames, wrist_frames, fps, chunks_size,
                    gop_size=gop_size,
                )
                episodes_meta.append({"episode_index": n_saved, "length": int(len(actions)), "task_index": task_id})
                n_saved += 1
                print(f"  [{task.name}] {demo_key} ({demo_i + 1}/{n_total}): OK, saved as episode_{n_saved - 1:06d} "
                      f"(len={len(actions)})", flush=True)
    finally:
        env.close()

    with open(task_dir / "meta" / "episodes.jsonl", "w") as f:
        for meta in episodes_meta:
            f.write(json.dumps(meta) + "\n")

    with open(task_dir / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": task_id, "task": task.language}) + "\n")

    with open(task_dir / "meta" / "info.json", "w") as f:
        json.dump(
            {
                "chunks_size": chunks_size,
                "fps": fps,
                "total_episodes": n_saved,
                "task_name": task.name,
                "task_description": task.language,
            },
            f,
            indent=2,
        )

    print(f"[{task.name}] saved {n_saved}/{n_total} successful demos -> {task_dir}", flush=True)
    return n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument(
        "--libero_raw_data_dir", required=True,
        help="Directory containing the raw *_demo.hdf5 files for this suite.",
    )
    parser.add_argument(
        "--libero_target_dir", required=True,
        help="Output directory for the converted LeRobot-format dataset "
             "(pass this as task.dataset_base_path in the hydra config).",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--task_ids", type=int, nargs="*", default=None,
        help="Optional subset of task indices to convert (default: all tasks in the suite).",
    )
    parser.add_argument(
        "--max_demos_per_task", type=int, default=None,
        help="Optional cap on the number of raw demos replayed per task "
             "(default: all demos in the HDF5 file). Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--gop_size", type=int, default=None,
        help="Optional ffmpeg -g (max keyframe interval) for the output mp4s. "
             "Default (unset) uses ffmpeg's default, which for these short "
             "clips means effectively one keyframe per episode. A small value "
             "(e.g. close to horizon) makes maniflow/dataset/libero_dataset.py's "
             "decord-based random-access reads cheaper at the cost of file size.",
    )
    args = parser.parse_args()

    task_suite = benchmark.get_benchmark_dict()[args.libero_task_suite]()
    task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))

    raw_data_dir = pathlib.Path(args.libero_raw_data_dir)
    target_dir = pathlib.Path(args.libero_target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for i, task_id in enumerate(task_ids):
        task = task_suite.get_task(task_id)
        print(f"=== Task {i + 1}/{len(task_ids)} (task_id={task_id}): {task.name} ===", flush=True)
        total += convert_task(
            task, task_id, raw_data_dir, target_dir, args.image_size, args.fps,
            max_demos=args.max_demos_per_task, gop_size=args.gop_size,
        )

    print(f"Done. {total} total episodes converted to {target_dir}", flush=True)


if __name__ == "__main__":
    main()
