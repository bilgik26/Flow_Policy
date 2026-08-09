"""
Deterministic, seed-fixed task-level train/val/seen-eval split for LIBERO
multi-suite mixes (e.g. libero_all4).

Shared by maniflow/dataset/libero_dataset.py (which tasks' episodes go into
train vs val) and maniflow/env_runner/libero_runner.py (which task_ids to
roll out for the "seen"/"unseen" success-rate metrics), so both sides always
agree on the same split without needing to pass data between them -- given
the same (task_suite_name, n_tasks, val_tasks_per_suite, seen_tasks_per_suite,
split_seed), both independently compute identical task_id lists.

Seeded by `split_seed`, deliberately separate from `training.seed`, so the
split never changes across training runs that use a different --seed.
"""

import zlib
from typing import List, Tuple

import numpy as np


def _suite_rng(split_seed: int, task_suite_name: str, purpose: str) -> np.random.Generator:
    # zlib.crc32 (not Python's built-in hash()) so this is reproducible
    # across processes/runs -- str hash() is salted per-process unless
    # PYTHONHASHSEED is fixed. `purpose` decorrelates the train/val split's
    # RNG stream from the seen-eval-task selection's RNG stream.
    key = f"{task_suite_name}|{purpose}"
    return np.random.default_rng((split_seed, zlib.crc32(key.encode())))


def split_train_val_tasks(
    task_suite_name: str,
    n_tasks: int,
    val_tasks_per_suite: int,
    split_seed: int,
) -> Tuple[List[int], List[int]]:
    """Partition a suite's task_ids (0..n_tasks-1) into (train_task_ids,
    val_task_ids), holding out exactly `val_tasks_per_suite` whole tasks."""
    rng = _suite_rng(split_seed, task_suite_name, "train_val_split")
    perm = rng.permutation(n_tasks).tolist()
    val_task_ids = sorted(perm[:val_tasks_per_suite])
    train_task_ids = sorted(perm[val_tasks_per_suite:])
    return train_task_ids, val_task_ids


def select_seen_eval_tasks(
    task_suite_name: str,
    train_task_ids: List[int],
    seen_tasks_per_suite: int,
    split_seed: int,
) -> List[int]:
    """Pick a fixed `seen_tasks_per_suite`-sized subset of train_task_ids to
    roll out for the "seen" success-rate metric (trained-on tasks, probed
    live to see how well the policy reproduces trained behavior)."""
    rng = _suite_rng(split_seed, task_suite_name, "seen_eval_tasks")
    perm = rng.permutation(len(train_task_ids))
    n = min(seen_tasks_per_suite, len(train_task_ids))
    return sorted(train_task_ids[i] for i in perm[:n])
