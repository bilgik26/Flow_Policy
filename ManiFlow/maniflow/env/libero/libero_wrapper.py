"""
Thin wrapper around the LIBERO benchmark (built on robosuite) for Flow_Policy.

Modeled after maniflow/env/robocasa/robocasa_wrapper.py, but the environment
and observation/action details follow openvla-oft's LIBERO integration
(experiments/robot/libero/libero_utils.py, run_libero_eval.py):

Observation format returned by step() and reset():
  image:       (C, H, W) float32 [0, 1] — agentview camera, CHW, 180°-rotated
  wrist_image: (C, H, W) float32 [0, 1] — wrist camera,     CHW, 180°-rotated
  agent_pos:   (8,)      float32
    = eef_pos(3) + axis_angle(eef_quat)(3) + gripper_qpos(2)
    (matches openvla-oft's PROPRIO_DIM=8 proprio layout for LIBERO)

Action format accepted by step():
  action: (7,) float32 — EEF delta_pos(3) + delta_axis_angle_ori(3) + gripper(1)
    gripper: -1 = open, +1 = close (native robosuite/LIBERO OSC_POSE convention)

Unlike RoboCasa's PandaOmron composite-controller action space (12-dim, base
locked via zero-padding), LIBERO tasks use a fixed-base Panda arm with the
OSC_POSE controller, whose native action space IS the 7-dim delta-pose+gripper
vector — so it is passed to env.step() unchanged, no reshaping/padding needed.

Note on the gripper convention: openvla-oft applies a normalize+invert step to
the model's gripper output before calling env.step(), because *their* offline
RLDS dataloader flips the gripper sign during training-data construction. Here
we control both ends of the pipeline (see scripts/convert_libero_to_lerobot.py
and maniflow/dataset/libero_dataset.py), so the raw recorded action's gripper
sign is preserved end-to-end and no inversion hack is required.

Image 180° rotation: LIBERO/robosuite render agentview_image and
robot0_eye_in_hand_image upside down on this platform (same as openvla-oft's
comment "IMPORTANT: rotate 180 degrees to match train preprocessing"). The
transform `img[::-1, ::-1]` must be applied identically here AND in
maniflow/dataset/libero_dataset.py's `_decode_video_frames` — keep both in
sync.

Rendering: LIBERO's OffScreenRenderEnv defers to MuJoCo's configured GL
backend (MUJOCO_GL env var). On servers without /dev/dri render-group
permissions, set MUJOCO_GL=osmesa and PYOPENGL_PLATFORM=osmesa (same as the
RoboCasa setup — see docs/setup_and_train_libero.md).
"""

import pathlib
from typing import Optional

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Per-suite episode horizon, taken verbatim from openvla-oft's
# experiments/robot/libero/run_libero_eval.py::TASK_MAX_STEPS.
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

_AGENT_POS_DIM = 8  # eef_pos(3) + axis_angle(eef_quat)(3) + gripper_qpos(2)
_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # no-op, gripper open


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert an (x, y, z, w) quaternion to axis-angle exponential coordinates.
    Ported from robosuite.utils.transform_utils.quat2axisangle (also copied
    verbatim into openvla-oft's libero_utils.py) so this wrapper has no hard
    dependency on robosuite's internal module layout.
    """
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * np.arccos(quat[3])) / den).astype(np.float32)


class LiberoEnv:
    """
    Gym-like wrapper around a single LIBERO task environment.

    Parameters
    ----------
    task_suite_name : str
        One of ``"libero_spatial"``, ``"libero_object"``, ``"libero_goal"``,
        ``"libero_10"``, ``"libero_90"``.
    task_id : int
        Index of the task within the suite (``0 <= task_id < task_suite.n_tasks``).
    image_size : int
        Height and width of rendered images (both cameras).
    num_steps_wait : int
        Dummy no-op steps executed right after reset to let objects settle
        physically, matching openvla-oft's `num_steps_wait` convention.
    seed : int
        Env RNG seed. openvla-oft always uses a fixed seed (0) and varies the
        episode instead via `init_state_idx` in `reset()`, since "seed seems
        to affect object positions even when using fixed initial state" (see
        libero_utils.py comment) — pass a per-episode seed only if you
        intentionally want different physics per trial.
    """

    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        image_size: int = 256,
        num_steps_wait: int = 10,
        seed: int = 0,
    ):
        benchmark_dict = benchmark.get_benchmark_dict()
        if task_suite_name not in benchmark_dict:
            raise ValueError(
                f"Unknown LIBERO task suite: {task_suite_name!r}. "
                f"Available: {sorted(benchmark_dict.keys())}"
            )
        self.task_suite = benchmark_dict[task_suite_name]()
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.task = self.task_suite.get_task(task_id)
        self.task_description = self.task.language
        self.image_size = image_size
        self.num_steps_wait = num_steps_wait

        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / self.task.problem_folder
            / self.task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=image_size,
            camera_widths=image_size,
        )
        # IMPORTANT: seed affects object placement even when a fixed initial
        # state is set later via set_init_state() — see class docstring.
        self._env.seed(seed)

        self._init_states = self.task_suite.get_task_init_states(task_id)

    # ── Public interface ────────────────────────────────────────────────────

    def reset(self, init_state_idx: Optional[int] = None) -> dict:
        # NOTE: LIBERO's ControlEnv (libero.libero.envs.env_wrapper) has no
        # public get_observation() method — reset() itself already returns
        # the fresh obs dict (it wraps robosuite's own env.reset()), and
        # set_init_state() (when a specific init state is requested) returns
        # a regenerated obs dict from the teleported state. This matches how
        # libero/lifelong/evaluate.py and benchmark_scripts/render_single_task.py
        # use this API.
        raw = self._env.reset()
        if init_state_idx is not None and len(self._init_states) > 0:
            init_state = self._init_states[init_state_idx % len(self._init_states)]
            raw = self._env.set_init_state(init_state)

        # Let physics settle after teleporting to the initial state.
        for _ in range(self.num_steps_wait):
            raw, _, _, _ = self._env.step(_DUMMY_ACTION)

        return self._process_obs(raw)

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : (7,) float32 — EEF delta_pos(3) + delta_axis_angle_ori(3) + gripper(1)
            Passed to the LIBERO OSC_POSE controller unchanged.

        Returns
        -------
        obs, reward, done, info
            ``done`` (and ``info["success"]``) is True exactly when LIBERO's
            own BDDL goal check succeeds — the env does not auto-terminate on
            a step-limit, so the caller (env_runner) must enforce
            ``max_episode_steps`` itself.
        """
        a = np.asarray(action, dtype=np.float32)
        raw, reward, done, info = self._env.step(a.tolist())
        obs = self._process_obs(raw)
        info["success"] = bool(done)
        return obs, reward, done, info

    def close(self):
        self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return TASK_MAX_STEPS.get(self.task_suite_name, 400)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _process_obs(self, raw: dict) -> dict:
        img = self._to_chw(raw["agentview_image"])
        wrist_img = self._to_chw(raw["robot0_eye_in_hand_image"])

        eef_pos = np.asarray(
            raw.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)), dtype=np.float32
        )
        eef_quat = np.asarray(
            raw.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=np.float32
        )
        gripper_qpos = np.asarray(
            raw.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32)), dtype=np.float32
        )
        agent_pos = np.concatenate(
            [eef_pos, _quat2axisangle(eef_quat), gripper_qpos[:2]]
        )  # shape (8,)

        return {"image": img, "wrist_image": wrist_img, "agent_pos": agent_pos}

    @staticmethod
    def _to_chw(img: np.ndarray) -> np.ndarray:
        # 180° rotate — LIBERO/robosuite renders upside down on this platform.
        # Must match maniflow/dataset/libero_dataset.py's _decode_video_frames.
        img = img[::-1, ::-1].copy()
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0
