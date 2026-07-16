"""
Thin wrapper around robosuite / robocasa environments for Flow_Policy.

Observation format returned by step() and reset():
  image:     (C, H, W) float32 [0, 1]  — left agentview camera, CHW, flipped right-side up
  agent_pos: (9,)      float32
    = eef_pos_relative(3) + eef_rot_relative(4) + gripper_qpos(2)
    → LeRobot observation.state[7:16] と整合
      (robot0_base_to_eef_pos + robot0_base_to_eef_quat + robot0_gripper_qpos)

Action format accepted by step():
  action: (7,) float32 — EEF delta_pos(3) + delta_ori(3) + gripper(1)
    → LeRobot action[5:12] に相当

The wrapper places the 7-dim EEF action into the correct positions of the
12-dim HYBRID_MOBILE_BASE action (base locked). This is the RAW robosuite
composite-controller action order (confirmed via
CompositeController.print_action_info() on a live env: right=0:6,
right_gripper=6:7, base=7:10, torso=10:11, mode=11:12), which is DIFFERENT
from the "LeRobot reordered" layout used by the offline dataset files
(see robocasa_dataset.py / docs/investigation_and_verification.md):
  [0:3]   EEF delta pos  ← policy action[0:3]
  [3:6]   EEF delta ori  ← policy action[3:6]
  [6:7]   gripper        ← policy action[6]
  [7:10]  base_motion = [0, 0, 0]   (locked, JOINT_VELOCITY control_dim=3)
  [10:11] torso = [0]               (neutral, unused by policy)
  [11:12] mode = [-1]               (locked base mode)

Controller config: loaded from the bundled robocasa_controller_configs.pkl.

EGL rendering: set MUJOCO_GL=egl and MUJOCO_EGL_DEVICE_ID=3 when /dev/dri
is not accessible (Mesa software EGL fallback).
"""

import pathlib
import pickle

import numpy as np
import robocasa  # must be imported before robosuite.make to register robocasa envs
import robosuite

_PKL_PATH = pathlib.Path(__file__).parent / "robocasa_controller_configs.pkl"

# Raw robosuite HYBRID_MOBILE_BASE action layout (right=0:6, right_gripper=6:7,
# base=7:10, torso=10:11, mode=11:12), NOT the "LeRobot reordered" layout used
# by the offline dataset files.
# 7-dim policy action → 12-dim env action mapping:
#   env[0:3]   = policy[0:3]  (EEF delta pos)
#   env[3:6]   = policy[3:6]  (EEF delta ori)
#   env[6]     = policy[6]    (gripper)
#   env[7:11]  = base+torso locked: [0, 0, 0, 0]
#   env[11]    = -1            (mode: base locked)

# 9-dim agent_pos: first 9 dims of robot0_proprio-state
# These correspond to joint/gripper/EEF state used by cosmos-policy for normalization.
_AGENT_POS_DIM = 9


class RoboCasaEnv:
    """
    Gym-like wrapper around a robosuite/robocasa environment.

    Parameters
    ----------
    task_name : str
        RoboCasa task name (e.g. ``"TurnOffMicrowave"``).
    camera_name : str
        Camera feature name without ``_image`` suffix.
    image_size : int
        Height and width of rendered images.
    layout_id : int or None
        Kitchen layout ID. ``None`` → random.
    style_id : int or None
        Kitchen style ID. ``None`` → random.
    obj_instance_split : str
        ``"target"`` for held-out test objects, ``"pretrain"`` for training objects.
    seed : int
        Random seed (passed as env seed if supported).
    """

    def __init__(
        self,
        task_name: str,
        camera_name: str = "robot0_agentview_left",
        image_size: int = 224,
        layout_id=None,
        style_id=None,
        obj_instance_split: str = "target",
        seed: int = 0,
    ):
        self.task_name = task_name
        self.camera_name = camera_name
        self.image_size = image_size
        self.obj_instance_split = obj_instance_split
        self._image_key = f"{camera_name}_image"

        with open(_PKL_PATH, "rb") as f:
            controller_cfg = pickle.load(f)

        kwargs = dict(
            robots="PandaOmron",   # dataset uses PandaOmron (Panda arm + Omron AMR base)
            controller_configs=controller_cfg,
            camera_names=[camera_name],
            camera_heights=image_size,
            camera_widths=image_size,
            reward_shaping=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            translucent_robot=False,
            obj_instance_split=obj_instance_split,
        )
        if layout_id is not None:
            kwargs["layout_ids"] = layout_id
        if style_id is not None:
            kwargs["style_ids"] = style_id

        self._env = robosuite.make(task_name, **kwargs)
        self._seed = seed

    # ── Public interface ────────────────────────────────────────────────────

    def reset(self) -> dict:
        raw = self._env.reset()
        return self._process_obs(raw)

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : (7,) float32 — EEF delta_pos(3) + delta_ori(3) + gripper(1)
            policy action[0:3] → env EEF pos  [0:3]
            policy action[3:6] → env EEF ori  [3:6]
            policy action[6]   → env gripper  [6]
            env base+torso [7:11] は [0,0,0,0] で固定（台車ロック）
            env mode [11] は -1 固定（台車ロックモード）

        Note: this is the RAW robosuite HYBRID_MOBILE_BASE action order
        (right=0:6, right_gripper=6:7, base=7:10, torso=10:11, mode=11:12),
        confirmed via CompositeController.print_action_info() on a live env.
        It is NOT the same as the "LeRobot reordered" layout used by the
        offline dataset files (base first) — do not conflate the two.
        """
        a = action.astype(np.float32)
        full_action = np.zeros(12, dtype=np.float32)
        full_action[0:3] = a[0:3]   # EEF delta pos
        full_action[3:6] = a[3:6]   # EEF delta ori
        full_action[6] = a[6]       # gripper
        full_action[11] = -1.0      # mode: base locked
        raw, reward, done, info = self._env.step(full_action)
        obs = self._process_obs(raw)
        success = bool(self._env._check_success())
        info["success"] = success
        return obs, reward, done or success, info

    def close(self):
        self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._env.horizon

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _process_obs(self, raw: dict) -> dict:
        img = raw[self._image_key]          # (H, W, 3) uint8, upside-down in robocasa
        img = np.flipud(img).copy()
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # → (C, H, W) [0,1]

        # 9-dim agent_pos = eef_pos_relative(3) + eef_rot_relative(4) + gripper_qpos(2)
        # LeRobot observation.state[7:16] に整合
        # (LEROBOT_STATE_TO_HDF5_STATE の定義より:
        #   robot0_base_to_eef_pos     → state[7:10]
        #   robot0_base_to_eef_quat    → state[10:14]
        #   robot0_gripper_qpos        → state[14:16])
        eef_pos_rel = raw.get("robot0_base_to_eef_pos", np.zeros(3, dtype=np.float32))
        eef_rot_rel = raw.get("robot0_base_to_eef_quat", np.array([0., 0., 0., 1.], dtype=np.float32))
        gripper_qpos = raw.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
        agent_pos = np.concatenate([
            eef_pos_rel.astype(np.float32),
            eef_rot_rel.astype(np.float32),
            gripper_qpos[:2].astype(np.float32),
        ])  # shape (9,)

        return {"image": img, "agent_pos": agent_pos}
