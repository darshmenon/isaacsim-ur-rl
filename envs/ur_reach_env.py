# UR10 Reach Task — Isaac Sim RL Environment
# Gymnasium-compatible wrapper around Isaac Sim for SAC/PPO training.
#
# The Isaac Sim SimulationApp must be launched BEFORE importing this module.
# See train.py for the correct launch pattern.

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class URReachEnv(gym.Env):
    """UR10 end-effector reach task in Isaac Sim.

    Observation (15-dim):
        joint positions  (6)
        joint velocities (6)
        target xyz       (3)

    Action (6-dim):
        normalized joint position deltas in [-1, 1],
        scaled by ``action_scale`` (rad).

    Reward:
        -dist(ee, target) per step
        +reach_bonus when dist < target_radius
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str = "human",
        action_scale: float = 0.3,
        target_radius: float = 0.05,
        reach_bonus: float = 10.0,
        max_episode_steps: int = 500,
        physics_dt: float = 1 / 500,
        rendering_dt: float = 1 / 50,
    ):
        super().__init__()

        # Deferred Isaac Sim imports (SimulationApp must already be running)
        from isaacsim.core.api import World
        from isaacsim.core.utils.prims import define_prim
        from isaacsim.robot.manipulators.examples.universal_robots.ur10 import UR10
        from isaacsim.storage.native import get_assets_root_path

        self.render_mode = render_mode
        self._action_scale = action_scale
        self._target_radius = target_radius
        self._reach_bonus = reach_bonus
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # --- World & robot ------------------------------------------------
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
        )

        assets_root_path = get_assets_root_path()

        # Ground plane
        ground = define_prim("/World/Ground", "Xform")
        ground.GetReferences().AddReference(
            assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
        )

        # UR10 robot
        self._robot = self._world.scene.add(
            UR10(
                prim_path="/World/UR10",
                name="ur10",
                position=np.array([0.0, 0.0, 0.0]),
            )
        )

        # Target visual marker (sphere)
        from isaacsim.core.api.objects import DynamicSphere
        self._target_obj = self._world.scene.add(
            DynamicSphere(
                prim_path="/World/Target",
                name="target",
                position=np.array([0.4, 0.0, 0.4]),
                radius=0.03,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )
        # Make target kinematic (no physics, just visual)
        from pxr import UsdPhysics
        from isaacsim.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path("/World/Target")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)

        self._world.reset()

        # --- Spaces -------------------------------------------------------
        self._n_joints = 6
        obs_dim = self._n_joints * 2 + 3  # qpos + qvel + target xyz

        joint_pos_hi = np.array([np.pi] * self._n_joints, dtype=np.float32)
        joint_vel_hi = np.array([np.pi] * self._n_joints, dtype=np.float32)
        obs_hi = np.concatenate([joint_pos_hi, joint_vel_hi, np.full(3, 2.0, dtype=np.float32)])

        self.observation_space = spaces.Box(
            low=-obs_hi, high=obs_hi, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_joints,), dtype=np.float32
        )

        self._target_pos = np.array([0.4, 0.0, 0.4], dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_pos(self) -> np.ndarray:
        return self._robot.end_effector.get_world_pose()[0].astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = self._robot.get_joint_positions()[:self._n_joints].astype(np.float32)
        qvel = self._robot.get_joint_velocities()[:self._n_joints].astype(np.float32)
        return np.concatenate([qpos, qvel, self._target_pos])

    def _randomize_target(self) -> np.ndarray:
        r = self.np_random.uniform(0.25, 0.55)
        theta = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        phi = self.np_random.uniform(0.15, np.pi / 2)
        return np.array(
            [
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._world.reset()
        self._step_count = 0

        self._target_pos = self._randomize_target()
        self._target_obj.set_world_pose(position=self._target_pos)

        for _ in range(10):  # settle
            self._world.step(render=False)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Apply joint position delta
        current_pos = self._robot.get_joint_positions()[:self._n_joints]
        target_pos = current_pos + action * self._action_scale
        self._robot.set_joint_positions(target_pos)

        self._world.step(render=(self.render_mode == "human"))
        self._step_count += 1

        obs = self._get_obs()
        ee_pos = self._get_ee_pos()
        dist = float(np.linalg.norm(ee_pos - self._target_pos))

        reward = -dist
        reached = dist < self._target_radius
        if reached:
            reward += self._reach_bonus

        terminated = bool(reached)
        truncated = self._step_count >= self._max_episode_steps

        return obs, reward, terminated, truncated, {"dist": dist, "reached": reached}

    def render(self):
        pass  # rendering is driven by world.step when render_mode="human"

    def close(self):
        self._world.stop()
