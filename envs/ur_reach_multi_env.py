# Multi-arm parallel UR10 Reach Environment for Isaac Sim
# Uses Isaac Sim's Cloner to spawn N identical UR10 robots side-by-side
# for massively parallel RL data collection.

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class URReachMultiEnv(gym.Env):
    """Parallel UR10 reach task — N arms simulated simultaneously.

    Observations and actions are stacked: shape = (n_robots, obs_dim) and
    (n_robots, 6). Rewards and dones are per-robot vectors.

    For SB3 VecEnv compatibility wrap with ``IsaacVecEnvWrapper`` (see train_multi.py).
    """

    def __init__(
        self,
        n_robots: int = 4,
        render_mode: str = "human",
        action_scale: float = 0.3,
        target_radius: float = 0.05,
        reach_bonus: float = 10.0,
        max_episode_steps: int = 500,
        physics_dt: float = 1 / 500,
        rendering_dt: float = 1 / 50,
        spacing: float = 1.5,
    ):
        super().__init__()

        from isaacsim.core.api import World
        from isaacsim.core.cloner import GridCloner
        from isaacsim.core.utils.prims import define_prim
        from isaacsim.robot.manipulators.examples.universal_robots.ur10 import UR10
        from isaacsim.core.objects import DynamicSphere
        from isaacsim.storage.native import get_assets_root_path
        from pxr import UsdPhysics
        from isaacsim.core.utils.prims import get_prim_at_path

        self.n_robots = n_robots
        self._action_scale = action_scale
        self._target_radius = target_radius
        self._reach_bonus = reach_bonus
        self._max_episode_steps = max_episode_steps
        self._step_counts = np.zeros(n_robots, dtype=int)
        self.render_mode = render_mode

        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
        )

        assets_root_path = get_assets_root_path()
        ground = define_prim("/World/Ground", "Xform")
        ground.GetReferences().AddReference(
            assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
        )

        # Spawn template robot at /World/Robots/Robot_0
        self._world.scene.add(
            UR10(prim_path="/World/Robots/Robot_0", name="ur10_0", position=np.array([0.0, 0.0, 0.0]))
        )

        # Clone into a grid
        cloner = GridCloner(spacing=spacing)
        cloner.define_base_env("/World/Robots")
        self._robot_paths = cloner.generate_paths("/World/Robots/Robot", n_robots)
        cloner.clone(
            source_prim_path="/World/Robots/Robot_0",
            prim_paths=self._robot_paths,
        )

        # Collect robot handles
        self._robots = [
            self._world.scene.get_object(f"ur10_{i}") for i in range(n_robots)
        ]

        # Target spheres (one per robot)
        self._target_prims = []
        self._target_positions = np.zeros((n_robots, 3), dtype=np.float32)
        for i in range(n_robots):
            offset = np.array([i * spacing, 0.0, 0.0])
            sphere = self._world.scene.add(
                DynamicSphere(
                    prim_path=f"/World/Targets/Target_{i}",
                    name=f"target_{i}",
                    position=np.array([0.4, 0.0, 0.4]) + offset,
                    radius=0.03,
                    color=np.array([1.0, 0.0, 0.0]),
                )
            )
            # Disable physics on target
            prim = get_prim_at_path(f"/World/Targets/Target_{i}")
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            self._target_prims.append(sphere)
            self._target_positions[i] = np.array([0.4 + i * spacing, 0.0, 0.4])

        self._world.reset()

        # Spaces (single-robot dims; caller stacks across robots)
        n_joints = 6
        self._n_joints = n_joints
        obs_hi = np.concatenate([
            np.full(n_joints, np.pi, dtype=np.float32),   # qpos
            np.full(n_joints, np.pi, dtype=np.float32),   # qvel
            np.full(3, 2.0, dtype=np.float32),             # target
        ])
        self.observation_space = spaces.Box(
            low=np.tile(-obs_hi, (n_robots, 1)),
            high=np.tile(obs_hi, (n_robots, 1)),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_robots, n_joints), dtype=np.float32
        )

    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = []
        for i, robot in enumerate(self._robots):
            qpos = robot.get_joint_positions()[:self._n_joints].astype(np.float32)
            qvel = robot.get_joint_velocities()[:self._n_joints].astype(np.float32)
            obs.append(np.concatenate([qpos, qvel, self._target_positions[i]]))
        return np.stack(obs)

    def _random_target(self) -> np.ndarray:
        r = self.np_random.uniform(0.25, 0.55)
        theta = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        phi = self.np_random.uniform(0.15, np.pi / 2)
        return np.array(
            [r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._world.reset()
        self._step_counts[:] = 0

        for i, (robot, target_obj) in enumerate(zip(self._robots, self._target_prims)):
            offset = np.array([i * 1.5, 0.0, 0.0])
            local_target = self._random_target()
            self._target_positions[i] = local_target + offset
            target_obj.set_world_pose(position=self._target_positions[i])

        for _ in range(10):
            self._world.step(render=False)

        return self._get_obs(), {}

    def step(self, actions: np.ndarray):
        for i, robot in enumerate(self._robots):
            current = robot.get_joint_positions()[:self._n_joints]
            robot.set_joint_positions(current + actions[i] * self._action_scale)

        self._world.step(render=(self.render_mode == "human"))
        self._step_counts += 1

        obs = self._get_obs()
        rewards = np.zeros(self.n_robots, dtype=np.float32)
        terminated = np.zeros(self.n_robots, dtype=bool)
        truncated = np.zeros(self.n_robots, dtype=bool)
        infos = []

        for i, robot in enumerate(self._robots):
            ee_pos = robot.end_effector.get_world_pose()[0].astype(np.float32)
            dist = float(np.linalg.norm(ee_pos - self._target_positions[i]))
            reached = dist < self._target_radius
            rewards[i] = -dist + (self._reach_bonus if reached else 0.0)
            terminated[i] = reached
            truncated[i] = self._step_counts[i] >= self._max_episode_steps
            infos.append({"dist": dist, "reached": reached})

        return obs, rewards, terminated, truncated, infos

    def render(self):
        pass

    def close(self):
        self._world.stop()
