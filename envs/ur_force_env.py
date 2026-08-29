# UR10 Force-Contact Reach Task — Isaac Sim RL/data-collection environment
# Gymnasium-compatible wrapper around Isaac Sim, extending the plain
# position-control reach task (see ../../isaacsim-ur-rl/envs/ur_reach_env.py)
# with wrist contact-force sensing and a real joint-impedance action mode.
#
# NOTE on ur_reach_env.py: that env drives joints via set_joint_positions(),
# which *teleports* the articulation to the target every step rather than
# commanding the drive through PhysX (see SingleArticulation.set_joint_positions
# docstring: "This method will immediately set (teleport) the affected
# joints... Use apply_action to control robot joints."). That's fine for a
# pure kinematic reach reward, but it means contact response and any
# force/impedance behavior built on top of it would not be physically
# meaningful. This env instead always drives the arm through
# get_articulation_controller().apply_action(...), so joints are actually
# integrated by PhysX and contact/impedance behavior is real.
#
# The Isaac Sim SimulationApp must be launched BEFORE importing this module.
# See run_policy_act.py / collect_data.py for the correct launch pattern.

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class URForceReachEnv(gym.Env):
    """UR10 end-effector reach/contact task in Isaac Sim, with wrist force sensing.

    Observation (21-dim):
        joint positions  (6)
        joint velocities (6)
        target xyz       (3)
        wrist wrench     (6)  -- [Fx, Fy, Fz, Tx, Ty, Tz]. Torque channels are
                                  zero in v1 (RigidContactView only reports net
                                  force); see docs on upgrading to a computed
                                  moment from get_contact_force_data().

    Action (6-dim): normalized joint position deltas in [-1, 1], scaled by
    ``action_scale`` (rad). Two ``control_mode``s:

    - ``"position"``: commands ``q_des = q + action*action_scale`` through the
      drive's own PD (``apply_action(ArticulationAction(joint_positions=...))``).
      When ``impedance_gain`` > 0, the effective scale is attenuated by
      measured contact force magnitude ("virtual compliance": soften the
      commanded step under load, but the underlying drive stiffness is
      whatever the USD asset's default Kp/Kd are).
    - ``"impedance"``: switches the articulation to PhysX "effort" mode (zeros
      the drive's internal Kp/Kd) and applies a manually computed joint torque
      ``tau = Kp*(q_des - q) + Kd*(0 - qdot)`` every step
      (``impedance_kp``/``impedance_kd``, clipped to ``max_effort``). This is
      genuine impedance control: compliance comes from choosing a soft Kp,
      not from post-hoc scaling of a position command.

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
        impedance_gain: float = 0.0,
        force_clip: float = 200.0,
        enable_camera: bool = False,
        camera_resolution: tuple = (224, 224),
        control_mode: str = "position",
        impedance_kp: float = 400.0,
        impedance_kd: float = 40.0,
        max_effort: float = 150.0,
        attach_gripper: bool = True,
    ):
        super().__init__()
        if control_mode not in ("position", "impedance"):
            raise ValueError(f"control_mode must be 'position' or 'impedance', got {control_mode!r}")

        # Deferred Isaac Sim imports (SimulationApp must already be running)
        from isaacsim.core.api import World
        from isaacsim.core.utils.prims import define_prim
        from isaacsim.core.api.sensors import RigidContactView
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators.examples.universal_robots.ur10 import UR10 as _UR10
        from isaacsim.storage.native import get_assets_root_path

        self._ArticulationAction = ArticulationAction  # stashed for use in step()

        class UR10(_UR10):
            # Upstream UR10.post_reset() calls self._gripper.post_reset()
            # unconditionally, which crashes when attach_gripper=False (the
            # default) since _gripper is None. Guard it here.
            def post_reset(self) -> None:
                super(_UR10, self).post_reset()
                self._end_effector.post_reset()
                if self._gripper is not None:
                    self._gripper.post_reset()

        self.render_mode = render_mode
        self._action_scale = action_scale
        self._target_radius = target_radius
        self._reach_bonus = reach_bonus
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self._impedance_gain = impedance_gain
        self._force_clip = force_clip
        self._enable_camera = enable_camera
        self._control_mode = control_mode
        self._impedance_kp = impedance_kp
        self._impedance_kd = impedance_kd
        self._max_effort = max_effort

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
        self._robot_root_path = "/World/UR10"
        self._robot = self._world.scene.add(
            UR10(
                prim_path=self._robot_root_path,
                name="ur10",
                position=np.array([0.0, 0.0, 0.0]),
                attach_gripper=attach_gripper,
            )
        )

        # Target visual marker (sphere, no physics)
        from isaacsim.core.api.objects import VisualSphere
        self._target_obj = self._world.scene.add(
            VisualSphere(
                prim_path="/World/Target",
                name="target",
                position=np.array([0.4, 0.0, 0.4]),
                radius=0.03,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )

        # --- Wrist contact-force sensor ------------------------------------
        # RigidContactView batches net contact force queries directly from
        # PhysX; preferred over the legacy ContactSensor schema-prim API
        # because it doesn't require per-instance USD sensor authoring and
        # vectorizes cleanly if this env is later cloned (GridCloner) for a
        # multi-arm variant, mirroring ur_reach_multi_env.py.
        #
        # NOTE: robot.prim_path is NOT the link's parent Xform -- for this
        # UR10 asset it resolves to ".../root_joint" (the articulation's fixed
        # joint), so the wrist link path is built from the literal Xform path
        # the robot was created under instead. filter_paths_expr must be a
        # concrete prim, not a broad recursive glob like "/World/**" -- that
        # matches the sensor's own body too and the physics tensor API
        # rejects it ("did not match the correct number of entries").
        # get_net_contact_forces() (used here) reports net force regardless
        # of the filter target; the filter mainly matters for the pairwise
        # get_contact_force_matrix()/get_contact_force_data() APIs.
        self._wrist_contact = RigidContactView(
            prim_paths_expr=f"{self._robot_root_path}/wrist_3_link",
            filter_paths_expr=["/World/Ground"],
        )

        # --- Optional wrist camera (for observation.images.wrist) ---------
        self._camera = None
        if self._enable_camera:
            from isaacsim.sensors.camera import Camera

            self._camera = Camera(
                prim_path=f"{self._robot_root_path}/wrist_3_link/wrist_camera",
                resolution=camera_resolution,
            )

        self._world.reset()
        self._wrist_contact.initialize()
        if self._camera is not None:
            self._camera.initialize()

        # --- Spaces -------------------------------------------------------
        self._n_joints = 6
        self._joint_indices = np.arange(self._n_joints)

        if self._control_mode == "impedance":
            # Zeros the drive's internal PD (Kp=Kd=0) so our manually computed
            # torque isn't fought by the position drive -- see
            # get_articulation_controller().apply_action() below. NOTE: the
            # high-level ArticulationController.switch_control_mode() wrapper
            # (unlike the lower-level articulation view it delegates to)
            # takes no joint_indices -- it applies to every DOF, gripper
            # included. We only ever send joint_efforts for the 6 arm joints
            # (via joint_indices in apply_action), so the now-driveless
            # gripper joints simply go passive; acceptable since this task
            # doesn't grasp anything yet.
            self._robot.get_articulation_controller().switch_control_mode("effort")
        obs_dim = self._n_joints * 2 + 3 + 6  # qpos + qvel + target xyz + wrench

        joint_pos_hi = np.array([np.pi] * self._n_joints, dtype=np.float32)
        joint_vel_hi = np.array([10.0] * self._n_joints, dtype=np.float32)
        wrench_hi = np.full(6, self._force_clip, dtype=np.float32)
        obs_hi = np.concatenate(
            [joint_pos_hi, joint_vel_hi, np.full(3, 2.0, dtype=np.float32), wrench_hi]
        )

        self.observation_space = spaces.Box(
            low=-obs_hi, high=obs_hi, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_joints,), dtype=np.float32
        )

        self._target_pos = np.array([0.4, 0.0, 0.4], dtype=np.float32)
        self._last_wrench = np.zeros(6, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_pos(self) -> np.ndarray:
        return self._robot.end_effector.get_world_pose()[0].astype(np.float32)

    def _get_wrench(self) -> np.ndarray:
        """Net force/torque on the wrist link, world frame. Torque channels
        are zero in v1 -- RigidContactView.get_net_contact_forces() only
        reports net force, not moment. See module docstring."""
        force = self._wrist_contact.get_net_contact_forces()
        force = np.asarray(force, dtype=np.float32).reshape(-1)[:3]
        force = np.clip(force, -self._force_clip, self._force_clip)
        wrench = np.zeros(6, dtype=np.float32)
        wrench[:3] = force
        self._last_wrench = wrench
        return wrench

    def _get_obs(self) -> np.ndarray:
        qpos = self._robot.get_joint_positions()[: self._n_joints].astype(np.float32)
        qvel = self._robot.get_joint_velocities()[: self._n_joints].astype(np.float32)
        wrench = self._get_wrench()
        obs = np.concatenate([qpos, qvel, self._target_pos, wrench])
        # Physics can produce transient outliers (contact events, drive
        # saturation) that briefly exceed the declared space bounds.
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_wrist_image(self):
        """RGB frame from the wrist camera, or None if disabled."""
        if self._camera is None:
            return None
        return self._camera.get_rgba()[:, :, :3]

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
        # world.reset() invalidates the physics simulation view, so sensor
        # views must be re-initialized after every reset, not just once in
        # __init__ (see RigidContactView.initialize()'s docstring: "needs to
        # be called after each hard reset").
        self._wrist_contact.initialize()
        if self._camera is not None:
            self._camera.initialize()
        self._step_count = 0

        self._target_pos = self._randomize_target()
        self._target_obj.set_world_pose(position=self._target_pos)

        for _ in range(10):  # settle
            self._world.step(render=False)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        current_pos = self._robot.get_joint_positions()[: self._n_joints]

        if self._control_mode == "impedance":
            current_vel = self._robot.get_joint_velocities()[: self._n_joints]
            q_des = current_pos + action * self._action_scale
            tau = self._impedance_kp * (q_des - current_pos) + self._impedance_kd * (0.0 - current_vel)
            tau = np.clip(tau, -self._max_effort, self._max_effort)
            self._robot.get_articulation_controller().apply_action(
                self._ArticulationAction(joint_efforts=tau, joint_indices=self._joint_indices)
            )
        else:
            # Optionally attenuate the commanded step by measured contact
            # force (virtual compliance). With impedance_gain == 0 this is
            # plain position control.
            if self._impedance_gain > 0.0:
                force_mag = float(np.linalg.norm(self._last_wrench[:3]))
                effective_scale = self._action_scale / (1.0 + self._impedance_gain * force_mag)
            else:
                effective_scale = self._action_scale
            target_pos = current_pos + action * effective_scale
            # apply_action (not set_joint_positions) so the command is
            # actually driven through PhysX rather than teleported.
            self._robot.get_articulation_controller().apply_action(
                self._ArticulationAction(joint_positions=target_pos, joint_indices=self._joint_indices)
            )

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

        info = {
            "dist": dist,
            "reached": reached,
            "wrench": self._last_wrench.copy(),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # rendering is driven by world.step when render_mode="human"

    def close(self):
        self._world.stop()
