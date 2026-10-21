"""
SO-101 push environment: MuJoCo-backed drop-in for the PushTWrapper contract.

Registered as gym id "so101_pushp" — deliberately NOT "pusht"/"pushobj", because
plan.py:446-447 divides actions by 100 for those names (their data is pixel-space).
Ours is metric: action = commanded planar EE delta (dx, dy) in METRES, action_scale 1.

State layout (with_velocity=True), mirroring pusht_env._get_obs:
    [ee_x, ee_y, obj_x, obj_y, obj_yaw, ee_vx, ee_vy]      float32, metres/radians
proprio = state[:2] (+) state[5:]  ->  4-dim.

obs = {"visual": (H, W, 3) uint8, "proprio": (4,)} — same as PushTEnv.reset/step.

Success (eval_state) is POSITION-ONLY: block centroid within `success_radius` of the
goal centroid, orientation ignored (project spec). Note the asymmetry: the planner
minimises distance to a goal *embedding*, which encodes orientation, so CEM/GD
implicitly chase rotation regardless. Expect successes with the P visibly rotated.

Physics/IK constants MUST match collect_data.py in the SO-101 scene repo — the
training data and the eval env have to be the same MDP. If you change one, change
both (they are duplicated deliberately: the two files live in different repos).

The renderer is created lazily on first use, never in __init__ — SubprocVectorEnv
forks workers, and an EGL context created before the fork is invalid in the child.
"""

import os

import gym
import numpy as np
from gym import spaces

# --- constants shared with collect_data.py (keep in sync!) -------------------
RENDER_SIZE = 224
STEPS_PER_ACTION = 25        # 25 * 0.002s = 0.05s => 20 Hz control
SETTLE_STEPS = 400
BLOCK_REST_Z = 0.020
PUSHER_Z = 0.018
WS_X = (-0.15, 0.15)
WS_Y = (0.23, 0.42)
HOME_CTRL = np.array([0.0, -1.5, 1.5, 0.5, 0.0, 0.0])
ARM_DOF = 6
IK_JOINTS = np.array([0, 1, 2, 3])
WRIST_ROLL_FIXED = 0.0
GRIPPER_CLOSED = -0.17453
MAX_STEP = 0.015


def _aggregate_dct(dcts):
    """np-only stand-in for utils.aggregate_dct (avoids importing torch here)."""
    full_dct = {}
    for dct in dcts:
        for k, v in dct.items():
            full_dct.setdefault(k, []).append(v)
    return {k: np.stack(v) for k, v in full_dct.items()}


class _Arm:
    """Damped-least-squares position IK on the `gripperframe` site (4 joints;
    wrist_roll/gripper pinned so the pusher is one rigid shape). Parameters tuned
    on a 7x7 workspace grid: 49/49 convergence, 0.046mm median error."""

    def __init__(self, model, data):
        import mujoco
        self.mj = mujoco
        self.m, self.d = model, data
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        if self.sid < 0:
            raise RuntimeError("site 'gripperframe' not found in scene")
        self.lo = model.jnt_range[:ARM_DOF, 0].copy()
        self.hi = model.jnt_range[:ARM_DOF, 1].copy()

    def site_pos(self):
        return self.d.site_xpos[self.sid].copy()

    def solve(self, target, q0, iters=300, tol=1e-3, damp=1e-4, k_null=0.005):
        q = np.clip(q0.copy(), self.lo, self.hi)
        saved = self.d.qpos[:ARM_DOF].copy()
        J = np.zeros((3, self.m.nv))
        ok = False
        for _ in range(iters):
            self.d.qpos[:ARM_DOF] = q
            self.mj.mj_forward(self.m, self.d)
            err = target - self.d.site_xpos[self.sid]
            if np.linalg.norm(err) < tol:
                ok = True
                break
            self.mj.mj_jacSite(self.m, self.d, J, None, self.sid)
            Ja = J[:, IK_JOINTS]
            JJt = Ja @ Ja.T + damp * np.eye(3)
            dq = Ja.T @ np.linalg.solve(JJt, err)
            P = np.eye(len(IK_JOINTS)) - Ja.T @ np.linalg.solve(JJt, Ja)
            q[IK_JOINTS] += dq + P @ (k_null * (HOME_CTRL[IK_JOINTS] - q[IK_JOINTS]))
            q[4] = WRIST_ROLL_FIXED
            q[5] = GRIPPER_CLOSED
            q = np.clip(q, self.lo, self.hi)
        self.d.qpos[:ARM_DOF] = saved
        self.mj.mj_forward(self.m, self.d)
        return q, ok


class SO101PushEnv(gym.Env):
    """Core MuJoCo env. PushTEnv-shaped API: reset() -> (obs, state),
    step(a) -> (obs, reward, done, info with info['state'])."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, with_velocity=True, with_target=True, scene_path=None,
                 success_radius=0.025, success_angle=None, friction_scale=1.0, mass_scale=1.0, **kwargs):
        super().__init__()
        # Dynamics-shift knobs for OOD eval, mirroring how the maze env scales
        # density/damping (env/pointmaze/maze_model.py:33-41). Applied to the
        # block's geoms after model load, so training data is unaffected.
        self.friction_scale = float(friction_scale)
        self.mass_scale = float(mass_scale)
        # colors/shape kwargs from update_env/eval overrides are accepted but inert
        self.with_velocity = with_velocity
        self.with_target = with_target          # unused; kept for config parity
        self.success_radius = success_radius
        # None = position-only (original). A float adds an orientation tolerance in
        # radians, mirroring PushT's criterion (pos AND angle must both pass).
        self.success_angle = None if success_angle is None else float(success_angle)
        self.scene_path = (scene_path
                           or os.environ.get("SO101_SCENE")
                           or "/so101/scene.xml")
        self.relative = True
        self.action_scale = 1.0                 # metric actions; NEVER 100
        self.shape = "P"

        self.action_space = spaces.Box(low=-MAX_STEP, high=MAX_STEP,
                                       shape=(2,), dtype=np.float32)
        state_dim = 7 if with_velocity else 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(state_dim,), dtype=np.float32)

        import mujoco
        self._mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(self.scene_path)
        self.d = mujoco.MjData(self.m)
        self._bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "push_object")
        self._ja = self.m.jnt_qposadr[self.m.body_jntadr[self._bid]]
        self._arm = _Arm(self.m, self.d)
        self._renderer = None                   # lazy: EGL context must be post-fork
        self._block_geoms = {g for g in range(self.m.ngeom)
                             if self.m.geom_bodyid[g] == self._bid}
        self._push_geoms = self._pusher_geom_ids()
        self._apply_dynamics_shift()

        self.reset_to_state = None
        self.random_state = np.random.RandomState()
        self._prev_ee = np.zeros(2)

    # ------------------------------------------------------------------ helpers
    def _apply_dynamics_shift(self):
        """Scale the block's friction and mass. friction_scale=1.0/mass_scale=1.0
        is the training distribution; anything else is an OOD dynamics condition."""
        if self.friction_scale == 1.0 and self.mass_scale == 1.0:
            return
        for g in self._block_geoms:
            self.m.geom_friction[g] *= self.friction_scale
        self.m.body_mass[self._bid] *= self.mass_scale
        self.m.body_inertia[self._bid] *= self.mass_scale
        self._mujoco.mj_setConst(self.m, self.d)

    def _pusher_geom_ids(self):
        mj = self._mujoco
        ids = set()
        for bname in ("gripper", "moving_jaw_so101_v1"):
            bid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                ids |= {g for g in range(self.m.ngeom)
                        if self.m.geom_bodyid[g] == bid}
        return ids

    def _render_frame(self, mode="rgb_array"):
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.m, RENDER_SIZE, RENDER_SIZE)
        self._renderer.update_scene(self.d, camera="overhead_cam")
        return self._renderer.render().copy()

    def _block_pose(self):
        pos = self.d.xpos[self._bid]
        q = self.d.xquat[self._bid]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                         1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return float(pos[0]), float(pos[1]), float(yaw % (2 * np.pi))

    def _get_state(self):
        ee = self._arm.site_pos()[:2]
        bx, by, byaw = self._block_pose()
        vel = (ee - self._prev_ee) / (STEPS_PER_ACTION * self.m.opt.timestep)
        if self.with_velocity:
            return np.array([ee[0], ee[1], bx, by, byaw, vel[0], vel[1]],
                            dtype=np.float32)
        return np.array([ee[0], ee[1], bx, by, byaw], dtype=np.float32)

    def _obs(self, state):
        proprio = state[:2]
        if self.with_velocity:
            proprio = np.concatenate((proprio, state[5:]))
        return {"visual": self._render_frame(), "proprio": proprio}

    # ------------------------------------------------------------------ gym API
    def seed(self, seed=None):
        self.random_state = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        mj = self._mujoco
        mj.mj_resetData(self.m, self.d)

        state = self.reset_to_state
        if state is None:
            rs = self.random_state
            state = np.array([
                rs.uniform(*WS_X), rs.uniform(*WS_Y),                # ee
                rs.uniform(WS_X[0] + 0.03, WS_X[1] - 0.03),          # block
                rs.uniform(WS_Y[0] + 0.03, WS_Y[1] - 0.03),
                rs.uniform(0, 2 * np.pi),
                0.0, 0.0,
            ], dtype=np.float32)[: (7 if self.with_velocity else 5)]

        bx, by, byaw = float(state[2]), float(state[3]), float(state[4])
        self.d.qpos[self._ja:self._ja + 3] = [bx, by, BLOCK_REST_Z]
        self.d.qpos[self._ja + 3:self._ja + 7] = [np.cos(byaw / 2), 0, 0,
                                                  np.sin(byaw / 2)]
        self.d.qpos[:ARM_DOF] = HOME_CTRL
        self.d.ctrl[:] = HOME_CTRL
        for _ in range(SETTLE_STEPS):
            mj.mj_step(self.m, self.d)

        target = np.array([float(state[0]), float(state[1]), PUSHER_Z])
        q, ok = self._arm.solve(target, self.d.qpos[:ARM_DOF].copy())
        if ok:
            self.d.ctrl[:ARM_DOF] = q
            # generous servo settle: position actuators approach the target
            # asymptotically and eval replay quality depends on hitting init states
            for _ in range(STEPS_PER_ACTION * 12):
                mj.mj_step(self.m, self.d)
        # if IK failed the arm stays home; state below reports reality

        self._prev_ee = self._arm.site_pos()[:2].copy()
        out = self._get_state()
        return self._obs(out), out

    def step(self, action):
        mj = self._mujoco
        action = np.asarray(action, dtype=np.float64) * self.action_scale
        ee = self._arm.site_pos()[:2]
        target_xy = ee + action if self.relative else action
        # the GD planner emits unbounded actions (planning/gd.py has no clamp);
        # the env saturates rather than erroring
        target_xy = np.clip(target_xy, [WS_X[0], WS_Y[0]], [WS_X[1], WS_Y[1]])

        info_extra = {}
        q, ok = self._arm.solve(np.array([target_xy[0], target_xy[1], PUSHER_Z]),
                                self.d.qpos[:ARM_DOF].copy())
        if ok:
            self.d.ctrl[:ARM_DOF] = np.clip(
                q, self.m.actuator_ctrlrange[:ARM_DOF, 0],
                self.m.actuator_ctrlrange[:ARM_DOF, 1])
        else:
            info_extra["ik_failed"] = True      # hold previous ctrl; no-op action

        self._prev_ee = self._arm.site_pos()[:2].copy()
        n_contacts = 0
        for _ in range(STEPS_PER_ACTION):
            mj.mj_step(self.m, self.d)
            for c in range(self.d.ncon):
                g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
                if ((g1 in self._push_geoms and g2 in self._block_geoms) or
                        (g2 in self._push_geoms and g1 in self._block_geoms)):
                    n_contacts += 1

        state = self._get_state()
        obs = self._obs(state)
        info = {
            "state": state,
            "pos_agent": state[:2].copy(),
            "block_pose": np.array([state[2], state[3], state[4]]),
            "n_contacts": n_contacts,
        }
        info.update(info_extra)
        return obs, 0.0, False, info

    def render(self, mode="rgb_array"):
        return self._render_frame(mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class SO101PushWrapper(SO101PushEnv):
    """PushTWrapper-contract layer: prepare / rollout / update_env /
    sample_random_init_goal_states / eval_state, as consumed by env/venv.py
    workers and plan.py."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)

        def gen():
            s = [rs.uniform(*WS_X), rs.uniform(*WS_Y),
                 rs.uniform(WS_X[0] + 0.03, WS_X[1] - 0.03),
                 rs.uniform(WS_Y[0] + 0.03, WS_Y[1] - 0.03),
                 rs.uniform(0, 2 * np.pi)]
            if self.with_velocity:
                s += [0.0, 0.0]
            return np.array(s, dtype=np.float32)

        return gen(), gen()

    def update_env(self, env_info):
        if "shape" in env_info:
            self.shape = env_info["shape"]      # only 'P' exists; stored for parity
        # color overrides are a PushT-render concept; inert here by design

    def eval_state(self, goal_state, cur_state):
        """POSITION-ONLY success: block centroid within success_radius of the goal
        block centroid. Orientation deliberately ignored (see module docstring)."""
        pos_diff = float(np.linalg.norm(
            np.asarray(goal_state[2:4]) - np.asarray(cur_state[2:4])))
        ok = pos_diff < self.success_radius
        # yaw lives at index 4 and is wrapped to [0, 2pi), so compare on the circle
        ang_diff = abs(float(goal_state[4]) - float(cur_state[4]))
        ang_diff = min(ang_diff, 2 * np.pi - ang_diff)
        if self.success_angle is not None:
            ok = ok and (ang_diff < self.success_angle)
        return {
            "success": ok,
            "state_dist": float(np.linalg.norm(
                np.asarray(goal_state) - np.asarray(cur_state))),
            "pos_diff": pos_diff,
            "ang_diff": ang_diff,
        }

    def prepare(self, seed, init_state):
        self.seed(seed)
        self.reset_to_state = np.asarray(init_state, dtype=np.float32)
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        obses, rewards, dones, infos = [], [], [], []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (_aggregate_dct(obses), np.stack(rewards), np.stack(dones),
                _aggregate_dct(infos))

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
