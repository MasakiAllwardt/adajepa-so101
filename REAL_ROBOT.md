# SO-101 real-robot handoff — AdaJEPA sim2real

Step-by-step instructions for taking the SO-101 PushP task from sim
(`env/so101/so101_env.py`) to the physical arm. Expands step 4 of
`HANDOFF.md`. All constants below are taken from `so101_env.py` /
`collect_data.py` — if either changes, update this doc.

The task is quasi-static, so the arm can simply **pause between replans**;
no real-time constraint on planning/adaptation latency.

## Validation gates (run in this order — do not skip ahead)

1. **Kinematic parity** — real arm reaches commanded EE targets to within a
   few mm of the MuJoCo twin's prediction across the workspace.
2. **Embedding-space observation parity** — real overhead images of matched
   scenes embed close to sim images (DINOv2 features), i.e. the visual gap
   is within what the corruption experiments already cover.
3. **Frozen baseline on real** — run the frozen world model; success rate is
   the sim2real reference point.
4. **Frozen vs AdaJEPA on real** — the actual experiment.

## 1. Physical setup

- Print the block: `~/Documents/SO-ARM100/Simulation/SO101/assets/P.stl`.
- Mark the workspace on the table, in the robot base frame:
  **x ∈ [−0.15, 0.15] m, y ∈ [0.23, 0.42] m** (`WS_X` / `WS_Y`).
- Overhead camera at **[0, 0.26, 0.44] m**, looking down, **~65° vertical
  fovy**, matched to the sim camera to **±2°** (verify by overlaying a real
  frame on a sim render of the marked workspace corners).
- **Fixed exposure / white balance** — auto-exposure is an uncontrolled
  visual shift.
- Frames must be delivered as **224×224 RGB uint8** (`RENDER_SIZE`); crop
  centrally to square before resizing.
- **No fiducials/markers on the block** — an AprilTag is an appearance
  confound for the vision encoder. Scoring is offline CV (see §5).

## 2. `SO101RealEnv` implementation spec

New file (suggested: `env/so101/so101_real_env.py`), implementing the same
PushTWrapper contract as `SO101PushWrapper` so `plan.py` and `env/venv.py`
need no changes: `prepare`, `rollout`, `step_multiple`, `update_env`,
`sample_random_init_goal_states`, `eval_state`.

**Action interface** (must match sim exactly — same MDP as training data):

- action = commanded planar EE delta **(dx, dy) in metres**, `action_scale
  = 1.0` (metric — never the ÷100 pixel convention of pusht/pushobj).
- Clamp per-step magnitude to **±0.015 m** (`MAX_STEP`) and clamp the target
  to the workspace box, exactly as `SO101PushEnv.step` does.
- Control period **0.05 s (20 Hz)** in sim (`STEPS_PER_ACTION × 2 ms`); on
  real, execute the delta and wait for the move to settle before grabbing
  the next observation — quasi-static, so timing need not be exact.
- Pusher height fixed at **z = 0.018 m** (`PUSHER_Z`); wrist_roll pinned to
  0, gripper held closed.

**Kinematic twin for FK/IK**: instantiate a headless MuJoCo model of the
scene (no stepping, no rendering) and reuse `_Arm.solve` (damped-least-
squares IK on the `gripperframe` site, 4 IK joints, wrist/gripper pinned)
to convert EE targets → joint targets, and `_Arm.site_pos` for FK on read-
back joint positions. This keeps real and sim using byte-identical
kinematics.

**LeRobot bridge** (new-calibration convention):

- Joint targets from the twin are in radians; convert per LeRobot's
  new-calib normalization for the SO-101.
- **Gripper**: LeRobot exposes 0–100; sim holds a fixed closed pose of
  **−0.1745 rad** (`GRIPPER_CLOSED`). Calibrate once: find the LeRobot
  gripper value whose physical jaw pose matches the sim closed pose, then
  hold it constant forever.

**Observation / state layout** (mirror `_get_obs` ordering):

- `state = [ee_x, ee_y, obj_x, obj_y, obj_yaw, ee_vx, ee_vy]` (float32,
  metres/radians, yaw wrapped to [0, 2π)).
- `proprio = [ee_x, ee_y, ee_vx, ee_vy]` — EE position from twin FK,
  velocity as finite difference over the control period, like
  `_get_state`.
- Block pose entries are only needed for *scoring*, not for the planner
  (it consumes visual + proprio). On real, fill them from the offline CV
  pipeline (§5) or NaN during rollout and score afterwards.
- Normalize with the **same `stats.json`** the sim dataset uses
  (`$DATASET_DIR/so101_pushp_16k/stats.json`, loaded by
  `datasets.so101_pushp_dset.load_stats`). Do not recompute stats on real
  data.

## 3. Gate 1 — kinematic parity

Command a grid of EE targets across the workspace (e.g. the same 7×7 grid
`_Arm` was tuned on; sim achieved 0.046 mm median IK error, 49/49
convergence). For each target: twin-IK → send joints → read back joints →
twin-FK → compare with a measured/ground-truth EE position. Pass when
errors are a few mm and unbiased across the workspace.

## 4. Gate 2 + goal pairs — real data collection

- Record real goal pairs as **(obs_t, obs_{t+25})** — 25 control steps
  apart, matching the sim goal-segment horizon — keeping only pairs where
  the block moved **≥ 15 mm** (same motion filter as sim
  `plan_targets.pkl`).
- Gate 2 check: embed matched real/sim frames with the model's DINOv2
  encoder; real frames should sit within the embedding spread that blur /
  snp corruptions produce on sim frames. If they are much further out,
  fix the camera match (geometry, exposure, background) before burning
  robot time.

## 5. Scoring (offline CV)

Success = block centroid within **25 mm** of goal centroid
(`success_radius`), position-only — same `eval_state` criterion as the
headline sim numbers. (Orientation-aware variant: `success_angle` in
radians adds a yaw tolerance; yaw compared on the circle.)

Centroid from the overhead frames by classical CV (color/contour
segmentation of the P block), computed **offline after the rollout** so
nothing about scoring touches the observations the planner sees.

## 6. Running the evals

Prerequisite — reconstruct the SO101 plan config (lost with the pod, see
HANDOFF.md "Known gap"): copy `conf/adajepa_plan_gd_pushobj.yaml` →
`conf/adajepa_plan_gd_so101.yaml`, then set:

- `model_name`:
  `so101_pushp_cos1e-2_agg32_projnone_dim384_hw14_sgTrue_lr1e-05_f5_h3_p1`
  (checkpoint:
  `~/Documents/SO-ARM100/Simulation/SO101/pod_salvage/so101_model/model_latest.pth`)
- `eval_data_path`: `$DATASET_DIR/so101_pushp_16k/plan_targets.pkl`
  (real runs: a real-pairs pickle in the same format, §4)
- `goal_source: segments`
- adapt knobs: `planner.adapt.lr: 5e-4`, `planner.adapt.steps: 1`
  (the sweep said nothing better; encoder adaptation did not help)
- export `SO101_SCENE` (twin needs the scene XML even on real) and
  `DATASET_DIR`.

Sanity-check the reconstructed config **in sim first**: rerun blur+adapt
n=50 and expect ≈0.66 (HANDOFF.md step 2). Then:

1. **Gate 3**: frozen model on real, n as budget allows (sim frozen clean
   reference: 0.82–0.84). Expect a drop; this is the baseline.
2. **Gate 4**: frozen vs AdaJEPA on the same goal set, interleaved A/B so
   drift (lighting, battery, table wear) hits both arms equally. Remember
   adapt runs cost ~10× frozen wall-clock in sim; on real the robot is the
   bottleneck, so the planner-side cost mostly disappears.

## Caution from sim results

Adaptation helped under *visual* shift (blur 0.62→0.66, snp5 0.44→0.48)
but **hurt** under shape shift (0.86→0.76) and was unneeded under dynamics
shift. The real robot is a bundle of all three shift types at once —
if AdaJEPA underperforms frozen on real, that is a consistent (and
publishable) outcome, not necessarily a bug.
