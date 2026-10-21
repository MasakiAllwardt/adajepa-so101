# HANDOFF — AdaJEPA × SO-101 sim2real (written 2026-10-21)

Context dump for continuing work in this repo. Everything below was verified
against files on disk, not recalled from memory.

## What this repo is

Fork of **temporal-straightening** (DINO-WM descendant) extended with:

- `planning/adajepa.py` — `AdaJEPATrainer`: test-time adaptation of the
  predictor (optionally encoder head) on observed trajectory segments, same
  sliding-window 1-step latent loss as training. Defaults: 1 Adam step,
  predictor last layer only.
- `planning/adajepa_mpc.py` — MPC planner that runs adapt steps between
  replan iterations.
- `planning/image_corruption.py` — eval-time visual shifts; `ood_corruption`
  takes `null | blur | snp1 | snp5 | dark` (+ optional `ood_level`), wired
  into `plan.py` (see `corrupt_obs_dict` call sites).
- `env/so101/so101_env.py` — SO-101 PushP MuJoCo env (PushTWrapper contract,
  metric EE-delta actions). Has `friction_scale` / `mass_scale` dynamics-shift
  knobs and `success_angle` (None = position-only success; float = also
  require yaw within tolerance, used by the `ori_*` runs).
- `datasets/so101_pushp_dset.py`, `conf/env/so101_pushp.yaml`.

**This repo is canonical.** The copy at
`~/Documents/SO-ARM100/Simulation/SO101/temporal-straightening/` is a stale
pre-AdaJEPA duplicate — do not edit it.

## Split across two directories (deliberate)

- **Here**: model/planning/adaptation code.
- **`~/Documents/SO-ARM100/Simulation/SO101/`**: scene assets (`scene.xml`,
  `scene_L.xml` L-block shape shift, URDFs, meshes incl. `assets/P.stl`),
  `collect_data.py` (data collector; physics/IK constants intentionally
  duplicated with `so101_env.py` — change both or neither), datasets
  (`data/so101_pushp` tiny local, `data/so101_pushp_16k` the real one:
  14,413 train / 1,607 val), and `pod_salvage/` (see below).

Runtime wiring: `SO101_SCENE` → path to `scene.xml` (or `scene_L.xml`);
`DATASET_DIR` → parent of `so101_pushp*`.

## State as of 2026-10-21

- Code here == final RunPod state. The last pod-side edit (`success_angle` in
  `so101_env.py`, Sep 1) was merged in from
  `pod_salvage/round2/round2.tgz` on 2026-10-21; verified byte-identical.
- The pod itself is gone. Salvaged artifacts live in
  `~/Documents/SO-ARM100/Simulation/SO101/pod_salvage/`:
  - `so101_model/model_latest.pth` — trained SO101 world model. Full name:
    `so101_pushp_cos1e-2_agg32_projnone_dim384_hw14_sgTrue_lr1e-05_f5_h3_p1`
    (DINOv2-S/14, straightening cos 1e-2, stop-grad, frameskip 5, hist 3).
  - `so101_model/logs/` + `eval_logs.tgz` — round-1 train/eval logs.
  - `round2/round2.tgz` — round-2 logs (lr/steps sweep, encoder-lr sweep,
    shape shift, orientation-criterion runs).
  - Four zips (`pusht_visual_shift`, `pushobj_shape_shift`,
    `diversemaze_layout_shift`, `mediummaze_dynamics_shift`) — AdaJEPA
    benchmark datasets; `small_artifacts.tgz` — benchmark eval outputs with
    resolved `.hydra` configs (useful reference for exact run settings).

## Known gap

The **pod-side launcher scripts and the SO101 plan config were not salvaged**.
`conf/` has `adajepa_plan_{gd,cem}_{pushobj,maze,diversemaze}.yaml` but no
SO101 variant. To rerun SO101 evals: copy `adajepa_plan_gd_pushobj.yaml`,
set `model_name` to the SO101 model, `eval_data_path` to
`$DATASET_DIR/so101_pushp_16k/plan_targets.pkl`, `goal_source: segments`,
export `SO101_SCENE`, and use `ood_corruption` / `env_kwargs_override`
(e.g. `{friction_scale: 5.0}`) for shift conditions. Key knobs:
`planner.adapt.lr` (pod used 5e-4), `planner.adapt.steps` (1).

## Results so far (sim, n=50 goals, 25 mm position-only success)

| Condition | Frozen | Adapt |
|---|---|---|
| Clean | 0.82–0.84 | — |
| Blur | 0.62 | 0.66 |
| Salt & pepper 5% | 0.44 | 0.48 |
| Dark | 0.60 | not run |
| Friction ×5 | 0.84 | run killed (rc=143) |
| Shape shift (L block) | 0.86 | 0.76 ← adaptation *hurt* |
| Orientation-aware: clean / fric×5 | 0.72 / 0.76 | truncated mid-run |

Round-2 sweeps: n=20 lr/steps sweep peaked 0.65 @ lr 5e-3, but n=50 confirm
gave 0.48 → sweep differences are noise; default 5e-4/1-step stands.
Encoder-lr 1e-4/1e-3 gave 0.55/0.50 → no win from adapting the encoder.
Adapt evals cost ~10× frozen wall-clock (5,013 s vs 466 s per 50 goals).

## Next steps (agreed order)

1. ~~Merge round-2 `so101_env.py`~~ — done 2026-10-21.
2. **Re-validate in sim**: rerun blur+adapt (n=50) with this checkout; expect
   ≈0.66. Confirms local code is behaviorally identical to what produced the
   numbers (rules out unsalvaged pod edits). Needs the SO101 plan config
   reconstructed first (see Known gap).
3. **Back up this repo** — currently the only copy in existence. git is
   broken on this Mac (Xcode CLT abort: `Symbol not found: _XPCTypeBool`);
   fix via `xcode-select --install` or brew git, then push.
4. **Real-robot handoff** (quasi-static task ⇒ arm can pause during replans):
   physical setup (print `P.stl`, mark workspace x∈[−0.15,0.15],
   y∈[0.23,0.42], overhead cam at [0, 0.26, 0.44] ~65° fovy, fixed
   exposure, geometry matched to ±2° fovy); `SO101RealEnv` implementing the
   PushTWrapper contract with a headless MuJoCo kinematic twin for FK/IK
   (reuse `_Arm.solve`), same `stats.json` normalization, LeRobot new-calib
   convention (gripper: LeRobot 0–100 vs sim fixed −0.1745 rad); real goal
   pairs recorded as (obs_t, obs_{t+25}) with the ≥15 mm motion filter;
   offline CV centroid scoring (no fiducials on the block — appearance
   confound). Gates: kinematic parity → embedding-space observation parity →
   frozen baseline on real → frozen vs AdaJEPA on real.
