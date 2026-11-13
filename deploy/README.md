# Deploying the peg policy on the real Franka

**Goal:** run the trained `Isaac-ForgeBlocks-Peg8mm` policies on the real robot with the same protocol
as the sim evaluation, so the results compare directly. That means 3D position error in three ranges
(0–1, 1–2.5, 2.5–5 mm), a 7.5 N force threshold, and the same metrics: success, time to insert, peak
force, and success-prediction precision/recall.

Sim baseline (`results/peg_forge_seed*.summary.json`, 1000 episodes per range):

| Error range | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 0–1 mm | 98.3% | 98.7% | 97.9% | 98.3% |
| 1–2.5 mm | 95.9% | 96.4% | 97.4% | 96.6% |
| 2.5–5 mm | 83.8% | 82.1% | 90.5% | 85.5% |

Start with **seed 2**: it has the best success at 2.5–5 mm. Keep seed 0 for comparison: its success
prediction is the most accurate.

## What the policy does

The policy is an LSTM (2 layers of 1024 units) that runs at **15 Hz**. The robot is already holding
the peg above the hole. At each step the policy outputs where the fingertips should be, and a Cartesian
impedance controller moves the arm there. The policy is **never told where the hole really is.** It gets
an estimate that is wrong by up to 5 mm, and finds the hole by touch: it presses down, slides in the
direction the contact force suggests, and drops in once the peg lines up.

**Inputs** (24 numbers, built by `ForgePolicyRunner.build_obs`):

| Input | Size | Notes |
|---|---|---|
| Fingertip position minus estimated hole-rim centre | 3 | The only position it sees, so where the robot base sits doesn't matter |
| Fingertip quaternion with w and z set to 0 | 4 | With the gripper pointing down, the remaining two mostly encode yaw |
| Linear velocity | 3 | Finite difference of fingertip position over about 1/120 s |
| Angular velocity, z only | 3 | x and y are set to 0 |
| Smoothed contact force, base-frame axes | 3 | Robot: `O_F_ext_hat_K` × `force_sign`, minus the tare offset, smoothed with `ForceFilter` |
| Force threshold | 1 | 7.5 N. The policy was trained with 5–10 N; this is its "how hard to push" setting |
| Previous smoothed actions | 7 | Roll and pitch set to 0 |

**Outputs** (7 numbers in [-1, 1]):

- **x, y, z:** where the fingertip should be, relative to the estimated hole, scaled to ±5 cm.
- **Roll and pitch:** ignored. The gripper always points straight down.
- **Yaw:** spans a 270° range. Hardly matters for a round peg.
- **Success prediction:** `(a+1)/2`. Above 0.5, the attempt stops.

**How outputs become a command:** the actions are smoothed (`a = 0.0625·new + 0.9375·old`, which takes
about 1 s to respond). The target is the estimated hole plus 5 cm × the smoothed action, then **clipped
to within 2 cm of the current fingertip pose**. With stiffness 565 N/m, that caps the push at about 11 N.

**Why impedance control:** the policy was trained with it, and it outputs targets for a spring, not
motions. To push harder it places the target deeper into the surface. The compliance also lets the peg
slide into the hole without hundreds of N from a stiff position loop.

## Numbers measured in sim

These come from `scripts/measure_frames.py`; the full output is in `results/measure_frames.json` on mrsd26c.

| Quantity | Value | Use on the robot |
|---|---|---|
| Fingertip midpoint, in the `panda_hand` frame | (0, 0, **0.1121 m**), same orientation | The point the policy calls the fingertip. See "Fingertip frame" below |
| Grasp height: fingertip midpoint above the peg's bottom | **32.1 mm** (sim range 29.9–34.1) | Grasp the real peg at this height; use the jig in `../hardware/` |
| Force while pressing the peg down onto the rim | **+z, about 9 N** | The sim reports the force the environment exerts *on the hand*. Set `force_sign` so the robot matches |
| Force in free space | under 0.1 N | Sim gravity is off for the robot, so the real reading must be tared to about 0 |
| Peg | Ø7.98 × 50 mm, 0.5 mm chamfers, 19 g | `../hardware/README.md` |
| Hole | **Ø9.0** × 25 mm deep, 0.5 mm top chamfer, so about **1 mm clearance** | The task config says 8.1 mm, but the collision mesh is 9.0. The mesh is what the policy trained against |

**Fingertip frame (FrankaPy → sim).** The lab's Lab2 code (`svvetal/Robot_Autonomy_Lab`, tested on
this robot) showed that FrankaPy's `get_pose()` is the hand frame shifted 0.103 m along its z axis,
with the same orientation as the hand frame. The sim's fingertip point is the same hand frame shifted
0.1121 m. So the bridge only needs a fixed shift, with no rotation:

```
sim_fingertip_pos = frankapy_pos + R_frankapy @ [0, 0, 0.0087]      sim_fingertip_quat = frankapy_quat
```

- **Orientation:** gripper pointing down at yaw 0 is `R = diag(1, −1, −1)` in FrankaPy (Lab2's
  `TOP_R`), which is the sim's start orientation (roll π, yaw 0).
- **Physical grasp:** the sim point sits at about the **ends of the fingers**, and the pad centres are
  about 9 mm higher. So the grasp leaves **32 mm of peg below the finger ends**, or 40.8 mm below
  FrankaPy's tool point.

**Where to put the hole.** In sim the hole is at (0.6, 0) ±5 cm from the robot base. Lab2's cube pickup
spot, (0.59, 0) on the table, already has proven top-down reach. Put the hole base there. The table top
is at z ≈ 0 in the base frame, so the boss top is at about z = 0.028 m, or more if the base is raised
on a board.

## Phases

### Phase 0: sim-side preparation (done 2026-11-12)

- [x] Exported all three seeds (`exports/peg_forge_seed{0,1,2}/`, from the `last_*_ep_200_*` checkpoints).
  All three pass `sim_bridge_test`.
- [x] Split the runner into `update_actions(mu)`, run at 15 Hz, and `target(pose)`, run at 120 Hz or
  faster. ForgeEnv recomputes the clipped target from the current pose on every physics step, so a
  bridge that holds one target for the whole 1/15 s would not match the sim. `sim_bridge_test` now
  checks the target on every substep (2384 checked on seed 2, PASS).
- [x] Measured the frames and force sign (table above).
- [x] Quaternion sign. The policy sees the quaternion's x and y only, and they flip with q → −q.
  In sim, x is always positive (0.93 to 1.0 over the ±45° start yaws), so the bridge picks the sign
  closest to (0, 1, 0, 0) (`BridgeConfig.quat_ref`).
- [ ] Stress evals on seed 2 (`results/stress/`), each changing one thing that will differ on the robot:

  | Variant | 0–1 mm | 1–2.5 mm | 2.5–5 mm |
  |---|---|---|---|
  | Baseline | 97.9% | 97.4% | 90.5% |
  | Force sign flipped (`--force_scale -1`) | 34.0% | 30.7% | 25.0% |
  | Force offset 1 N, random direction | 96.5% | 95.4% | 85.0% |
  | Force offset 2 N | 87.0% | 83.5% | 70.6% |
  | Force offset 4 N | 34.7% | pending | pending |
  | One policy step of action delay | pending | | |
  | Stiffness × 0.7 | pending | | |
  | Stiffness × 1.3 | pending | | |

  **Force offset is the biggest risk.** 1 N costs a few points, 2 N costs 10–20, and 4 N is as bad as
  a flipped sign. The tare before each attempt has to leave well under 1 N, and the robot's force
  estimate must not drift by more than that during the 2–10 s attempt. If it does, retrain with a
  random per-episode force offset (an `obs_rand` option like the stress test, but during training).

### Phase 1: hardware

Peg, hole base and grasp jig: see [`../hardware/README.md`](../hardware/README.md); the printable parts
are in `../hardware/peg_hole_MK4/`. The hole needs to be clamped or bolted to the table, because in
sim it's welded down. Set the gripper + peg load in Desk, or the force estimate will carry a constant
offset.

### Phase 2: the FrankaPy bridge

Code: [`franka_bridge.py`](franka_bridge.py) (robot side) and [`run_peg.py`](run_peg.py) (command
line). Tests with a fake arm: `tests/test_franka_bridge.py`. Robot access, the gripper and the safety
pattern follow Lab2's `hardware/robot.py` and `hardware/execute.py`, which already work on this robot.
These parts are new: streamed impedance targets, force tare and filtering, velocities from J·q̇, and
the policy loop.

```
each control tick (default 200 Hz):
    state ← FrankaPy (pose, joints, joint velocities, O_F_ext_hat_K, Jacobian)
    fingertip = tool pose + 8.7 mm along tool z;  quat sign picked like the sim
    force = ForceFilter(raw − tare)         (runner applies force_sign)
    velocity = J·q̇ moved to the fingertip  (pose difference if FrankaPy has no Jacobian)
    safety check: force, box, speed
    15 Hz:     obs → LSTM → actions (runner.step);  stop if success prediction > 0.5 or 10 s
    otherwise: runner.target(latest fingertip pose)
    stream the target to FrankaPy's dynamic Cartesian impedance skill (565 N/m, 28 N·m/rad)
```

**Check these on the lab computer before the first run:**

```bash
python3 --version
python3 -c "import torch; print(torch.__version__)"   # the policy was exported with torch 2.7.0
pip show frankapy                                      # version / location
```

TorchScript files from torch 2.7 may not load in older torch; FrankaPy setups often run Python 3.8,
where the newest torch is 2.4. If loading fails, the policy needs a re-export (ONNX, or TorchScript
from a matching torch). The FrankaPy calls are all in `FrankaPyArm`, written from FrankaPy's examples
and Lab2; check them against the installed version.

### Phase 3: calibration on the robot

Commands are `python3 deploy/run_peg.py <command>`. Moving commands need `--ack-safety` and an
E-stop operator. State (`calibration.json`, `hole.json`, logs) goes to `deploy/runs/`.

1. `check`: read-only. Prints joints, tool and fingertip poses, raw force, gripper width, and whether
   FrankaPy provides a Jacobian.
2. `rate-test --export exports/peg_forge_seed2`: holds the current pose while streaming it back.
   Reports the achieved stream rate, how often the robot state really updates, and the policy's
   inference time. If the stream rate is well below 120 Hz, lower `--control_hz` and tell me, so
   the effect can be measured in sim.
3. `grasp`: closes on the peg (40 N) with the finger ends on the 32 mm line.
4. `press-test`: lowers the held peg onto a **free flat surface** (not the hole) until the force
   builds up, and saves `force_sign` so that pressing reads +z, as in sim. Use the robot for this,
   not your hand: a hand pushing down on the gripper is a force in the opposite direction.
5. `register-hole`: guide mode, 5 times: seat the peg fully by hand and hold it still. Saves the
   seated fingertip pose; the rim centre is 7.1 mm below it (32.1 mm grasp − 25 mm depth). The spread
   should be under 0.5 mm.

**Success** (Factory's definition): the peg's bottom within 2.5 mm of the hole axis and within 1 mm of
fully seated. The bridge checks the fingertip against the registered seated pose.

### Phase 4: dry runs, in order of risk

1. `shadow --export ...`: guide mode; move the peg around above the hole by hand. The policy runs on
   real observations and prints its targets, but nothing is commanded. Targets should point toward the
   hole and down.
2. `run --export ... --air --trials 1`: the estimate is 10 cm above the real hole, so the policy
   descends toward it and never reaches the boss.
3. `run --export ... --band 0 0 --trials 3`: real insertions with no added error.

**Safety limits** (`SafetyLimits`, checked every tick; a stop ends the attempt and the series):
- Smoothed, tared force above 25 N.
- Fingertip more than 4 cm (x or y) from both the hole and the estimate.
- Fingertip below the seated height minus 3 mm (peg slipped, or wrong hole), or more than 9 cm above
  the estimate.
- Speed above 0.3 m/s.
- Franka's own collision thresholds stay on (franka-interface configuration).

### Phase 5: evaluation protocol

`run --band MIN MAX --trials 20` mirrors `eval_sweep.py --paper_bins`. Run it for each of the three
bands, then `summary`:

- **Error:** the hole estimate is the registered rim plus a radius uniform in the band, in a random 3D
  direction.
- **Start pose:** the **registered** rim + (0, 0, 4.7 cm), plus uniform noise of ±2 cm (x, y), ±1 cm
  (z) and ±45° yaw. As in Factory, the start is relative to the true hole, not the estimate.
- **Size:** 20 attempts per band × 3 bands = 60 per checkpoint, about 45–60 minutes. Seed 2 first,
  then seeds 0 and 1.
- **Report:** `summary` prints success rate, median time, peak force and prediction precision/recall
  per band, in the same form as `results/*.summary.json`. Per-step logs are in `runs/attempts/`.
- **Optional:** `--force_threshold 5` and `10`.

## Risks

| Risk | Mitigation |
|---|---|
| Force offset or wrong sign | Robot press test; tare before each attempt; offsets above 1 N cost a lot (stress table); retrain with random offsets if the real estimate drifts |
| Peg slipping in the grasp | 40 N grasp; the safety floor catches a peg pushed up in the fingers |
| Real impedance controller ≠ sim controller (the sim also cancels inertia and has a dead zone) | Stiffness stress evals; the sim already randomizes stiffness by ±41% |
| Stream or state rate too low | `rate-test`; then measure the effect in sim |
| Policy file won't load on the lab computer | Check the torch version first; re-export |
| Error in the hole registration | Repeat the hand-guided registration 5 times |
| Printed parts off-size | Clearance coupon (`../hardware/README.md`); measure and report the actual clearance |
