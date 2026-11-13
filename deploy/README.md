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

**Fingertip frame.** What matters is that the same point is used both to register the hole and at run
time, and that the **real grasp puts that point 32 mm above the peg's bottom.** A constant offset along
the tool axis then cancels out. Use FrankaPy's end-effector frame at the fingertips (Franka's default
is 0.1034 m from the flange; the sim's fingers put the point 0.1121 m from the hand frame). Check that
yaw matches as well: the sim hand frame is the flange rotated −45° about z, like Franka's `panda_hand`.

## Phases

### Phase 0: sim-side preparation (done 2026-11-12)

- [x] Exported all three seeds (`exports/peg_forge_seed{0,1,2}/`, from the `last_*_ep_200_*` checkpoints).
  All three pass `sim_bridge_test`.
- [x] Split the runner into `update_actions(mu)`, run at 15 Hz, and `target(pose)`, run at 120 Hz or
  faster. ForgeEnv recomputes the clipped target from the current pose on every physics step, so a
  bridge that holds one target for the whole 1/15 s would not match the sim. `sim_bridge_test` now
  checks the target on every substep (2384 checked on seed 2, PASS).
- [x] Measured the frames and force sign (table above).
- [ ] Stress evals on seed 2 (`results/stress/`), each changing one thing that will differ on the robot:

  | Variant | 0–1 mm | 1–2.5 mm | 2.5–5 mm |
  |---|---|---|---|
  | Baseline | 97.9% | 97.4% | 90.5% |
  | Force sign flipped (`--force_scale -1`) | pending | | |
  | Force offset 1 N, random direction | pending | | |
  | Force offset 2 N | pending | | |
  | Force offset 4 N | pending | | |
  | One policy step of action delay | pending | | |
  | Stiffness × 0.7 | pending | | |
  | Stiffness × 1.3 | pending | | |

### Phase 1: hardware

Peg, hole base and grasp jig: see [`../hardware/README.md`](../hardware/README.md). The hole needs to
be clamped or bolted to the table, because in sim it's welded down. Set the gripper + peg load in Desk,
or the force estimate will carry a constant offset.

### Phase 2: the FrankaPy bridge (not written yet)

```
1 kHz robot state ─► ForceFilter( force_sign · (O_F_ext_hat_K − tare) )
                 └─► fingertip pose = O_T_EE (EE frame at the fingertips)
                     velocity = pose difference over ≈ 1/120 s
15 Hz:   obs → LSTM → runner.update_actions(mu)        (runner.step does both)
≥120 Hz: runner.target(latest fingertip pose) → FrankaPy Cartesian impedance
         stiffness 565 N/m (translation), 28 N·m/rad (rotation)
```

- **Tare** the force for about 0.5 s in free space before each attempt. `ForceFilter.reset()` at the
  start of each attempt, as in sim.
- **Stop** when the success prediction exceeds 0.5, or after 10 s (150 steps).
- **Log** observations, actions and targets at 15 Hz and raw force at 1 kHz, so attempts can be
  replayed and compared with the sim.
- **Time the LSTM on the robot computer's CPU.** It must fit well inside the 66 ms step.

### Phase 3: calibration on the robot

1. **Force sign.** Press down on the gripper (or press the held peg onto the table) and read
   `O_F_ext_hat_K` z. The sim reads **+z** for this press, so set `force_sign` to +1 if the robot
   also reads positive, and −1 if it reads negative. A wrong sign makes the policy slide *away* from
   the hole (the flipped stress test shows how badly).
2. **Hole registration.** Grasp the peg at 32 mm, hand-guide it until it's fully seated, and record
   the fingertip pose. The hole-rim centre is that x, y, with z = the seated fingertip z − 32 mm +
   25 mm (rim height above the hole floor). Repeat about 5 times; the spread should be under 0.5 mm.
3. **Success check** (Factory's definition): the peg's bottom is within 2.5 mm of the hole axis and
   within 1 mm of the floor. On the robot: fingertip z within 1 mm of the seated z from step 2.

### Phase 4: dry runs, in order of risk

1. **Shadow mode:** the policy runs on real observations, but nothing is sent. Actions should look
   like sim actions.
2. **In the air:** the hole estimate is set 5 cm above the real hole. The policy should descend
   smoothly and stop near the estimate.
3. **Real insertions** with 0 mm of added error, someone holding the enable switch.

**Safety limits for every run:**
- Software stop if |force| > 25 N.
- Bounding box of ±3 cm around the hole, with a z floor.
- Franka collision thresholds set above the expected 10–15 N contacts, but not disabled.

### Phase 5: evaluation protocol

This mirrors `eval_sweep.py --paper_bins`:

- **Error:** for each attempt, offset the hole estimate from the registered position by a radius drawn
  uniformly from the range, in a random 3D direction.
- **Start pose:** the estimate + (0, 0, 4.7 cm), plus random noise within ±2 cm (x, y), ±1 cm (z) and
  ±45° yaw. This matches the sim's `hand_init_*` settings.
- **Size:** 20 attempts per range × 3 ranges = 60 per checkpoint, about 45–60 minutes with automatic
  retract and reset. Seed 2 first, then seeds 0 and 1.
- **Report** the same metrics as `results/*.summary.json`, next to the sim numbers.
- **Optional:** force thresholds of 5 and 10 N.

## Risks

| Risk | Mitigation |
|---|---|
| Force offset or wrong sign | Tare before each attempt; the press test; the stress evals show the tolerance |
| Peg slipping in the grasp | Grasping jig; check the seated height after each attempt |
| Real impedance controller ≠ sim controller (the sim also cancels inertia and has a dead zone) | Stiffness stress evals; the sim already randomizes stiffness by ±41% |
| Error in the hole registration | Repeat the hand-guided registration 5 times |
| Printed parts off-size | Clearance coupon (`../hardware/README.md`); measure and report the actual clearance |
