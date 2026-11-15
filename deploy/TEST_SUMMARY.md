# Test summary: FORGE peg insertion, sim to real

Tests run so far (as of 2026-11-14) to get the FORGE peg-insertion policy (`Isaac-ForgeBlocks-Peg8mm-Direct-v0`,
8 mm peg) ready for the real Franka. Tests 1-5 ran in simulation or against a fake arm. Test 6 is the first
robot session (measurements only, no insertions).

| # | Test | Where | Result |
|---|---|---|---|
| 1 | Unit tests | laptop + mrsd26c container | pass |
| 2 | Sim bridge equivalence | mrsd26c, Isaac Lab | pass, all 3 seeds |
| 3 | Frame and sign measurements | mrsd26c, Isaac Lab | done, constants used in the bridge |
| 4 | Baseline evaluation, 3 seeds | mrsd26c, Isaac Lab | 82–99% success |
| 5 | Sim-to-real stress evaluations | mrsd26c, Isaac Lab | force offset and dead band are the risks |
| 6 | Robot session 1 (measurements) | 16662 lab Franka, 2026-11-14 | sensing fine; **joint friction** is the new risk |

---

## 1. Unit tests

`tests/test_observation.py`, `tests/test_franka_bridge.py` (plus the other `tests/` files).

- **Laptop:** 39 run, 14 skipped (they need torch), all pass.
- **mrsd26c container:** 33 run, all pass.

What they cover:

- Observation helpers, including the force dead band: per axis `[4, -7, 12] N → [0, -2, 7] N` with a 5 N
  band; magnitude mode keeps the direction.
- Bridge frames: tool ↔ fingertip offset (8.7 mm), quaternion sign canonicalisation, the hole and seated
  geometry.
- Full attempts on a **fake arm** (virtual clock, contact model, sensor bias, force spikes, sensor dead
  band, command lag):
  - safety stops for force, box, floor and speed;
  - force tare;
  - `press-test` sign detection;
  - `force-curve` recovers the fake arm's dead band;
  - `force-scale` separates a 1.5 N actuation dead zone from a 3 N sensing dead band;
  - `force-weights` reads hung masses, fits the dead band, and flags a `force_sign` that disagrees;
  - `force-drift` recovers a 0.05 N/s drift and a 0.2 N/cm pose-dependent error, and stops on contact;
  - `rate-test` reports the fake arm's 50 ms lag;
  - `friction-breakaway` recovers the fake arm's stiction and the force it leaks into the estimate;
  - `friction-sweep` separates Coulomb and viscous friction, the fake force opposing motion, and a
    pose-dependent error;
  - `friction-steps` stops short of the target by the kinetic friction / stiffness, less with stiffness × 1.3;
  - start-pose and error samplers;
  - saving and loading the hole registration.

## 2. Sim bridge equivalence (`scripts/sim_bridge_test.py`)

Runs the exported policy through `deploy/policy_runner.py` (the code the robot uses) next to the Isaac Lab
env, and compares the two at every step. The impedance target is checked on **every physics substep**
(120 Hz), not just at the 15 Hz policy steps.

| Export | Worst obs diff | Worst action diff | Worst target pos diff | Worst target quat diff | Result |
|---|---|---|---|---|---|
| seed 0 | 6.6e-7 | 6.8e-7 | 3.4e-8 m | 3.4e-8 | PASS |
| seed 1 | 7.7e-7 | 7.8e-7 | 3.4e-8 m | 3.0e-8 | PASS |
| seed 2 (after the runner split) | 8.9e-7 | 8.9e-7 | 3.4e-8 m | 3.4e-8 | PASS, 2384 substeps checked |

The differences are float32 rounding. The runner reproduces the sim's observations, actions and controller
targets exactly.

## 3. Frame and sign measurements (`scripts/measure_frames.py`)

Measured in sim (`results/measure_frames.json`, `results/measure_quat.json`), then used as constants in
`deploy/franka_bridge.py`:

| Quantity | Measured | Used for |
|---|---|---|
| `panda_fingertip_centered` in `panda_hand` | 0.1121 m along z, same orientation | `TIP_OFFSET` = 0.1121 − 0.1034 (FrankaPy tool point) = **8.7 mm** |
| Fingers in hand frame | ±4.0 mm in y, 58.4 mm in z | Gripper geometry check |
| Fingertip above the peg's bottom after grasp | **32.1 mm** (range 29.9–34.1) | `GRASP_TIP_ABOVE_PEG_BOTTOM`; grasp jig height |
| Force, free space | (−0.06, 0.00, 0.08) N | Noise floor |
| Force, pressing down | (0.64, 0.37, **+9.1**) N | **Pressing reads +z** (force of the environment on the hand); the robot's `press-test` must give the same sign |
| Fingertip quaternion sign | x > 0 (0.93–1.0) | Reference (0, 1, 0, 0) for picking the sign on the robot |

## 4. Baseline evaluation (`scripts/eval_sweep.py --paper_bins`)

1000 episodes per band. "Error" is the 3D distance between the true hole and the hole estimate given to
the policy (bands from the FORGE paper). Contact threshold 7.5 N; success prediction cut-off 0.5.

| Seed | 0–1 mm | 1–2.5 mm | 2.5–5 mm | Peak force | Time to success | Prediction precision |
|---|---|---|---|---|---|---|
| 0 | 98.3% | 95.9% | 83.8% | 9.9–10.2 N | 2.4–3.2 s | 0.81–0.91 |
| 1 | 98.7% | 96.4% | 82.1% | 9.5 N | 2.3–3.3 s | 0.68–0.79 |
| **2** | 97.9% | 97.4% | **90.5%** | 10.3–10.7 N | 2.4–3.2 s | 0.78–0.83 |

**Seed 2 is the deployment checkpoint:** it is best at large errors and equal elsewhere.

## 5. Sim-to-real stress evaluations (seed 2)

Each run changes one thing that may differ on the real robot. 1000 episodes per band. Results are in
`results/stress/`.

| Variant | Flag | 0–1 mm | 1–2.5 mm | 2.5–5 mm | Peak force |
|---|---|---|---|---|---|
| **Baseline** | | **97.9%** | **97.4%** | **90.5%** | 10.5 N |
| Force sign flipped | `--force_scale -1` | 34.0% | 30.7% | 25.0% | 13.8 N |
| Force offset 1 N, random direction | `--force_bias_N 1` | 96.5% | 95.4% | 85.0% | 10.5 N |
| Force offset 2 N | `--force_bias_N 2` | 87.0% | 83.5% | 70.6% | 10.6 N |
| Force offset 4 N | `--force_bias_N 4` | 34.7% | 35.9% | 27.6% | 10.0 N |
| One policy step (67 ms) of action delay | `--action_delay 1` | 86.1% | 83.8% | 75.0% | 11.1 N |
| Stiffness × 0.7 | `--gain_scale 0.7` | 96.9% | 94.6% | 85.9% | 9.4 N |
| Stiffness × 1.3 | `--gain_scale 1.3` | 98.6% | 97.1% | 91.0% | 11.3 N |
| Force dead band 2.5 N, per axis | `--force_deadband_N 2.5` | 91.5% | 89.3% | 79.4% | 11.4 N |
| Force dead band 5 N, per axis | `--force_deadband_N 5` | 78.2% | 74.0% | 61.5% | 12.4 N |
| Force dead band 5 N, on the magnitude | `... --force_deadband_mode norm` | 86.2% | 80.5% | 68.8% | 12.4 N |
| Contact threshold 10 N (vs 7.5) | `--force_threshold 10` | 97.8% | 96.9% | 90.9% | 10.5 N |
| Dead band 5 N per axis + threshold 10 N | both | 76.5% | 70.7% | 59.7% | 12.3 N |

How the force perturbations are modelled:

- **Offset:** a constant force of that size in a random direction per episode, added to the observed force.
- **Dead band:** the observed force reads 0 below the band and is reduced by the band above it. This
  models the TA's note that Franka's force estimate is unreliable below ~5 N.

### Takeaways

1. **The force sign must be right.** A flipped sign drops success to 25–34%. The robot's `press-test`
   has to confirm that pressing reads +z.
2. **Force offset is the biggest risk.** 1 N costs a few points, 2 N costs 7–20, and 4 N is as bad as a flipped
   sign. The tare before each attempt has to leave well under 1 N, and the estimate must not drift
   more than that during an attempt.
3. **A weak-force dead band hurts.** 5 N costs 12–29 points and 2.5 N costs 6–11. Raising the contact threshold
   to 10 N does not recover it. With less force to read, the policy pushes harder (peak 12.4 N vs
   10.5 N), but this stays well under the bridge's 25 N safety limit.
4. **Latency matters.** One 67 ms step of delay costs 12–16 points. The bridge must apply each action
   within the same 15 Hz tick.
5. **Stiffness mismatch is harmless.** ±30% stiffness changes success by under 5 points.

## 6. Robot session 1 (2026-11-14)

Run with `deploy/run_peg.py` on the `lukas` branch. The bridge changes this session led to are described
in [`README.md`](README.md) (Phase 2, "Lessons from the first robot session"). The lab copy had uncommitted
edits that were discarded; the earlier `force_curve` runs and the press-test were made with that code.

| Check | Result |
|---|---|
| Versions | Python 3.8.10, torch 2.4.1 (loads the torch 2.7 export), FrankaPy from a git checkout in the container |
| `check` | Load setting OK (<3 N at rest, low pose); Jacobian available; topic state matches `get_pose()` exactly |
| `rate-test` | 200 Hz stream, 100 Hz state (after the bridge fixes; 53 / 29 Hz before); policy 8 ms median, 53 ms max; 2 mm step: no motion (friction) |
| `press-test` | force_sign −1 (earlier the same day, with code that was later discarded) |
| `force-weights` | slope 0.73, dead band 0.13 N, sign agrees, drift 0.12 N over the test |
| `force-scale` | estimate ≈ 1.09 × scale + 0.6 N, no dead band; scale responds from ~2.2 N commanded |
| `force-drift` | hold within 0.15 N after taring once the stream runs (−1.74 N before the fix); controller start shifts 0.3-0.6 N; sweep: 1.5-2.4 N opposing the motion |
| `force-curve` (earlier runs) | ~2 N to start moving, 4-5 N to move at 4 mm/s; its contact detection fires in mid-air |

### Takeaways

1. **The sensing risks from test 5 did not appear.** No dead band, a near-correct gain when pressing, and
   negligible drift at rest once the tare happens after the stream starts.
2. **Joint friction is the main gap, and the stress evals don't model it.** Small target moves (1-5 N of
   spring pull) barely move the arm, and friction leaks into the force estimate as a force opposing
   motion. Measure it sideways, then decide: controller change, or retrain with friction.

## Still to do on the real robot

1. A sideways (x/y) sweep in `force_drift`, to see whether friction also fakes lateral force.
2. Decide on friction (see takeaway 2) before any insertion.
3. `register-hole`, then session 2 (`LAB_RUNBOOK.md`): shadow, air run, first insertions.
4. Optional: `force-weights` at a low pose, to explain 0.73 vs 1.09.
