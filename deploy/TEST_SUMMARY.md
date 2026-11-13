# Test summary: FORGE peg insertion, sim to real

Tests run so far (as of 2026-11-13) to get the FORGE peg-insertion policy (`Isaac-ForgeBlocks-Peg8mm-Direct-v0`,
8 mm peg) ready for the real Franka. Everything below ran in simulation or against a fake arm. **Nothing
has run on the real robot yet.**

| # | Test | Where | Result |
|---|---|---|---|
| 1 | Unit tests | laptop + mrsd26c container | pass |
| 2 | Sim bridge equivalence | mrsd26c, Isaac Lab | pass, all 3 seeds |
| 3 | Frame and sign measurements | mrsd26c, Isaac Lab | done, constants used in the bridge |
| 4 | Baseline evaluation, 3 seeds | mrsd26c, Isaac Lab | 82–99% success |
| 5 | Sim-to-real stress evaluations | mrsd26c, Isaac Lab | force offset and dead band are the risks |

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

## Still to do on the real robot

In order, using `deploy/run_peg.py`:

1. Version checks on the lab PC: Python, torch (the export used 2.7.0), `pip show frankapy`.
2. `check`: read-only state, force reading, and whether FrankaPy provides a Jacobian.
3. `rate-test`: stream and state rates, policy inference time, and 2 mm step latency (first motion
   and 63% of the step).
4. `press-test`: force sign; it must read +z when pressing, as in sim (test 3).
5. `force-curve`, then `force-scale` (kitchen scale), `force-weights` (hung masses) and `force-drift`
   (15 s hold + 4 cm move in free space). These give the real sensing dead band, the actuation loss,
   a second sign check, and the drift, to compare against the stress results in test 5.

**Decision after step 5:** if the dead band is ≥ 2 N or the offset drifts more than ~1 N, retrain with a
random dead band (0–5 N) and a random force offset during training (~1h45m per seed on mrsd26c).
Otherwise, continue with the seed 2 checkpoint into the shadow and air runs (phase 4 in `README.md`).
