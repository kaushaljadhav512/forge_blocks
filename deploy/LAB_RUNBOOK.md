# Lab runbook: peg insertion on the real Franka

A step-by-step guide for the robot sessions. It assumes you have run a FrankaPy program on this robot
before (for example, Lab2). Background is in [`README.md`](README.md) and
[`TEST_SUMMARY.md`](TEST_SUMMARY.md); you don't need them to run this.

| Session | Date | What | Status |
|---|---|---|---|
| 1 | 2026-11-14 | Setup and force-estimate checks, no insertions | Done: record below |
| 2 | 2026-11-16 | Joint friction measurements, no insertions | Done: record below |
| 3 | next | Register the hole, then insertions with `peg_fric_curr_seed2` | **To run** |

Sessions 1 and 2 are written as they were actually run (commands in order, with the results), so the
data in their zips can be traced back to a step. Session 3 is the plan for the next lab visit.

> **Safety, every time**
> - Two people: one runs the commands, one holds the **E-stop** the whole time the arm can move.
> - Every command that moves the arm needs `--ack-safety` and asks `[y/N]` first: type `y` to go on
>   (Enter alone or `n` cancels). **The arm moves as soon as you answer `y`**: be ready before you do.
> - Keep hands out of the workspace while the arm moves, except where a step tells you to guide it.
> - If anything looks wrong (unexpected motion, grinding, the peg slipping), press the E-stop first and
>   ask questions later. Then write down what happened.
> - Do not change any limit or default in the code. If a step fails, stop and report it.

---

## 0. Lab computer setup (every session)

**Code.** On the lab computer's **host** (not in the container), use the `lukas` branch:

```bash
cd forge_blocks
git checkout lukas && git pull
git status                     # must be clean: the lab copy must not have local edits
git log --oneline -1           # write down the commit
```

**On the 16662 lab PCs** (what worked in sessions 1 and 2):
- Run all `run_peg.py` commands **inside the course Docker container** (`bash run_docker.sh`, then
  `bash terminal_docker.sh`), from the repo under `src/devel_packages/<team>/`, which the container
  mounts (`/home/ros_ws/src/devel_packages/<team>/forge_blocks`). FrankaPy exists only in the container.
- Run `git pull` **on the host**: the container has no SSH keys. If the clone's remote is HTTPS,
  `ssh -A` won't help; switch it to `git@github.com:...` (ask your teammates first).
- Files written from the container are owned by root. If git or `script` on the host says
  "permission denied", run `chown -R 1001:1001 <repo>` in the container (check your host uid with
  `id -u`). Inside the container git may need `git config --global --add safe.directory <repo>`.
- Record each session with `script -af deploy/runs/session<N>_terminal.log` inside the container (`exit`
  stops it). It stops if the container closes; start it again with `-a` to keep appending. With `-f`
  a teammate can follow the log live (`ssh … tail -F <log>`).
- "Unable to init server: Connection refused", `Gdk-CRITICAL` and the `autolab_core` warning are
  printed by every command and are harmless (no display in the container).

Everything a session produces goes into `deploy/runs/`. At the end, zip that folder and send it back.

---

## Session 1 (2026-11-14): setup and force-estimate checks

**Goal:** check that the force estimate the policy relies on is usable (sign, gain, dead band, drift),
and that the bridge streams fast enough. No insertions. Policy for the timing test:
`exports/peg_forge_seed2` (copied from mrsd26c). Hole base: **9.0 mm** bore. Also brought: a kitchen
scale and water bottles of known weight.

**How it went.** The lab clone had uncommitted edits (one made `run_peg.py check` only echo its
arguments). The `press-test` and `force-curve` runs were made with that code before it was discarded;
everything after used the `lukas` branch. Two bridge fixes were made during the session (state
from the 100 Hz topic, tare after the stream starts), and `rate-test` and `force-drift` were re-run
after them. Step 1.8 (`register-hole`) was not run.

### 1.1 Versions and policy load

```bash
python3 --version
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import torch; print('torch', torch.__version__)"
pip show frankapy | head -3
python3 -c "import sys; sys.path.insert(0,'deploy'); from policy_runner import ForgePolicyRunner as R; R.from_export('exports/peg_forge_seed2'); print('policy loads OK')"
```

Result: Python 3.8.10, torch 2.4.1 (loads the torch 2.7 export), FrankaPy from a git checkout in the
container.

### 1.2 Robot up and read-only check

Robot started and FrankaPy run as in Lab2; Desk set to the Franka Hand with its standard load.

```bash
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py check
```

Result: under 3 N per axis at rest; Jacobian available; the topic state matches `get_pose()` exactly.
The at-rest force changes with pose and controller history (0.6 N low, 3.7 N at home, 1.8 N at home
after another command). The attempts tare after the controller starts, so this doesn't matter.

### 1.3 Force sign (press test), with the discarded lab code

The peg 1–5 cm above a free flat spot on the table:

```bash
python3 deploy/run_peg.py guide --seconds 20 --ack-safety
python3 deploy/run_peg.py press-test --ack-safety
```

Result: `force_sign = -1` (twice). Saved in `calibration.json`; session 2 confirmed it.

### 1.4 Force curve, with the discarded lab code

```bash
python3 deploy/run_peg.py force-curve --ack-safety
```

Result: its table is **not valid**: contact detection fires when the arm lags 3 mm behind its target,
and joint friction alone causes that in mid-air. What it did show: about 2 N of spring pull before the
arm moves, and 4–5 N to move at 4 mm/s.

### 1.5 Rate and delay

The arm holds still for 5 s, then the target steps 2 mm up.

```bash
python3 deploy/run_peg.py rate-test --export exports/peg_forge_seed2 --ack-safety
```

Run several times while fixing the bridge. Result: 53 Hz send / 29 Hz state before the fixes, **200 Hz
send / 100 Hz state** after. Policy 8 ms median, 53 ms max. The 2 mm step **never moved the arm**: it
pulls with only 1.1 N, below the breakaway friction.

### 1.6 Grasp

```bash
python3 deploy/run_peg.py open --ack-safety
# peg vertical between the fingers: finger ENDS on the 32 mm line (hardware/README.md)
python3 deploy/run_peg.py grasp --ack-safety
```

Result: the lab gripper's indented fingers hold the peg at **6.3–6.4 mm** (`grasp` then accepted 6–10 mm;
now 3.5–10 mm, since the divots that centre the peg can close the fingers to about 4 mm).
It took a few tries: `grasp` sometimes read the width mid-close (~45 mm) and refused. Fixed after the
session (it now waits for the fingers to stop).

### 1.7 Force estimate checks

**a) `force-weights`** (at home, peg at least 15 cm above anything, a string loop tied below the
fingers). Masses hung one after another, the total typed in grams each time, `0` at the start and end:

```bash
python3 deploy/run_peg.py force-weights --ack-safety
```

Result: estimate = **0.73 × true**, dead band 0.13 N, sign agrees with the press test, 0.12 N drift
over the test. The arm sags about 9 mm per 500 g (565 N/m spring).

**b) `force-scale`** (low pose, peg about **5 mm** above a zeroed kitchen scale on a free spot; from
1–5 cm the friction makes the contact detection fire before the peg reaches the scale). At each of 8
levels (1–10 N) the scale reading in grams is typed in:

```bash
python3 deploy/run_peg.py guide --seconds 60 --ack-safety
python3 deploy/run_peg.py force-scale --ack-safety
```

Result: estimate ≈ **1.09 × scale + 0.6 N**, no dead band (the key number). The scale responds from
about 2.2 N commanded. Why pressing (1.09, low pose) and hanging (0.73, home) differ is unexplained;
insertion is pressing at a low pose, so 1.09 is the one that applies.

**c) `force-drift`** (from home, at least 5 cm clear below the peg). Holds 15 s, then moves 4 cm down
and back:

```bash
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py force-drift --ack-safety
```

Run four times. The first sat at -1.74 N in z from its first sample (it tared before the controller
started); after the fix (tare once the stream runs), three more runs. Result: hold **within 0.15 N**;
the controller start shifts the estimate by 0.3–0.6 N (different each start, removed by the tare);
during the sweep the estimate reads **1.5–2.4 N opposing the motion**: joint friction leaks into it.

### 1.8 Register the hole: not run

### 1.9 Finish

```bash
python3 deploy/run_peg.py open --ack-safety
exit
zip -r session1_$(date +%Y%m%d).zip deploy/runs
```

**Outcome:** sensing is fine (no dead band, gain near 1 when pressing, negligible drift). **Joint
friction is the problem**: small target moves barely move the arm, and friction shows up in the force
estimate as a force opposing the motion. Hence session 2.

---

## Session 2 (2026-11-16): joint friction

**Goal:** measure the joint friction so it can be modelled in the sim (or compensated). No insertions.
About 1 h, 12:18–13:21 local, recorded in `deploy/runs/session2_terminal.log`. Code: `lukas`
(`1dadb13`).

At the start, the joint torques were confirmed to be in FrankaPy's state message
(`tau_J`, `tau_ext_hat_filtered`), so every friction command also logged `q`, `dq` and the torques.

**How it went.** All of the planned measurements ran, with three differences from the plan: the
high-pose speed sweep ran **x three times** (no y or z at the high pose; y and z were measured at the
low pose in 2.7); the second mid-session walk kept the label `mid` (its file has a later timestamp);
and `register-hole` was **skipped on purpose**, because the hole base was going to be re-clamped.

### 2.1 Robot up, check, rate, grasp

```bash
git pull                                       # in the container this needed the safe.directory fix
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py check
python3 deploy/run_peg.py rate-test --export exports/peg_forge_seed2 --ack-safety
python3 deploy/run_peg.py grasp --ack-safety   # finger ENDS on the 32 mm line
```

Result: 0.3° from "gripper down", force under 1 N per axis at home, `state source: topic`. Rate: 200 Hz
send, 98.6 Hz state, policy 8 ms median (78 ms max), 2 mm step: no motion (as in session 1). Grasp
width 6.38 mm.

### 2.2 Random walk, cold

First, while the joints were cold. The peg wanders for 3 min over the area around the hole like a
policy would (random directions, 1–20 mm/s, stops, reversals, 0.5–8 mm jumps), in free space, so
everything the force estimate reads is error. The same `--seed` makes exactly the same motion, so it
can be repeated later and replayed in sim. Area (robot base frame): x 0.40–0.68 m, y ±0.15 m, peg
bottom 9–27 cm above the table. It starts from `home`, moves across at home height, then straight down
into the area.

```bash
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py friction-walk --seed 0 --label cold --ack-safety
```

Result: full 184 s. Printed fit, fake force opposing the motion (moving + / moving −): x 2.01 / 0.88 N,
y 0.61 / 0.65 N, z 0.63 / −0.12 N. (This printed fit uses a speed-based model, which later turned out
to be misleading for x; see the outcome below.)

### 2.3 Force sign (re-check)

Session 1's sign came from discarded code. Peg 1–5 cm above a free flat spot:

```bash
python3 deploy/run_peg.py guide --seconds 20 --ack-safety
python3 deploy/run_peg.py press-test --ack-safety
```

Result: `force_sign = -1`, confirmed.

### 2.4 High pose: drift, breakaway (stiction), speed sweep, small steps

Peg guided to the high pose (planned: about **15 cm above the hole**, 10 cm clear all around).

```bash
python3 deploy/run_peg.py guide --seconds 20 --ack-safety
python3 deploy/run_peg.py force-drift --seconds 30 --sweep_m 0 --ack-safety
python3 deploy/run_peg.py friction-breakaway --ack-safety       # three times in a row
python3 deploy/run_peg.py friction-breakaway --ack-safety
python3 deploy/run_peg.py friction-breakaway --ack-safety
python3 deploy/run_peg.py friction-sweep --axis x --speeds 2 5 10 20 --ack-safety   # three times in a row
python3 deploy/run_peg.py friction-sweep --axis x --speeds 2 5 10 20 --ack-safety
python3 deploy/run_peg.py friction-sweep --axis x --speeds 2 5 10 20 --ack-safety
python3 deploy/run_peg.py friction-steps --steps_mm 0.5 1 2 4 8 --ack-safety
```

- `force-drift` (hold only): max |f| 0.10 / 0.14 / 0.15 N, no drift; controller start shift up to 0.24 N.
- `friction-breakaway`: the target creeps away at 0.5 mm/s until the fingertip moves 0.5 mm; the
  spring pull then is the breakaway force. Runs 1 / 2 / 3 (N):

  | +x | −x | +y | −y | +z (up) | −z (down) |
  |---|---|---|---|---|---|
  | 2.31 / 2.53 / 2.38 | 1.14 / 1.43 / 1.42 | 0.84 / 1.00 / 1.03 | 1.07 / 1.09 / 1.30 | 2.95 / 2.61 / 2.78 | 1.03 / 1.21 / 1.17 |

- `friction-sweep` x (3 cm and back at 2, 5, 10, 20 mm/s): friction fit Fc = 3.47 / 3.81 / 3.41 N,
  fake force along the motion −0.6 to −1.6 N. The hand covered under 1 cm of each 3 cm stroke in x, so
  the printed "pose error" column is a fit artefact.
- `friction-steps` (target jumps of 0.5–8 mm held 1.5 s; distance moved, mm):

  | Step (mm) | 0.5 | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|---|
  | +x / −x | 0.01 / 0.01 | 0.08 / 0.15 | 0.13 / 0.68 | 0.53 / 1.43 | 1.93 / 3.30 |
  | +y / −y | 0.01 / −0.06 | 0.07 / 0.00 | 0.54 / 0.07 | 2.16 / 1.20 | 5.67 / 4.97 |
  | +z / −z | −0.15 / 0.38 | −0.02 / 0.38 | −0.01 / 0.58 | 0.16 / 1.89 | 3.41 / 4.09 |

  Steps of 2 mm or less barely move the arm; even 8 mm gets only 2–6 mm.

### 2.5 Random walk, mid-session, twice

```bash
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py friction-walk --seed 0 --label mid --ack-safety
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py friction-walk --seed 0 --label mid --ack-safety    # planned label: mid2
```

Result: fits within about 0.15 N of the cold walk (x 2.17 / 0.82 and 2.16 / 0.87 N).

### 2.6 Low pose: drift, breakaway, one-speed sweeps

Peg guided to the low pose (planned: about **5 cm above the hole base's top**, the hole about 10 cm to
one side): roughly where the attempts start.

```bash
python3 deploy/run_peg.py guide --seconds 20 --ack-safety
python3 deploy/run_peg.py force-drift --seconds 30 --sweep_m 0 --ack-safety
python3 deploy/run_peg.py friction-breakaway --ack-safety
python3 deploy/run_peg.py friction-sweep --axis x --speeds 5 --ack-safety
python3 deploy/run_peg.py friction-sweep --axis y --speeds 5 --ack-safety
python3 deploy/run_peg.py friction-sweep --axis z --speeds 5 --ack-safety
```

- `force-drift`: max |f| 0.10 / 0.03 / 0.10 N, no drift; start shift −0.50 N in x.
- Breakaway (N): +x 2.93, −x 0.94, +y 1.01, −y 1.04, +z 1.88, −z 0.99: like the high pose.
- Sweeps at 5 mm/s: friction 4.69 / 2.06 / 3.23 N (x / y / z), fake force along the motion −1.15 /
  −0.50 / −0.89 N.

### 2.7 Stiffness × 1.3 (low pose)

```bash
python3 deploy/run_peg.py friction-steps --steps_mm 0.5 1 2 4 8 --stiffness_scale 1.3 --ack-safety
```

Result: somewhat further (for example +y 1.15 / 2.99 / 7.54 mm at 2 / 4 / 8 mm, vs 0.54 / 2.16 /
5.67 mm), but at most 1.2 mm for a 2 mm step and still nothing at 0.5–1 mm in most directions. Not a
fix on its own.

### 2.8 Random walk, warm, and a second walk

```bash
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py friction-walk --seed 0 --label warm --ack-safety
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py friction-walk --seed 1 --label warm --ack-safety
```

Result: seed 0 warm matches the cold walk within about 0.15 N, so warm-up barely matters. Seed 1 is a
different walk, kept as a hold-out for checking the sim's friction model (its printed x fit flips to
1.16 / 2.72 N, which is why the speed-based fit can't be trusted for x).

### 2.9 Register the hole: skipped on purpose (the base was to be re-clamped; done in session 3)

### 2.10 Finish

```bash
python3 deploy/run_peg.py open --ack-safety
python3 deploy/run_peg.py home --ack-safety
exit
zip -r session2_$(date +%Y%m%d).zip deploy/runs
```

**Outcome:** while moving, the hand lags its target as if the friction were about 4.6 / 1.9 N in x
(+/−), 1.5 / 2.0 N in y and 2.1 / 1.0 N in z. The fake force in the estimate follows the controller's
spring pull rather than the speed: about 0.25–0.43 of the pull leaks in, plus an offset near −1 N in x
and z. Both went into the sim (fingertip friction and a pull-based leak), and the session-3 policy was
trained with them.

---

## Session 3: insertions (to run)

**Goal:** register the hole, then run the friction-trained policy `peg_fric_curr_seed2` and measure its
success rate. About **2 h**: setup 20 min, hole registration 10 min, first runs 20 min, evaluation
about 1 h.

> During an attempt the bridge stops the arm by itself (a **safety stop**) if the force goes over
> 25 N, the fingertip leaves a 4 cm box around the hole or goes below the seated depth, or it moves
> faster than 0.3 m/s. In sim this policy peaks around 10–11 N and 0.18 m/s, so a safety stop means
> something differs from the sim: stop and report it.

### Before you go

Do section 0 (`git pull`). Then copy the policy: it isn't in git (53 MB). Copy the folder
`exports/peg_fric_curr_seed2/` from Lukas's laptop (USB stick or `scp`) to `forge_blocks/exports/` on
the lab computer, and check it arrived intact:

```bash
ls exports/peg_fric_curr_seed2         # policy.pt  runner_config.json  agent.yaml  env.yaml  SOURCE_CHECKPOINT
sha256sum exports/peg_fric_curr_seed2/policy.pt
# must be 549984a7b1805dbe90644288393f5b7bbd7033e781c8165ff42c80edbcbf295c
```

**Bring:**
- [ ] The printed peg (Ø7.98 × 50 mm) and the hole base with the **9.0 mm** bore. See
      [`../hardware/README.md`](../hardware/README.md).
- [ ] 2 clamps for the hole base.
- [ ] A phone to film the attempts.

Write each result, or **PASS / FAIL**, in the boxes.

### 3.1 Start recording, policy loads

```bash
cd forge_blocks
mkdir -p deploy/runs
script -af deploy/runs/session3_terminal.log
python3 -c "import sys; sys.path.insert(0,'deploy'); from policy_runner import ForgePolicyRunner as R; R.from_export('exports/peg_fric_curr_seed2'); print('policy loads OK')"
```

- [ ] Prints `policy loads OK`. If not, copy the error and stop.

### 3.2 Robot up and read-only check

1. Start the robot, unlock the joints in Desk and start FrankaPy **exactly the way you did for Lab2**.
2. In Desk, check that the end-effector is set to the **Franka Hand** with its standard load.

```bash
python3 deploy/run_peg.py home --ack-safety      # gripper pointing down, above the table
python3 deploy/run_peg.py check
```

- [ ] `angle from 'gripper down, yaw 0'` is under about 5°.
- [ ] The raw force has each axis under **3 N** while nothing touches the arm.
- [ ] The last line says `state source: topic` with a small age. Otherwise stop and report it.
- [ ] The calibration it prints includes `force_sign` (-1 in sessions 1 and 2). If there is none,
      `deploy/runs/` was cleared: do the press test in 3.4 before anything else.

### 3.3 Hole base and peg

1. Clamp the hole base flat on the table so that its bore is about **60 cm in front of the robot base
   and centred left-right** (±5 cm is fine). It must not move when pushed. **From here on, don't touch
   the base or the clamps**: if it moves, redo 3.5.
2. Grasp the peg:
   ```bash
   python3 deploy/run_peg.py open --ack-safety
   # hold the peg vertically between the fingers: finger ENDS on the 32 mm line (hardware/README.md)
   python3 deploy/run_peg.py grasp --ack-safety     # 60 N; --force 70 is the maximum
   ```
- [ ] The printed width is between 0.0035 and 0.010 m (6.3–6.4 mm in sessions 1 and 2; the finger
      divots centre the peg and can close to about 4 mm) and the peg sits vertically and doesn't wiggle
      when you touch it lightly. Below 3.5 mm `grasp` refuses: the fingers missed the peg.

### 3.4 Force sign (only if 3.2 found no `force_sign`)

Guide the peg **1–5 cm above a free flat spot on the table, at least 15 cm from the hole**:

```bash
python3 deploy/run_peg.py guide --seconds 20 --ack-safety     # move the arm by hand, then let go
python3 deploy/run_peg.py press-test --ack-safety              # don't touch the arm
```

- [ ] `force_sign = -1` is printed and saved. **If it says +1, stop and report it.**

### 3.5 Register the hole

With the peg grasped:

```bash
python3 deploy/run_peg.py register-hole --ack-safety
```

This repeats 5 times:
1. Enter starts guide mode for 15 s. Push the peg **fully into the hole** by hand, gently, until it
   bottoms out, then hold it still until guide mode ends. The arm then holds the peg in the hole.
2. It asks whether you're ready to lift: Enter starts guide mode for 10 s, lift the peg out.
3. It asks you to confirm the peg is clear and still at the 32 mm line. If the peg slipped in the
   fingers, Ctrl-C, regrasp (3.3) and start over: a slipped peg shifts every sample.

- [ ] The printed `spread` is **under 0.5 mm**. If not, check the clamps and run it again.
- [ ] Write down the `hole rim` it prints.

Every later command uses this registration (`deploy/runs/hole.json`).

### 3.6 Shadow (policy runs, arm doesn't)

Film every attempt from here on, from the side, close enough to see the peg and the hole.

```bash
python3 deploy/run_peg.py shadow --export exports/peg_fric_curr_seed2
```

Guide mode for 30 s: move the peg around **above** the hole by hand. It prints the policy's targets
relative to the fingertip.

- [ ] When the peg is beside the hole, the targets point toward it. When it is above the hole, they
      point down.

### 3.7 Air run (no contact possible)

The policy is told the hole is 10 cm higher than it is, so it moves but can't touch anything.

```bash
python3 deploy/run_peg.py run --export exports/peg_fric_curr_seed2 --air --trials 1 --ack-safety
```

- [ ] The motion is smooth and slow. It ends with a timeout or a success prediction, **not** a safety
      stop.
- [ ] Note whether the arm **stalls** (stops while the policy keeps commanding motion) or moves in
      jerks (stick-slip). Session 2 found that small target moves barely move the arm.

### 3.8 First real insertions (perfect hole estimate)

```bash
python3 deploy/run_peg.py run --export exports/peg_fric_curr_seed2 --band 0 0 --trials 3 --ack-safety
```

Before each trial it asks for confirmation: check the peg is out of the hole and the E-stop is ready.
After each trial the arm pulls the peg 6 cm up.

Each trial prints `-> <outcome>, success=<True/False>, ..., peak F=<N>`. The outcome is how the attempt
ended: `seated` (the peg was seated for 0.3 s), `predicted_success` (the policy said it's in first),
`timeout` or `safety_stop`. `success` is whether the peg was seated at any point, from the registered hole
(like the sim's success); the prediction and `success` can disagree.

| Trial | Outcome | `success` | Peak force (N) | Notes (stalls, sliding on the rim, ...) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**After any safety stop, stop and report it.**

> Since session 3's first try (trial 2 tilted to 37° and the robot state froze): the arm now aims the
> gripper straight down instead of following its own tilt, and an attempt stops on a tilt over 10° or
> on robot state that hasn't changed for 100 ms. It also ends each attempt 0.3 s after the peg is
> seated: right after seating the policy shoved sideways into the bore wall and tilted the arm.
> During attempts the wrist is stiffer than in sim (80 instead of 28 Nm/rad), so it starts upright and
> side loads tip it less. Re-run 3.8 with this before 3.9.

### 3.9 Evaluation (only if 3.8 had at least 1 success out of 3)

Three error bands (how far the hole estimate given to the policy is from the true hole), 20 trials
each, about 15–20 min per band:

```bash
python3 deploy/run_peg.py run --export exports/peg_fric_curr_seed2 --band 0 1   --trials 20 --ack-safety
python3 deploy/run_peg.py run --export exports/peg_fric_curr_seed2 --band 1 2.5 --trials 20 --ack-safety
python3 deploy/run_peg.py run --export exports/peg_fric_curr_seed2 --band 2.5 5 --trials 20 --ack-safety
python3 deploy/run_peg.py summary
```

- Check the grasp between bands. If the peg slipped in the fingers (it sits visibly higher or lower),
  `open`, re-grasp on the line, and continue.
- A series stops on its own after a safety stop. Note what happened, and continue with the same
  command only if the cause is clear and fixed.

| Band (mm) | Success (sim, with the session-2 friction) | Success (robot) | Mean peak force (sim / robot) | Safety stops |
|---|---|---|---|---|
| 0–1 | 99.3 % | | 10.5 N / | |
| 1–2.5 | 96.3 % | | 10.5 N / | |
| 2.5–5 | 86.0 % | | 10.7 N / | |

### 3.10 Finish and send back

```bash
python3 deploy/run_peg.py open --ack-safety    # catch the peg
exit                                           # stops the terminal recording
zip -r session3_$(date +%Y%m%d).zip deploy/runs
```

Send the zip, the videos, the `summary` output, this page with the boxes filled in, the commit from
section 0, and anything odd you noticed.

---

## Quick reference

| Command | Moves? | What it does |
|---|---|---|
| `check` | no | State, poses, raw force, Jacobian, calibration so far |
| `home` | yes | FrankaPy's home pose |
| `guide --seconds N` | by hand | Guide mode for N s |
| `open` / `grasp` | gripper | Open / close on the peg |
| `rate-test --export E` | 2 mm | Stream rate, delay, policy time |
| `press-test` | contact | Force sign → `calibration.json` |
| `force-curve` | contact | Estimate vs commanded push, 0–15 N (contact detection unreliable with friction) |
| `force-scale` | contact | Commanded vs kitchen scale vs estimate |
| `force-weights` | holds still | Hung masses vs estimate; second sign check |
| `force-drift [--sweep_m 0]` | 4 cm, free space | Drift while holding; with the sweep, error while moving |
| `friction-breakaway` | a few mm, free space | Spring pull needed to start moving, 6 directions |
| `friction-sweep --axis A --speeds ...` | 3 cm and back, free space | Friction vs speed and the fake force in the estimate |
| `friction-steps --steps_mm ... [--stiffness_scale S]` | up to 8 mm, free space | How far policy-sized target steps get (S in 0.7–1.3) |
| `friction-walk --seed N --label L` | 3 min over the hole area, free space | Seeded random walk; target, fingertip, force and joints at 100 Hz |
| `register-hole` | by hand | Saves the hole position → `hole.json` |
| `shadow --export E` | by hand | Policy runs, nothing commanded |
| `run --export E --air` / `--band A B` | yes | Policy attempts → `results.jsonl` |
| `summary` | no | Success rate, time, peak force and safety stops per band |

The free-space checks stop with a safety stop if the force goes over 10 N (something touched the peg:
check the clearance). Everything is saved in `deploy/runs/`: `calibration.json` (force sign and the
summary of each check), `hole.json`, `results.jsonl` (one line per attempt), `attempts/` (per-step
logs), the raw data of each check (`force_*.json`, `friction_*.json`) and `events.jsonl` (a log of
the commands that were run).

**Problems:**
- `no force_sign yet: run press-test first`: do 3.4.
- `operator cancelled`: you answered `n` or just pressed Enter at a `[y/N]` question. Run it again.
- A safety stop during an attempt on **force**: the peg jammed or pressed hard. On the **box** or
  **z**: the peg slipped in the fingers, or the hole base moved (redo 3.5). On **speed**: report it;
  the sim never gets near the limit.
- `robot state is … ms old`: the state topic stopped. Check the three FrankaPy terminals.
- A command crashes in `FrankaPyArm`: copy the full error into your notes and stop.
- `run_peg.py check` prints only its own arguments: someone edited the lab copy. Check `git status`.
