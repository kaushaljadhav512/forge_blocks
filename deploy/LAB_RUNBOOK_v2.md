# Lab runbook v2: joint friction, then insertions

This is the follow-up to [`LAB_RUNBOOK.md`](LAB_RUNBOOK.md) (v1), for the lab sessions after the first
robot session on 2026-11-14. Setup, safety rules and the basic commands are the same as in v1. Where a
step is the same, this page points to the v1 section instead of repeating it.

**Why this session.** Session 1 ([`TEST_SUMMARY.md`](TEST_SUMMARY.md), test 6) showed that force
**sensing** is fine: no dead band, gain ≈ 1.09, drift under 0.15 N. The problem is **joint friction**:
- A 2 mm target step (1.1 N of spring pull) didn't move the arm at all. About 2 N is needed to start
  moving, and 4–5 N to move at 4 mm/s.
- While moving, friction shows up in the force estimate as a fake force opposing the motion (1.5–2.4 N
  in the vertical sweep).

The policy works in exactly this range. The spring is 565 N/m, so a target 2–8 mm from the fingertip
pulls with only 1–4.5 N. The sim has no friction. This session measures the friction so we can
**(a)** put a matching friction model into the sim and retrain, or **(b)** compensate for it in the
controller. Each test below gives one number that model needs.

| Test | Measures | Feeds |
|---|---|---|
| 2.3 `friction-breakaway` | Force needed to start moving (stiction), per axis and direction | sim stiction; controller compensation |
| 2.4 `friction-sweep` | Friction while moving vs speed (Coulomb + viscous), and how much leaks into the force estimate | sim friction model; fake-force observation noise |
| 2.5 `friction-steps` | How far short of small, policy-sized targets the arm stops, and how fast it gets there | whether the policy can position precisely enough |
| 2.6 Pose repeat | Whether the friction changes between the hole height and higher up | whether one model is enough |
| 2.3a / 2.6 `force-drift` (hold only) | Drift of the force estimate while holding still, at both poses (session 1: one run, one pose) | sim force-offset risk: needs well under 1 N over an attempt |
| 2.7 Stiffness × 1.3 | Whether a stiffer spring shrinks the stuck zone (sim: ±30% stiffness is harmless) | cheapest fix to try first |

There are two sessions:
- **Session 2 (about 1.5 h): friction measurements, no insertions.** Then send the results back.
- **Session 3 (about 2 h): insertions**, with whatever fix session 2 leads to. This is v1's "Session 2".

> **Safety, every time.** Same as v1: two people, one on the **E-stop** whenever the arm can move;
> `--ack-safety` on every moving command; **`rate-test` and `guide` start the moment you press Enter**;
> hands out of the workspace; E-stop first, questions later. Do not change any limit or default in the
> code. If a step fails, stop and report it.
>
> The friction tests run **in free space**. Nothing should touch the peg, so any force over 10 N
> triggers a safety stop. Before each one, check that there is **at least 10 cm clear around the peg in
> every direction** (more below it for z).

---

## 0. Before you go to the lab

**Code.** Use `main` (the `lukas` fixes are merged into it). On the **host**, not in the container:

```bash
cd forge_blocks
git checkout main && git pull
git status                     # must be clean: the lab copy must not have local edits
git log --oneline -1           # write down the commit
```

> **Check the friction commands are there.** In the container, `python3 deploy/run_peg.py --help` must
> list `friction-breakaway`, `friction-sweep` and `friction-steps`. If they are missing, the pull didn't
> get the new code: **don't go**, because without them this session only repeats session 1.

> **Check the joint torques reach the logs (not yet verified on the robot).** The friction commands
> also save `q`, `dq` and the joint torques for a later per-joint friction fit. The torques are read
> from the state message as `tau_J` and `tau_ext_hat_filtered`, libfranka's names; nobody has confirmed
> that FrankaPy's message uses them. With the controller running, in the container:
>
> ```bash
> rostopic echo -n1 /robot_state_publisher_node_1/robot_state | grep -E "^(tau_J|tau_ext_hat_filtered):"
> ```
>
> - [ ] Both fields are printed. If not, write down the torque field names that `rostopic echo -n1`
>   does print. The tests still run, but their `joints` rows will have `None` for the torques.

**Policy:** the same `exports/peg_forge_seed2` as v1 (see v1 §0 for the `scp`). It's only needed for
session 3.

**Bring:**
- [ ] The printed peg and the hole base (**9.0 mm** bore) from session 1, and 2 clamps.
- [ ] A tape measure or ruler (for 2.6).
- [ ] Phone to film the attempts (session 3).

The kitchen scale and the weights are not needed.

**On the 16662 lab PCs:** same as v1 §0 (run commands in the course Docker container, `git pull` on the
host, `chown` if git says "permission denied").

---

## Session 2: joint friction (no insertions)

```bash
cd forge_blocks
mkdir -p deploy/runs
script -af deploy/runs/session2_terminal.log    # `exit` to stop
```

Write each result, or **PASS / FAIL**, in the boxes. Every friction command also saves its full data in
`deploy/runs/friction_*.json`, so the tables here are only a summary for a quick look.

### 2.1 Robot up, check, grasp

Do v1 §1.2 and §1.3 (`home`, `check`), then grasp the peg as in v1 §1.5:

```bash
python3 deploy/run_peg.py home --ack-safety
python3 deploy/run_peg.py check
python3 deploy/run_peg.py open --ack-safety
# finger ENDS on the 32 mm line
python3 deploy/run_peg.py grasp --ack-safety
```

- [ ] `check`: angle under about 5°, each raw force axis under 3 N, `state source: topic`.
- [ ] Grasp width **6.3–6.4 mm**, and the peg doesn't wiggle.

If the hole base was moved since session 1, clamp it again as in v1 §1.5.

### 2.2 Force sign (quick re-check)

Session 1's `force_sign −1` came from lab code that has since been discarded. Move the peg 1–5 cm above
a **free flat spot** at least 15 cm from the hole, then:

```bash
python3 deploy/run_peg.py guide --seconds 20 --ack-safety
python3 deploy/run_peg.py press-test --ack-safety      # don't touch the arm
```

- [ ] `force_sign` = ______. If it is **not −1**, carry on, but report it.

**For 2.3–2.5:** use `guide` to move the peg to the **high pose**, about **15 cm above the hole**,
with 10 cm clear all around. Write down roughly where it is: ______ cm above the table.

### 2.3 Breakaway force (stiction)

For each of the 6 directions (+x, −x, +y, −y, +z, −z), the target moves away from the fingertip slowly
(0.5 mm/s, so the spring pull rises by 0.28 N/s) until the fingertip moves 0.5 mm, then returns. The
pull at that moment is the breakaway force. It stops at 8 N if the arm never moves. The motion is a few
mm per direction; about 3 min in total.

**2.3a First, the drift hold** (high pose). Session 1 measured this once. `--sweep_m 0` skips the
vertical sweep, which `friction-sweep` now covers.

```bash
python3 deploy/run_peg.py force-drift --seconds 30 --sweep_m 0 --ack-safety
```

| Output | Want | Session 1 | Got (x / y / z) |
|---|---|---|---|
| hold: max \|f\| | < 1 N | ≤ 0.15 N | |
| hold: drift (N/s) | < 0.03 | | |
| stream start shift | (just record it) | 0.3–0.6 N | |

Then the breakaway test:

```bash
python3 deploy/run_peg.py friction-breakaway --ack-safety
```

| Direction | +x | −x | +y | −y | +z (up) | −z (down) |
|---|---|---|---|---|---|---|
| Breakaway pull (N) | | | | | | |
| Force estimate at breakaway (N) | | | | | | |

Session 1 rough value: ~2 N in z.

### 2.4 Friction while moving (speed sweep)

For each axis, the target moves 3 cm and back at a constant speed, for each speed in turn. Once the arm
is moving steadily, the lag behind the target × 565 N/m is the friction force, and the force estimate
shows how much of it leaks into the observation. About 4 min in total.

```bash
python3 deploy/run_peg.py friction-sweep --axis x --speeds 2 5 10 20 --ack-safety
python3 deploy/run_peg.py friction-sweep --axis y --speeds 2 5 10 20 --ack-safety
python3 deploy/run_peg.py friction-sweep --axis z --speeds 2 5 10 20 --ack-safety
```

Speeds are in mm/s. Each command prints a row per speed and direction, plus a fit at the end.

| Axis | Friction fit: `F = Fc + b·v` | Fake force in the estimate, along the motion (N, at 5 / 20 mm/s) | Fake force across the motion (N, worst) | Position-dependent error (N/cm) |
|---|---|---|---|---|
| x | Fc = ___ N, b = ___ N/(m/s) | | | |
| y | Fc = ___ N, b = ___ N/(m/s) | | | |
| z | Fc = ___ N, b = ___ N/(m/s) | | | |

The sweep goes both ways over the same 3 cm, so it separates two errors. Friction flips sign with the
direction of motion. An error that depends only on position doesn't. The last column is that
position-dependent part (what `force-drift`'s one-way sweep mixed with friction in session 1).

Session 1 rough value in z: 1.5–2.4 N of fake force, opposing the motion. **The x and y columns are the
most important ones**: the policy finds the hole using sideways force.

### 2.5 Small steps (policy-sized targets)

For each axis and direction, the target jumps by 0.5, 1, 2, 4 and 8 mm (0.3–4.5 N of pull), is held for
1.5 s, then jumps back. This is the kind of motion the policy commands near the hole. About 2 min.

```bash
python3 deploy/run_peg.py friction-steps --steps_mm 0.5 1 2 4 8 --ack-safety
```

| Step (mm) | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| x: distance moved after 1.5 s (mm) | | | | | |
| y: distance moved after 1.5 s (mm) | | | | | |
| z: distance moved after 1.5 s (mm) | | | | | |
| Worst time to 63% (ms) | | | | | |

`rate-test`'s 2 mm step gave **no motion** in session 1. The hole tolerance is ±0.5 mm (8.0 mm peg,
9.0 mm bore), and a success needs the peg within 2.5 mm of the axis. If the arm stops more than ~1 mm
short at 2–4 mm steps, the policy can't position the peg precisely.

### 2.6 Pose repeat: at the hole height

Joint friction depends on the arm's posture, and session 1's at-rest force changed with pose (0.6 N
low, 3.7 N at home). Use `guide` to move the peg to the **low pose**: about **5 cm above the hole
base's top**, with the hole about 10 cm to one side so nothing is below or beside the peg within the
sweep range. This is roughly where the attempts start. Then repeat the breakaway test and one speed of
the sweep:

```bash
python3 deploy/run_peg.py force-drift --seconds 30 --sweep_m 0 --ack-safety
python3 deploy/run_peg.py friction-breakaway --ack-safety
python3 deploy/run_peg.py friction-sweep --axis x --speeds 5 --ack-safety
python3 deploy/run_peg.py friction-sweep --axis y --speeds 5 --ack-safety
python3 deploy/run_peg.py friction-sweep --axis z --speeds 5 --ack-safety
```

The z sweep moves **up** 3 cm and back down to the start (every sweep goes along +axis and returns), so
it needs free space above the peg, not below. It is the most important one here: the policy descends
onto the hole, and the vertical fake force is what read as contact in session 1.

| | +x | −x | +y | −y | +z | −z |
|---|---|---|---|---|---|---|
| Breakaway, low pose (N) | | | | | | |
| Breakaway, high pose (from 2.3) (N) | | | | | | |

- [ ] The x / y / z fake force at 5 mm/s, low pose: ______ / ______ / ______ N.
- [ ] `force-drift` hold, low pose: max |f| ______ N, drift ______ N/s (want < 1 N, < 0.03 N/s).

### 2.7 Stiffness × 1.3

In sim, a 30% stiffer spring costs nothing (TEST_SUMMARY test 5). On the robot, a stiffer spring pulls
harder for the same target offset, so it may get through friction better. Stay at the low pose:

```bash
python3 deploy/run_peg.py friction-steps --steps_mm 0.5 1 2 4 8 --stiffness_scale 1.3 --ack-safety
```

| Step (mm) | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Worst axis: distance moved after 1.5 s, × 1.3 (mm) | | | | | |

### 2.8 Register the hole

As in v1 §1.8 (skipped in session 1). With the peg still grasped:

```bash
python3 deploy/run_peg.py register-hole --ack-safety
```

Five times: in guide mode, push the peg **fully into the hole** by hand until it bottoms out and hold
it still. Then lift it out when asked.

- [ ] `spread` is **under 0.5 mm**. If not, check the clamps and run it again.
- [ ] **Don't move the hole base or the clamps after this.** Session 3 uses this registration.

### 2.9 Finish and send back

```bash
python3 deploy/run_peg.py open --ack-safety    # catch the peg
exit
zip -r session2_$(date +%Y%m%d).zip deploy/runs
```

Send the zip, this page with the boxes filled in, the commit from §0, and anything odd (grinding,
jerky stick-slip motion, a direction that behaves very differently). **Don't start session 3** until
we've looked at the numbers. What happens next:

| Result | Plan |
|---|---|
| Drift hold over 1 N at either pose (2.3a, 2.6) | Report it first: offset is the biggest sim risk, and it must be fixed before anything else |
| Steps reach within ~1 mm at 2 mm (2.5), and sideways fake force under ~1 N (2.4) | Session 3 with the current policy |
| Stiffness × 1.3 fixes the steps (2.7), and fake force under ~1 N | Session 3 with stiffness × 1.3 (the sim says it's harmless) |
| Otherwise | Retrain with the measured friction (Fc, b, stiction, fake force) in the sim, and/or add friction compensation to the controller; then session 3 |

---

## Session 3: insertions (only after the go-ahead)

Same as v1 "Session 2" (§2.1–2.5), with these changes:

- Use the policy folder and the stiffness we tell you to use (they may change after session 2). Replace
  `exports/peg_forge_seed2` below with that folder.
- Record to `deploy/runs/session3_terminal.log` and zip as `session3_<date>.zip`.
- Use the `hole.json` from 2.8. If the hole base moved, redo 2.8 first.

```bash
script -af deploy/runs/session3_terminal.log
python3 deploy/run_peg.py check
python3 deploy/run_peg.py open --ack-safety
python3 deploy/run_peg.py grasp --ack-safety     # finger ends on the 32 mm line
```

Film every attempt from the side.

### 3.1 Shadow (policy runs, arm doesn't)

```bash
python3 deploy/run_peg.py shadow --export exports/peg_forge_seed2
```

Move the peg around **above** the hole by hand for 30 s.

- [ ] Beside the hole, the targets point toward it. Above the hole, they point down.

### 3.2 Air run (no contact possible)

```bash
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --air --trials 1 --ack-safety
```

- [ ] Smooth and slow, no jerks. It ends with a timeout or a success prediction, **not** a safety stop.
- [ ] Note whether the arm **stalls** (stops while the policy keeps commanding motion) or moves in jerks
      (stick-slip).

### 3.3 First real insertions (perfect hole estimate)

```bash
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 0 0 --trials 3 --ack-safety
```

| Trial | Outcome (`success` / `timeout` / `safety_stop`) | Peak force (N) | Notes (stalls, sliding on the rim, ...) |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

**After any safety stop, stop and report it.**

### 3.4 Evaluation (only if 3.3 had at least 1 success out of 3)

```bash
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 0 1   --trials 20 --ack-safety
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 1 2.5 --trials 20 --ack-safety
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 2.5 5 --trials 20 --ack-safety
python3 deploy/run_peg.py summary
```

Sim reference (seed 2): **97.9% / 97.4% / 90.5%**, peak force about 10.5 N. Check the grasp between
bands and re-grasp on the line if the peg slipped. A series stops on its own after a safety stop:
continue only if the cause is clear and fixed.

### 3.5 Finish and send back

```bash
python3 deploy/run_peg.py open --ack-safety
exit
zip -r session3_$(date +%Y%m%d).zip deploy/runs
```

Send the zip, the videos, the `summary` output and your notes.

---

## Quick reference

Commands, output files and troubleshooting are the same as in v1 (see its "Quick reference"). New in v2:

| Command | Moves? | What it does |
|---|---|---|
| `friction-breakaway` | a few mm, 6 directions, free space | Spring pull needed to start moving, per direction → `friction_breakaway_*.json` |
| `friction-sweep --axis A --speeds ...` | 3 cm and back per speed, free space | Friction vs speed (Coulomb + viscous fit), the fake force in the estimate, and the position-dependent error → `friction_sweep_*.json` |
| `force-drift --sweep_m 0` | holds still | v1's drift test with the sweep skipped |
| `friction-steps --steps_mm ...` | up to 8 mm steps, free space | Distance reached and time to 63% for policy-sized steps → `friction_steps_*.json` |
| `--stiffness_scale S` (on `friction-steps`) | | Runs the steps with all six gains × S; only 0.7–1.3 is accepted (the range the sim covered) |

All three stop with a safety stop if the tared force goes over 10 N or the hand gets more than 5 cm from
where the check started. A safety stop here means something touched the peg: check the clearance.
