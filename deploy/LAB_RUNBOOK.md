# Lab runbook: peg insertion on the real Franka

This is a step-by-step guide for running the robot tests. It assumes you have already run a FrankaPy program
on this robot (for example, Lab2). The background (why each test exists, what the numbers mean) is in
[`README.md`](README.md) and [`TEST_SUMMARY.md`](TEST_SUMMARY.md); you don't need them to run this.

**Status (2026-11-14):** session 1 has been run once (all of it except 1.8). Results and the open
questions are in [`TEST_SUMMARY.md`](TEST_SUMMARY.md) (test 6). The bridge fixes from that session are on
the `lukas` branch; use it until it is merged.

There are two sessions:

- **Session 1 (about 1.5 h): setup and measurements.** There are no insertions. Afterwards, send the
  results back. We use them to decide whether the policy needs retraining before session 2.
- **Session 2 (about 2 h): insertions.** Only run it after you hear back about session 1.

> **Safety, every time**
> - Two people: one runs the commands, one holds the **E-stop** the whole time the arm can move.
> - Every command that moves the arm needs `--ack-safety`. Most also ask you to type `YES`, **but
>   `rate-test` and `guide` start the moment you press Enter**: be ready before you do.
> - Keep hands out of the workspace while the arm moves, except where a step tells you to touch it
>   (hanging weights, guide mode).
> - If anything looks wrong (unexpected motion, grinding, the peg slipping), press the E-stop first and
>   ask questions later. Then write down what happened.
> - Do not change any limit or default in the code. If a step fails, stop and report it.

---

## 0. Before you go to the lab

**Bring:**
- [ ] Printed peg (Ø7.98 × 50 mm) and hole base (start with the **9.0 mm** bore). See
      [`../hardware/README.md`](../hardware/README.md).
- [ ] 2 clamps for the hole base.
- [ ] A **kitchen scale** with 1 g resolution, a tare/zero button, and at least 2 kg capacity.
- [ ] A string loop (about 30 cm) and 4–6 objects to hang, for example water bottles, a bag of
      screws or a tool. **Weigh each one on the kitchen scale** and write the grams on it with tape.
      Aim for roughly 100 g, 250 g, 500 g and 1000 g (combinations are fine).
- [ ] Phone to film the attempts (session 2).

**Get the code and the policy onto the lab computer** (the one that runs FrankaPy):

```bash
git clone git@github.com:kaushaljadhav512/forge_blocks.git    # or git pull if it is already there
cd forge_blocks
# The policy isn't in git (53 MB). Copy it from mrsd26c (or ask Kaushal for a zip):
mkdir -p exports
scp -r teamc@mrsd26c:~/forge/forge_blocks/exports/peg_forge_seed2 exports/
ls exports/peg_forge_seed2      # policy.pt  runner_config.json  agent.yaml  env.yaml
```

**On the 16662 lab PCs** (what worked on 2026-11-14):
- Run all `run_peg.py` commands **inside the course Docker container** (`bash run_docker.sh`, then
  `bash terminal_docker.sh`), from the repo under `src/devel_packages/<team>/`, which the container
  mounts (`/home/ros_ws/src/devel_packages/<team>/forge_blocks`). FrankaPy exists only in the container.
- Run `git pull` **on the host**, not in the container: the container has no SSH keys. If the clone's
  remote is HTTPS, `ssh -A` won't help; switch it to `git@github.com:...` (ask your teammates first).
- Files written from the container are owned by root. If git or `script` on the host says
  "permission denied", run `chown -R 1001:1001 <repo>` in the container (check your host uid with `id -u`).
  Inside the container git may need `git config --global --add safe.directory <repo>`.
- `script` (below) runs inside the container and stops if the container closes; start it again with
  `-a` to keep appending. Add `-f` so a teammate can follow the log live (`ssh … tail -F <log>`).

Session 1 needs only numpy and FrankaPy. The policy (and torch) are needed for the timing part of
`rate-test` and for all of session 2.

---

## Session 1: setup and measurements (no insertions)

Record the whole terminal session so nothing gets lost:

```bash
cd forge_blocks
mkdir -p deploy/runs
script -a deploy/runs/session1_terminal.log     # everything printed from now on is saved; `exit` to stop
```

Every command below is run from the `forge_blocks` folder. For each step, write the result or **PASS
or FAIL** in the boxes.

### 1.1 Versions (no robot)

```bash
python3 --version
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import torch; print('torch', torch.__version__)"
pip show frankapy | head -3
python3 -c "import sys; sys.path.insert(0,'deploy'); from policy_runner import ForgePolicyRunner as R; R.from_export('exports/peg_forge_seed2'); print('policy loads OK')"
```

- [ ] Write down all versions. If the last command fails (the torch version is too old for the file),
      **carry on with session 1 anyway**, skip `--export` in step 1.4, and report the error.

### 1.2 Robot up

1. Start the robot, unlock the joints in Desk and start FrankaPy (control PC and `franka-interface`)
   **exactly the way you did for Lab2**.
2. In Desk, check that the end-effector is set to the **Franka Hand** with its standard load. The peg
   weighs 19 g and can be ignored. A wrong load shows up as a constant force in the next step.

### 1.3 Read-only check

```bash
python3 deploy/run_peg.py home --ack-safety      # gripper pointing down, above the table
python3 deploy/run_peg.py check
```

`check` prints the joints, poses, raw force, gripper width and whether a Jacobian is available.

- [ ] `angle from 'gripper down, yaw 0'` is under about 5°.
- [ ] The raw force has each axis under **3 N** while nothing touches the arm. If it is larger, check
      the load setting in Desk.
- [ ] Write down `Jacobian from FrankaPy: yes/no`.
- [ ] The last line says `state source: topic` with a small age and differences of ~0. If it says
      anything else, the bridge isn't reading the 100 Hz state topic: stop and report it.

On the lab robot, the at-rest force changes with the pose and with which controller ran last (0.6 N
low, 3.7 N at home, 1.8 N at home after another command). The attempts tare after starting the
controller, so this is expected and is not a reason to stop.

### 1.4 Rate and delay

The arm holds still for 5 s, then moves **2 mm up**. Make sure the space above the gripper is clear.

```bash
python3 deploy/run_peg.py rate-test --export exports/peg_forge_seed2 --ack-safety
# if the policy didn't load in 1.1:  python3 deploy/run_peg.py rate-test --ack-safety
```

| Output | Want | Got |
|---|---|---|
| `send_hz` | ≥ 150 (minimum 120) | |
| `state_update_hz` | ≥ 100 | |
| `step_first_motion_ms` | < 30 | |
| `step_63pct_ms` | < 67 (one policy step) | |
| `policy_ms_max` | < 30 | |

If `send_hz` is under 120, try `--control_hz 100` once, write both results down, and carry on.

On 2026-11-14 this gave `send_hz` 200, `state_update_hz` 100, policy 8 ms (median), and **no motion**
for the 2 mm step (`null`). The step pulls with only 1.1 N, less than the joints' breakaway friction
(~2 N), so `null` is expected on this robot. It is not a broken stream.

### 1.5 Hole base and peg

1. Clamp the hole base flat on the table so that its bore is about **60 cm in front of the robot base
   and centred left-right** (anywhere within ±5 cm is fine). It must not move when pushed.
2. Pick a **free flat spot on the table at least 15 cm from the hole** for the force tests (steps
   1.6–1.8). The kitchen scale also goes there.
3. Grasp the peg:
   ```bash
   python3 deploy/run_peg.py open --ack-safety
   # hold the peg vertically between the fingers: finger ENDS on the 32 mm line (hardware/README.md)
   python3 deploy/run_peg.py grasp --ack-safety
   ```
- [ ] The printed width is about 0.008 m and the peg doesn't wiggle when you touch it lightly. The
      lab gripper's indented fingers read **6.3-6.4 mm** on the peg, just above `grasp`'s 6 mm limit.

### 1.6 Force sign (press test)

Move the peg **1–5 cm above the free flat spot** (not the hole):

```bash
python3 deploy/run_peg.py guide --seconds 20 --ack-safety     # move the arm by hand, then let go
python3 deploy/run_peg.py press-test --ack-safety
```

The arm lowers the peg slowly until it presses with about 6 N, holds for 1 s and lifts back. **Don't touch
the arm during this test**: a hand pressing on it gives the wrong sign.

- [ ] `force_sign = +1` or `-1` is printed and saved. Write it down.

### 1.7 Force estimate checks

Run these in this order. Each one saves its results in `deploy/runs/`.

**a) `force-curve`**: the same start as 1.6, 1–5 cm above the free spot. **On the lab robot this
test's table is not valid:** it detects contact when the arm lags 3 mm behind the target, and joint
friction alone causes that in mid-air. Skip it unless the contact detection has been changed.

```bash
python3 deploy/run_peg.py force-curve --ack-safety
```

The arm pushes into the table, increasing slowly up to 15 N, then lifts. It prints a table; nothing to
judge yet.

**b) `force-scale`**: put the **kitchen scale** on the free spot, press **zero** on it with nothing on
it, and move the peg **about 5 mm** above the middle of the scale (`guide`). Not 1–5 cm: friction makes
the contact detection fire early, so from higher up the peg may never reach the scale. Only the
`sensing` line is meaningful then; the `actuation` line also contains that early detection.

```bash
python3 deploy/run_peg.py force-scale --ack-safety
```

The arm presses on the scale and holds at 8 levels (1 to 10 N). At each level it asks
`scale reading (g)?`: type the number on the scale display (grams) and press Enter. Wait for the
display to settle first, about 2 s. The arm keeps holding while you type.

- [ ] Two lines are printed at the end. Write them down:
  - `actuation (commanded -> scale): responds from ~X N`
  - `sensing (scale -> estimate): responds from ~Y N` ← **the key number**

**c) `force-weights`**: lift the peg **at least 15 cm above anything** (`guide`). Tie the string loop
around the peg below the fingers, where it can't slip off. **Nothing should hang on it yet.**

```bash
python3 deploy/run_peg.py force-weights --ack-safety
```

It asks you to `hang the next mass and type its TOTAL grams`. Do this in order:

1. Type `0` (nothing hanging).
2. Hang about 100 g and type the total grams.
3. Add more to reach about 250 g and type the **total**; then about 500 g, then about 1000 g.
4. Remove everything and type `0`.
5. Press Enter on an empty line to finish.

Hang the weights gently, let them stop swinging, then type. The arm sags under load, which is normal:
the spring is 565 N/m, so about 9 mm per 500 g (more with friction). A water bottle works: weigh it,
pour some out away from the robot, weigh again, re-hang.

- [ ] `sign agrees with press-test: True`. **If it says False, stop and report it.**
- [ ] The final `0` row reads under 0.5 N.
- [ ] Write down the `sensing (true -> estimate)` line.

**d) `force-drift`**: remove the string, and keep the peg in free space with **at least 5 cm clear
below it**.

```bash
python3 deploy/run_peg.py force-drift --ack-safety
```

The arm holds still for 15 s, then moves 4 cm down and back. Don't touch anything during this test.

| Output | Want | Got (x / y / z) |
|---|---|---|
| hold: max \|f\| | < 1 N | |
| hold: drift (N/s) | < 0.05 | |
| sweep: max \|f\| | < 1 N | |

### 1.8 Register the hole

With the peg still grasped:

```bash
python3 deploy/run_peg.py register-hole --ack-safety
```

This repeats 5 times:
1. Guide mode turns on for 15 s. Push the peg **fully into the hole** by hand until it bottoms out,
   then hold it still until guide mode ends.
2. When asked, lift the peg out (guide mode turns on again for 5 s).

- [ ] The printed `spread` is **under 0.5 mm**. If not, check the clamps and run the command again.

### 1.9 Finish and send back

```bash
python3 deploy/run_peg.py open --ack-safety    # catch the peg
exit                                           # stops the terminal recording
zip -r session1_$(date +%Y%m%d).zip deploy/runs
```

Send the zip, the filled-in boxes from this page, and anything odd you noticed. **Don't start session
2** until we have looked at the numbers. The sensing dead band from 1.7b/c decides whether the policy
gets retrained first:

| Sensing dead band | Plan |
|---|---|
| under ~2 N | Session 2 with the current policy |
| ~2 N or more | Retrain with a simulated dead band (about 2 h), then session 2 with the new policy |

---

## Session 2: insertions (only after the go-ahead)

Same setup as session 1: same hole position (**don't move the clamps**; if the base moved, redo 1.8),
the same robot, and the same `deploy/runs/` folder. Use the policy folder we tell you to use (it may be
a new one).

```bash
script -a deploy/runs/session2_terminal.log
python3 deploy/run_peg.py check               # same as 1.3
python3 deploy/run_peg.py open --ack-safety
python3 deploy/run_peg.py grasp --ack-safety  # finger ends on the 32 mm line
```

Film every attempt from the side, close enough to see the peg and the hole.

### 2.1 Shadow (policy runs, arm doesn't)

```bash
python3 deploy/run_peg.py shadow --export exports/peg_forge_seed2
```

Guide mode for 30 s: move the peg around **above** the hole by hand. The policy's targets are printed.

- [ ] When the peg is beside the hole, the targets point toward the hole. When it is above the hole,
      they point down.

### 2.2 Air run (no contact possible)

The policy is told the hole is 10 cm higher than it is, so it moves but should never touch anything.

```bash
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --air --trials 1 --ack-safety
```

- [ ] The motion is smooth and slow, with no jerks. It ends with a timeout or a success prediction,
      **not** a safety stop.

### 2.3 First real insertions (perfect hole estimate)

```bash
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 0 0 --trials 3 --ack-safety
```

Before each trial it asks for confirmation. Make sure the peg is out of the hole and the E-stop is
ready. After each trial the arm pulls the peg 6 cm up.

- [ ] Write down the outcome of each trial (`success`, `timeout`, `safety_stop`) and its peak force.
      **After any safety stop, stop and report.**

### 2.4 Evaluation (only if 2.3 had at least 1 success out of 3)

Three error bands, 20 trials each (about 15–20 min per band). Check the peg grasp between bands.

```bash
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 0 1   --trials 20 --ack-safety
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 1 2.5 --trials 20 --ack-safety
python3 deploy/run_peg.py run --export exports/peg_forge_seed2 --band 2.5 5 --trials 20 --ack-safety
python3 deploy/run_peg.py summary
```

- A series stops on its own after a safety stop. Inspect the setup, note what happened, and continue
  with the same command only if the cause is clear and fixed.
- If the peg slips in the fingers (it sits visibly higher or lower), `open`, re-grasp on the line,
  and continue.

### 2.5 Finish and send back

```bash
python3 deploy/run_peg.py open --ack-safety
exit
zip -r session2_$(date +%Y%m%d).zip deploy/runs
```

Send the zip, the videos, the `summary` output and your notes.

---

## Quick reference

| Command | Moves? | What it does |
|---|---|---|
| `check` | no | State, poses, raw force, Jacobian, calibration so far |
| `home` | yes | FrankaPy's home pose |
| `guide --seconds N` | by hand | Guide mode for N s (**starts without a YES prompt**) |
| `open` / `grasp` | gripper | Open / close on the peg (8 mm, 40 N) |
| `rate-test` | 2 mm | Stream rate, delay, policy time (**starts without a YES prompt**) |
| `press-test` | contact | Force sign → `calibration.json` |
| `force-curve` | contact | Estimate vs commanded push, 0–15 N |
| `force-scale` | contact | Commanded vs scale vs estimate |
| `force-weights` | holds still | Hung masses vs estimate; second sign check |
| `force-drift` | 4 cm, free space | Drift, motion- and pose-dependent error, controller-start shift |
| `register-hole` | by hand | Saves the hole position → `hole.json` |
| `shadow` | by hand | Policy runs, nothing commanded |
| `run --air` / `run --band A B` | yes | Policy attempts |
| `summary` | no | Success rate etc. per band |

Everything is saved in `deploy/runs/`:
- `calibration.json`: force sign and the summary of each check;
- `hole.json`: the registered hole;
- `results.jsonl`: one line per attempt;
- `attempts/`: per-step logs;
- `force_*.json`: raw data from each force check;
- `events.jsonl`: a log of the commands that were run.

**Problems:**
- A command crashes in `FrankaPyArm`: FrankaPy versions differ. Copy the full error into your notes and
  stop.
- `no contact within the descent limit`: the surface is more than 6–8 cm below the peg. Start lower.
- `operator cancelled`: you typed something other than `YES`. Run the command again.
- `unexpected peg grasp width 0.045 m`: an old bridge read the width while the fingers were still
  closing (FrankaPy's `goto_gripper` doesn't wait for grasps). Fixed on `lukas`; pull it.
- `robot state is … ms old`: the state topic stopped. Check the three FrankaPy terminals.
- `run_peg.py check` prints only its own arguments: someone edited the lab copy. Check `git status`.
