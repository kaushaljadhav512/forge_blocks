# TODO

Goal: a running reproduction of the peg task on the training PC (`mrsd26c`, RTX 4080).

Isaac Lab v3 isn't needed. This repo runs on v2.3.2 in Docker; v3 at `~/IsaacLab` belongs to teammates
and only needs to stay untouched.

## Done

- [x] SSH over ethernet: `mrsd26c.lan.local.cmu.edu`
- [x] Docker and NVIDIA's container toolkit installed; a test container sees the GPU
- [x] `~/forge/IsaacLab` at v2.3.2, this repo copied to `~/forge/forge_blocks`
- [x] Image `forge-isaaclab` built, with the install fixes in `docker/Dockerfile.forge`
- [x] NVIDIA Isaac Sim license accepted

Isaac Sim has not been started yet, so expect some fixes in the steps below.

## Next: first run

Run these on the PC from `~/forge/forge_blocks`. Stop and fix anything that fails before moving on.

1. [ ] **Get a shell that has Docker access.** Tmux sessions started before Docker was installed don't
   have it, and teammates share that server, so don't kill it. Use a fresh login and a separate server:

   ```bash
   tmux -L forge new -s forge
   cd ~/forge/forge_blocks
   ```

2. [ ] **Check the GPU is free,** and tell teammates before starting long runs:

   ```bash
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ```

3. [ ] **Upstream test, 2 epochs.** Shows whether Isaac Sim works in the container, separately from this
   repo's code. Also downloads the assets.

   ```bash
   docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p $ISAACLAB_PATH/scripts/reinforcement_learning/rl_games/train.py --task Isaac-Forge-PegInsert-Direct-v0 --headless agent.params.config.name=upstream_smoke agent.params.config.full_experiment_name=seed0 agent.params.config.max_epochs=2'
   ```

4. [ ] **Preflight.** Must print `ALL PASSED`. Also prints the real time per epoch.

   ```bash
   docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p scripts/preflight.py --headless'
   ```

5. [ ] **End-to-end smoke test on the peg, 2 epochs.** `sim_bridge_test.py` must report PASS.

   ```bash
   docker/run.sh 'scripts/train_seeds.sh Isaac-ForgeBlocks-Peg8mm-Direct-v0 peg_smoke "0" agent.params.config.max_epochs=2'
   docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p scripts/eval_sweep.py --task Isaac-ForgeBlocks-Peg8mm-Direct-v0 --headless --paper_bins --episodes 128 --checkpoint logs/rl_games/peg_smoke/seed0/nn/last_*.pth --out results/peg_smoke.csv'
   docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p scripts/export_policy.py logs/rl_games/peg_smoke/seed0/nn/last_*.pth --out exports/peg_smoke'
   docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p scripts/sim_bridge_test.py --task Isaac-ForgeBlocks-Peg8mm-Direct-v0 --export exports/peg_smoke --headless'
   ```

6. [ ] **Full training, seed 0.** About 1.5-3 hours (estimate; use the time from preflight).

   ```bash
   docker/run.sh 'scripts/train_seeds.sh Isaac-ForgeBlocks-Peg8mm-Direct-v0 peg_forge "0"'
   ```

7. [ ] **Evaluate.** Success should fall as error grows, especially above 2.5 mm, with forces near 7.5 N.

   ```bash
   docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p scripts/eval_sweep.py --task Isaac-ForgeBlocks-Peg8mm-Direct-v0 --headless --paper_bins --checkpoint logs/rl_games/peg_forge/seed0/nn/last_peg_forge_ep_200_rew_*.pth --out results/peg_forge_seed0.csv'
   ```

8. [ ] **Seeds 1 and 2,** if you want them (another 3-6 hours): rerun step 6 with `"1 2"`, then step 7.

9. [ ] **Confirm v3 is unchanged.** No diff output means it matches. Delete the baseline afterwards.

   ```bash
   diff <(tail -n +2 ~/forge/v3_baseline.txt) <({ git -C ~/IsaacLab rev-parse HEAD; git -C ~/IsaacLab status --short; VIRTUAL_ENV=~/IsaacLab/.venv ~/.local/bin/uv pip freeze; } 2>/dev/null)
   rm ~/forge/v3_baseline.txt
   ```

## After the peg runs

- [ ] **Fix the slot asset before training any block task.** `slot_*.usda` has the articulation root and
  rigid body on one prim, so `fix_root_link=True` moves the root to the parent `env_0` prim. This will
  probably crash the slot's setup (from reading the code, not tested). Generate a root prim with a child
  rigid body and a fixed joint in `scripts/make_assets.py`, and drop `fix_root_link` from
  `forge_blocks/tasks/block_env_cfg.py`. Test with
  `scripts/preflight.py --task Isaac-ForgeBlocks-Block2mm-Direct-v0`; the default preflight only covers
  the peg.
- [ ] Measure the real block and slots, update `forge_blocks/geometry.py`, and rerun `make_assets.py`.
- [ ] Train and evaluate the block tasks (README, "Workflow").
- [ ] Make `scripts/train_seeds.sh` require `ISAACLAB_PATH`. Outside Docker, it falls back to `~/IsaacLab`,
  which is v3 on the training PC.
- [ ] Add a read-only deploy key on the PC, so it can pull instead of relying on a copied repo.
