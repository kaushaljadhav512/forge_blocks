# Running in Docker

Keeps Isaac Lab v2.3.2 and Isaac Sim 5.1 apart from anything else on a shared PC. The GPU and NVIDIA
driver are still shared, so check who is using the GPU before training.

```bash
# Once per machine, with sudo. Log out and back in afterwards.
sudo bash docker/install_host.sh

# Layout: this repo and Isaac Lab sit side by side.
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab

docker/run.sh build
docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p scripts/preflight.py --headless'
```

`docker/run.sh build` builds Isaac Lab's own image, then `Dockerfile.forge` on top. That layer finishes
the `isaaclab` install, which upstream's build can fail without reporting it. `docker/aliases.sh` has
shorter commands: `source` it, then use `flab scripts/preflight.py --headless`.

Run long jobs inside tmux. Starting a container accepts NVIDIA's Isaac Sim license, because Isaac
Lab's `.env.base` sets `ACCEPT_EULA=Y`.

Checkpoints and results stay in this repo's `logs/`, `results/` and `exports/` on the host.

## Removing Docker

This also deletes all images and Docker volumes on the machine, so check that nobody else uses them.

```bash
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit
sudo rm -rf /etc/docker /var/lib/docker /var/lib/containerd
sudo rm /etc/apt/sources.list.d/docker.list /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo rm /etc/apt/keyrings/docker.asc /etc/apt/keyrings/nvidia-container-toolkit.asc
```
