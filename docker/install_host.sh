#!/usr/bin/env bash
# One-time host setup: installs Docker and NVIDIA's container toolkit. Needs sudo:
#
#   sudo bash docker/install_host.sh
#
# It stops before installing if apt would touch the NVIDIA driver or the kernel, and it never runs
# a general upgrade. Undo with docker/README.md, "Removing Docker".
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "Run with sudo." >&2; exit 1; }
[ -n "${SUDO_USER:-}" ] || { echo "Run with sudo from your own login, not as root." >&2; exit 1; }
. /etc/os-release
[ "$VERSION_CODENAME" = jammy ] || { echo "Written for Ubuntu 22.04, found $VERSION_CODENAME." >&2; exit 1; }
# A rerun after a failed install is fine; a config from anything else is left alone.
if [ -e /etc/docker/daemon.json ] && ! grep -q 192.168.127.1 /etc/docker/daemon.json; then
    echo "/etc/docker/daemon.json already exists; not overwriting." >&2
    exit 1
fi

driver_before=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

keyrings=/etc/apt/keyrings
install -d -m 0755 "$keyrings"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$keyrings/docker.asc"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$keyrings/nvidia-container-toolkit.asc"
chmod a+r "$keyrings/docker.asc" "$keyrings/nvidia-container-toolkit.asc"
echo "deb [arch=$(dpkg --print-architecture) signed-by=$keyrings/docker.asc] https://download.docker.com/linux/ubuntu jammy stable" \
    > /etc/apt/sources.list.d/docker.list
echo "deb [signed-by=$keyrings/nvidia-container-toolkit.asc] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update

packages=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit)
touched=$(apt-get install -s --no-install-recommends "${packages[@]}" | awk '/^(Inst|Remv) /{print $2}')
echo "apt would change:" $touched
if echo "$touched" | grep -E '^(nvidia-(driver|dkms|kernel|utils|compute|settings)|libnvidia-(gl|compute)|linux-(image|headers|modules))'; then
    echo "Stopping: apt wants to change the packages listed above." >&2
    exit 1
fi

# Must exist before Docker first starts. Its default 172.17-172.31 range overlaps this PC's ethernet
# and wifi and the VPN's client addresses, which could cut off SSH.
install -d -m 0755 /etc/docker
cat > /etc/docker/daemon.json <<'JSON'
{
  "bip": "192.168.127.1/24",
  "default-address-pools": [{"base": "192.168.128.0/18", "size": 24}],
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "3"}
}
JSON

apt-get install -y --no-install-recommends "${packages[@]}"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
usermod -aG docker "$SUDO_USER"

echo
echo "=== Checks"
driver_after=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
[ "$driver_before" = "$driver_after" ] && echo "driver unchanged: $driver_after" || echo "WARNING: driver went from $driver_before to $driver_after"
ip -br addr show docker0
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
echo "Done. Log out and back in as $SUDO_USER so the docker group applies."
