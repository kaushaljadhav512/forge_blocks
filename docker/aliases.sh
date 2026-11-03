# Shortcuts for docker/run.sh. Load with: source ~/forge/forge_blocks/docker/aliases.sh
FORGE_RUN=~/forge/forge_blocks/docker/run.sh

fshell() { "$FORGE_RUN"; }                 # shell inside the container
fbuild() { "$FORGE_RUN" build; }           # build the image
flab()   { "$FORGE_RUN" "\$ISAACLAB_PATH/isaaclab.sh -p $(printf '%q ' "$@")"; }
ftrain() { "$FORGE_RUN" "scripts/train_seeds.sh $(printf '%q ' "$@")"; }
