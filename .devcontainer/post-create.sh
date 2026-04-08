#!/usr/bin/env bash
# Runs once when the dev container is first created. The firewall is NOT yet
# active at this point, so pip / conda / wget can talk to the open internet
# and pull whatever the project's pinned environment needs.
#
# After this script finishes, postStartCommand runs init-firewall.sh, which
# locks the network down to a small allowlist for runtime. So: do all your
# noisy installs here, not at runtime.

set -euo pipefail

cd /workspace

# -----------------------------------------------------------------------------
# Conda environment install. /opt/conda/envs is volume-mounted in
# devcontainer.json, so this only runs the slow path on first creation;
# subsequent rebuilds find the env already present and skip ahead.
#
# We gate on a SUCCESS_MARKER written at the very end of env creation, not on
# directory existence — otherwise a previous half-built env (e.g. pip install
# failed partway through) would be silently kept and the rebuild would skip.
# -----------------------------------------------------------------------------
SUCCESS_MARKER="/opt/conda/envs/robodiff/.post-create-done"
if [ -f "$SUCCESS_MARKER" ]; then
    echo "[post-create] robodiff env already built, skipping conda env create"
else
    if [ -d "/opt/conda/envs/robodiff" ]; then
        echo "[post-create] found partial robodiff env, removing before retry"
        conda env remove -n robodiff -y || rm -rf /opt/conda/envs/robodiff
    fi
    echo "[post-create] creating robodiff conda env (this can take 10+ minutes)"
    conda env create -f conda_environment.yaml
    touch "$SUCCESS_MARKER"
fi

# Make `conda activate robodiff` work in interactive shells.
conda init bash zsh >/dev/null
{
    echo ""
    echo "# Auto-activate the project's conda env."
    echo "conda activate robodiff"
} >> "$HOME/.bashrc"
{
    echo ""
    echo "# Auto-activate the project's conda env."
    echo "conda activate robodiff"
} >> "$HOME/.zshrc"

echo "[post-create] done"
