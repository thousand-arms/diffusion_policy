# NVIDIA's PyTorch container ships Python, PyTorch, CUDA, cuDNN, NCCL,
# FlashAttention, and a tuned conda installation. We layer Node.js +
# Claude Code + iptables sandboxing on top so the same image can both
# run training jobs *and* host an interactive Claude Code session.
#
# Tag scheme is YY.MM-py3 — bump to a newer release when convenient.
# Release notes: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/
FROM nvcr.io/nvidia/pytorch:24.10-py3

ARG TZ
ENV TZ="$TZ"
ARG CLAUDE_CODE_VERSION=latest
ARG GIT_DELTA_VERSION=0.18.2
ARG ZSH_IN_DOCKER_VERSION=1.2.0
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# -----------------------------------------------------------------------------
# System packages: dev tools, Node.js (for Claude Code), and the iptables /
# ipset / dig / jq / aggregate stack that init-firewall.sh depends on.
# libgl1 and libglib2 are required by py-opencv (cv2.imshow / VideoCapture).
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg2 \
      less git procps sudo fzf zsh man-db unzip \
      iptables ipset iproute2 dnsutils aggregate jq \
      nano vim wget \
      libgl1 libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs gh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Non-root user matching the host UID/GID so files written into mounted
# volumes don't end up root-owned on the host.
# -----------------------------------------------------------------------------
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/zsh $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Persist bash history across rebuilds via the volume mount in devcontainer.json.
RUN mkdir -p /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME:$USERNAME /commandhistory \
    && SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo "$SNIPPET" >> /home/$USERNAME/.bashrc

# Workspace + Claude config dirs (Claude config dir is also volume-mounted).
# Also pre-create ~/.cache owned by the user — the huggingface cache bind mount
# under it would otherwise cause Docker to create the parent as root, and then
# VS Code server can't mkdir its own subdirs (e.g. .cache/Microsoft).
RUN mkdir -p /workspace /home/$USERNAME/.claude /home/$USERNAME/.cache \
    && chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME/.claude /home/$USERNAME/.cache

# -----------------------------------------------------------------------------
# Miniforge (conda). The NVIDIA PyTorch base image uses system python + pip and
# does NOT ship conda, but the project's pinned conda_environment.yaml wants
# python 3.9 / torch 1.12 (a 2022 research snapshot), which is incompatible
# with the base image's python 3.10 / torch 2.5 anyway. Easiest path: install
# our own conda at /opt/conda and let post-create.sh build the robodiff env
# inside it. /opt/conda/envs is volume-mounted so the slow env create only
# runs on first creation.
# -----------------------------------------------------------------------------
RUN ARCH=$(uname -m) && \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean -afy && \
    chown -R $USER_UID:$USER_GID /opt/conda
ENV PATH=/opt/conda/bin:$PATH

ENV DEVCONTAINER=true
WORKDIR /workspace

# git-delta gives Claude pretty diffs in the terminal.
RUN ARCH=$(dpkg --print-architecture) && \
    wget -q "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# -----------------------------------------------------------------------------
# Switch to the unprivileged user. Claude Code installs into the user's
# global npm prefix; the firewall script gets passwordless sudo via the
# sudoers entry above.
# -----------------------------------------------------------------------------
USER $USERNAME

ENV NPM_CONFIG_PREFIX=/home/$USERNAME/.npm-global
ENV PATH=$PATH:/home/$USERNAME/.npm-global/bin
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Default powerline10k zsh setup so terminal feels familiar.
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
      -p git \
      -p fzf \
      -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
      -a "source /usr/share/doc/fzf/examples/completion.zsh" \
      -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
      -x

# Install Claude Code globally for this user.
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# -----------------------------------------------------------------------------
# Firewall script: needs root to run, so install it system-wide and grant
# the unprivileged user passwordless sudo specifically for it. (The blanket
# NOPASSWD sudoers above already covers this; the dedicated entry below is
# kept so a future hardening pass can drop the blanket rule and keep just
# this targeted exception.)
# -----------------------------------------------------------------------------
USER root
COPY init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh \
    && echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" \
       > /etc/sudoers.d/${USERNAME}-firewall \
    && chmod 0440 /etc/sudoers.d/${USERNAME}-firewall

USER $USERNAME
