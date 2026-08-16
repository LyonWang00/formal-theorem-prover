FROM python:3.12.3-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG ELAN_VERSION=3.1.0
ARG UV_VERSION=0.8.24
ARG LEAN_VERSION=4.29.1
ARG PYPANTOGRAPH_COMMIT=ffa7f243824d2762825abddb1e9f6e939ede761f
ARG PANTOGRAPH_COMMIT=842c0fe6e76b0771cc7f7939604c7a0b90e17433
ARG GITHUB_PREFIX=https://ghfast.top/https://github.com

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ELAN_HOME=/opt/elan \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/home/prover/.cache/uv \
    LEAN_PROJECT_PATH=/opt/lean-project \
    LEAN_BACKEND=pantograph \
    PATH=/opt/venv/bin:/opt/elan/bin:/home/prover/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgmp-dev \
        passwd \
        pkg-config \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${GROUP_ID}" prover \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" \
        --create-home --shell /bin/bash prover \
    && mkdir -p /workspace /opt/elan /opt/lean-project /opt/venv \
        /home/prover/.cache/uv /home/prover/.cache/mathlib \
    && chown -R prover:prover /workspace /opt/elan /opt/lean-project \
        /opt/venv /home/prover

# The wrapper only rewrites github.com downloads; all revisions remain pinned.
COPY --chown=prover:prover scripts/wsl-bin/curl /usr/local/bin/curl
RUN chmod 0755 /usr/local/bin/curl

USER prover
WORKDIR /workspace

# Install the same Elan and uv versions as the verified WSL environment.
RUN curl -fsSL \
      "${GITHUB_PREFIX}/leanprover/elan/releases/download/v${ELAN_VERSION}/elan-x86_64-unknown-linux-gnu.tar.gz" \
      -o /tmp/elan.tar.gz \
    && tar -xzf /tmp/elan.tar.gz -C /tmp \
    && /tmp/elan-init -y --no-modify-path --default-toolchain none \
    && rm -f /tmp/elan.tar.gz /tmp/elan-init \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && elan --version \
    && uv --version

# Copy only Lean dependency declarations first, preserving this expensive layer
# when Python or project source changes.
COPY --chown=prover:prover lean_project/lean-toolchain \
    lean_project/lakefile.lean lean_project/lake-manifest.json \
    /opt/lean-project/
COPY --chown=prover:prover scripts/bootstrap_lake_deps.py /tmp/

RUN cd /opt/lean-project \
    && test "$(cat lean-toolchain)" = \
        "leanprover/lean4:v${LEAN_VERSION}" \
    && curl -fsSL \
        "${GITHUB_PREFIX}/leanprover/lean4/releases/download/v${LEAN_VERSION}/lean-${LEAN_VERSION}-linux.tar.zst" \
        -o /tmp/lean.tar.zst \
    && mkdir -p "/tmp/lean-${LEAN_VERSION}" \
        "/opt/elan/toolchains/leanprover--lean4---v${LEAN_VERSION}" \
    && tar --zstd -xf /tmp/lean.tar.zst -C "/tmp/lean-${LEAN_VERSION}" \
        --strip-components=1 \
    && cp -a "/tmp/lean-${LEAN_VERSION}/." \
        "/opt/elan/toolchains/leanprover--lean4---v${LEAN_VERSION}/" \
    && rm -rf /tmp/lean.tar.zst "/tmp/lean-${LEAN_VERSION}" \
    && lean --version \
    && lake --version \
    && cp lake-manifest.json /tmp/lake-manifest.locked.json \
    && git config --global "url.${GITHUB_PREFIX}/.insteadOf" \
        "https://github.com/" \
    && git config --global \
        "url.${GITHUB_PREFIX}/leanprover-community/mathlib4.git.insteadOf" \
        "https://mirror.sjtu.edu.cn/git/lean4-packages/mathlib4/" \
    && python /tmp/bootstrap_lake_deps.py /opt/lean-project \
    && cmp /tmp/lake-manifest.locked.json lake-manifest.json \
    && cache_ok=0 \
    && for attempt in 1 2 3; do \
         if lake exe cache get; then cache_ok=1; break; fi; \
         echo "mathlib cache attempt ${attempt} failed; retrying" >&2; \
         sleep 5; \
       done \
    && test "$cache_ok" = 1

# Recreate the exact local path dependency required by uv.lock from clean Git
# clones. No host .vendor build output enters the image.
RUN mkdir -p /workspace/.vendor \
    && git clone "${GITHUB_PREFIX}/stanford-centaur/PyPantograph.git" \
        /workspace/.vendor/PyPantograph \
    && git -C /workspace/.vendor/PyPantograph checkout --detach \
        "${PYPANTOGRAPH_COMMIT}" \
    && rm -rf /workspace/.vendor/PyPantograph/src \
    && git clone "${GITHUB_PREFIX}/leanprover/Pantograph.git" \
        /workspace/.vendor/PyPantograph/src \
    && git -C /workspace/.vendor/PyPantograph/src checkout --detach \
        "${PANTOGRAPH_COMMIT}" \
    && test "$(git -C /workspace/.vendor/PyPantograph \
        ls-tree HEAD src | awk '{print $3}')" = "${PANTOGRAPH_COMMIT}" \
    && grep -qx 'leanprover/lean4:v4.29.1' \
        /workspace/.vendor/PyPantograph/src/lean-toolchain

COPY --chown=prover:prover pyproject.toml uv.lock /workspace/

RUN uv sync --frozen --extra dev --no-install-project \
    && python -c \
      "from pantograph.server import get_version; assert get_version() == '0.3.15'"

COPY --chown=prover:prover lean_project/ /opt/lean-project/
RUN cd /opt/lean-project \
    && lake build \
    && lake env lean PantographImportSmoke.lean

COPY --chown=prover:prover lean_prover/ /workspace/lean_prover/
COPY --chown=prover:prover scripts/ /workspace/scripts/
COPY --chown=prover:prover tests/ /workspace/tests/
COPY --chown=prover:prover README.md /workspace/README.md
COPY --chown=prover:prover scripts/docker-entrypoint.sh /opt/docker-entrypoint.sh

RUN chmod 0755 /opt/docker-entrypoint.sh \
        /workspace/scripts/docker-entrypoint.sh \
        /workspace/scripts/check-toolchain.sh \
        /workspace/scripts/docker-smoke-test.sh \
    && python -m pytest -q

ENTRYPOINT ["/opt/docker-entrypoint.sh"]
CMD ["bash"]
