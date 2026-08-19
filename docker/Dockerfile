ARG DEBIAN_IMAGE
FROM ${DEBIAN_IMAGE}

ARG LEAN_TOOLCHAIN
ARG LEAN_REV
ARG MATHLIB_REV
ARG LOOGLE_REV

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.elan/bin:${PATH}"

# Minimal build/install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install elan, but don't choose a global Lean version.
# The project's lean-toolchain file determines Lean.
RUN curl https://elan.lean-lang.org/elan-init.sh -sSf \
    | sh -s -- -y --default-toolchain none

# ------------------------------------------------------------------
# Mathlib
# ------------------------------------------------------------------

RUN git clone https://github.com/leanprover-community/mathlib4.git /opt/mathlib \
    && cd /opt/mathlib \
    && git checkout "${MATHLIB_REV}" \
    && test "$(cat lean-toolchain)" = "${LEAN_TOOLCHAIN}" \
    && lake exe cache get

# Force Lean/toolchain installation, verify the precise Lean revision, and
# verify Mathlib works.
RUN cd /opt/mathlib \
    && lake env lean --version | tee /tmp/lean-version \
    && grep -F "commit ${LEAN_REV}," /tmp/lean-version \
    && printf 'import Mathlib\nexample : 1 + 1 = 2 := by norm_num\n' > /tmp/Test.lean \
    && lake env lean /tmp/Test.lean \
    && rm /tmp/Test.lean /tmp/lean-version

# ------------------------------------------------------------------
# Loogle
# ------------------------------------------------------------------

RUN git clone https://github.com/nomeata/loogle.git /opt/loogle \
    && cd /opt/loogle \
    && git checkout "${LOOGLE_REV}" \
    && cp /opt/mathlib/lean-toolchain ./lean-toolchain \
    && lake build

# Precompute Loogle index during image creation.
RUN cd /opt/mathlib \
    && lake env /opt/loogle/.lake/build/bin/loogle \
         --module Mathlib \
         --index-file /opt/loogle/mathlib.loogle-index \
         --index-mode write \
         --max-results 1 \
         'Nat.add_comm'

# ------------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------------

RUN useradd --create-home --uid 10001 sandbox \
    && mkdir /workspace \
    && chown sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

CMD ["sleep", "infinity"]
