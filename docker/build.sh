#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
version_file="${script_dir}/version.env"

# shellcheck source=version.env
source "${version_file}"

required_variables=(
    DEBIAN_IMAGE
    LEAN_TOOLCHAIN
    LEAN_REV
    MATHLIB_REV
    LOOGLE_REV
)

for variable_name in "${required_variables[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        printf 'Missing required pin %s in %s\n' "${variable_name}" "${version_file}" >&2
        exit 1
    fi
done

if [[ ! "${DEBIAN_IMAGE}" =~ ^debian:[^@]+@sha256:[0-9a-f]{64}$ ]]; then
    printf 'DEBIAN_IMAGE must contain an immutable sha256 digest: %s\n' "${DEBIAN_IMAGE}" >&2
    exit 1
fi

for revision_variable in LEAN_REV MATHLIB_REV LOOGLE_REV; do
    if [[ ! "${!revision_variable}" =~ ^[0-9a-f]{40}$ ]]; then
        printf '%s must be a full 40-character Git revision\n' "${revision_variable}" >&2
        exit 1
    fi
done

docker build \
    --build-arg "DEBIAN_IMAGE=${DEBIAN_IMAGE}" \
    --build-arg "LEAN_TOOLCHAIN=${LEAN_TOOLCHAIN}" \
    --build-arg "LEAN_REV=${LEAN_REV}" \
    --build-arg "MATHLIB_REV=${MATHLIB_REV}" \
    --build-arg "LOOGLE_REV=${LOOGLE_REV}" \
    --tag "${IMAGE_TAG:-formalizer:latest}" \
    "$@" \
    "${script_dir}"
