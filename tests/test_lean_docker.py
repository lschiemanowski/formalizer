import asyncio

import pytest

from formalizer.lean import DockerLeanChecker
from formalizer.settings import SandboxSettings


@pytest.mark.integration
async def test_valid_lean_file_is_accepted() -> None:
    checker = DockerLeanChecker(SandboxSettings())

    result = await checker.check(
        """
        import Mathlib

        example : 1 + 1 = 2 := by
          norm_num
        """
    )

    assert result.accepted


@pytest.mark.integration
async def test_invalid_lean_file_is_rejected_with_diagnostics() -> None:
    checker = DockerLeanChecker(SandboxSettings())

    result = await checker.check(
        """
        import Mathlib

        example : False := by
          trivial
        """
    )

    assert not result.accepted
    assert result.exit_code != 0
    assert "error:" in result.stdout + result.stderr


async def formalizer_lean_container_ids() -> list[str]:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "--all",
        "--quiet",
        "--filter",
        "name=formalizer-lean-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace"))

    return stdout.decode().splitlines()


@pytest.mark.integration
async def test_timed_out_lean_check_removes_its_container() -> None:
    containers_before = set(await formalizer_lean_container_ids())

    checker = DockerLeanChecker(
        SandboxSettings(lean_timeout_s=1),
    )

    result = await checker.check(
        """
        run_cmd IO.sleep 60_000
        """
    )

    assert result.timed_out
    assert result.exit_code is None
    assert not result.accepted
    assert "timed out" in result.stderr.lower()

    containers_after = set(await formalizer_lean_container_ids())
    assert containers_after <= containers_before
