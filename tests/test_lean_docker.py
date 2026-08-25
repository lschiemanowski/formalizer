import asyncio

import pytest

from formalizer.lean import DockerLeanChecker, LeanInfrastructureError
from formalizer.settings import SandboxSettings

TRUSTED_PROBLEM = """\
import Mathlib.Data.Nat.Basic

namespace FormalizerProblem

def Target : Prop := 1 + 1 = 2

end FormalizerProblem
"""


@pytest.mark.integration
async def test_solution_can_add_imports_and_open_scopes() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=TRUSTED_PROBLEM,
    )

    result = await checker.check(
        """\
import FormalizerProblem
import Mathlib.Tactic

open scoped BigOperators

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := by
  norm_num [FormalizerProblem.Target]

end FormalizerSubmission
"""
    )

    assert result.accepted


@pytest.mark.integration
async def test_unrelated_compiling_solution_is_rejected() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=TRUSTED_PROBLEM,
    )

    result = await checker.check(
        """\
import FormalizerProblem

namespace FormalizerSubmission

example : True := by
  trivial

end FormalizerSubmission
"""
    )

    assert not result.accepted
    assert result.exit_code != 0
    assert "FormalizerSubmission.solution" in result.stdout + result.stderr


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


async def formalizer_lean_container_ids() -> set[str]:
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

    return set(stdout.decode().splitlines())


async def wait_for_new_formalizer_container(
    existing: set[str],
) -> str:
    async with asyncio.timeout(10):
        while True:
            new_containers = (await formalizer_lean_container_ids()) - existing

            if new_containers:
                return next(iter(new_containers))

            await asyncio.sleep(0.05)


@pytest.mark.integration
async def test_cancelled_lean_check_removes_its_container() -> None:
    containers_before = await formalizer_lean_container_ids()
    checker = DockerLeanChecker(SandboxSettings())

    task = asyncio.create_task(
        checker.check(
            """
            run_cmd IO.sleep 60_000
            """
        )
    )

    await wait_for_new_formalizer_container(containers_before)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    containers_after = await formalizer_lean_container_ids()
    assert containers_after <= containers_before


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


@pytest.mark.integration
async def test_missing_image_is_an_infrastructure_error() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(
            docker_image="formalizer-image-that-does-not-exist",
        )
    )

    with pytest.raises(LeanInfrastructureError):
        await checker.check("example : True := by trivial")


@pytest.mark.integration
async def test_candidate_cannot_modify_mathlib() -> None:
    checker = DockerLeanChecker(SandboxSettings())

    result = await checker.check(
        """
        import Mathlib

        run_cmd
          IO.FS.writeFile
            "/opt/mathlib/formalizer-write-probe"
            "modified"
        """
    )

    assert not result.accepted
    assert result.exit_code != 0
    assert "/opt/mathlib/formalizer-write-probe" in (result.stdout + result.stderr)
