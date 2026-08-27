import asyncio

import pytest

from formalizer.lean import (
    DockerLeanChecker,
    InvalidLeanProblem,
    LeanInfrastructureError,
    LeanWorkspace,
)
from formalizer.settings import SandboxSettings

TRUSTED_PROBLEM = """\
import Mathlib.Data.Nat.Basic

namespace FormalizerProblem

def Target : Prop := 1 + 1 = 2

end FormalizerProblem
"""

CLASSICAL_PROBLEM = """\
import Mathlib.Logic.Basic

namespace FormalizerProblem

def Target : Prop := ∀ p : Prop, p ∨ ¬p

end FormalizerProblem
"""


@pytest.mark.integration
async def test_valid_trusted_problem_passes_validation() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=TRUSTED_PROBLEM,
    )

    await checker.validate_problem()


@pytest.mark.integration
async def test_invalid_trusted_problem_reports_lean_diagnostics() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code="""\
namespace FormalizerProblem

def Target : Prop := 1 + 1

end FormalizerProblem
""",
    )

    with pytest.raises(InvalidLeanProblem) as exc_info:
        await checker.validate_problem()

    diagnostic = str(exc_info.value)
    assert "FormalizerProblem.lean" in diagnostic
    assert "failed to synthesize" in diagnostic


@pytest.mark.integration
async def test_trusted_problem_without_target_is_invalid() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code="""\
namespace FormalizerProblem

def Other : Prop := True

end FormalizerProblem
""",
    )

    with pytest.raises(InvalidLeanProblem) as exc_info:
        await checker.validate_problem()

    assert "FormalizerProblem.Target" in str(exc_info.value)


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
async def test_solution_can_import_saved_auxiliary_modules_in_dependency_order() -> None:
    workspace = LeanWorkspace()
    workspace.save(
        """\
import FormalizerProblem
import FormalizerWorkspace.Arithmetic.Base

namespace FormalizerWorkspace.Arithmetic

theorem helper : FormalizerProblem.Target := Base.helper

end FormalizerWorkspace.Arithmetic
""",
        "Arithmetic/Derived.lean",
    )
    workspace.save(
        """\
import FormalizerProblem

namespace FormalizerWorkspace.Arithmetic.Base

theorem helper : FormalizerProblem.Target := by rfl

end FormalizerWorkspace.Arithmetic.Base
""",
        "Arithmetic/Base.lean",
    )
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=TRUSTED_PROBLEM,
    )
    main = """\
import FormalizerWorkspace.Arithmetic.Derived

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := FormalizerWorkspace.Arithmetic.helper

end FormalizerSubmission
"""

    accepted = await checker.check(main, auxiliary_sources=workspace.sources)
    assert accepted.accepted, accepted

    workspace.delete("Arithmetic/Derived.lean")
    missing_import = await checker.check(main, auxiliary_sources=workspace.sources)

    assert not missing_import.accepted
    assert "FormalizerWorkspace.Arithmetic.Derived" in (
        missing_import.stdout + missing_import.stderr
    )


@pytest.mark.integration
async def test_solution_using_an_axiom_from_auxiliary_module_is_rejected() -> None:
    workspace = LeanWorkspace()
    workspace.save(
        """\
import FormalizerProblem

namespace FormalizerWorkspace

axiom cheat : FormalizerProblem.Target

end FormalizerWorkspace
""",
        "Cheat.lean",
    )
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=TRUSTED_PROBLEM,
    )

    result = await checker.check(
        """\
import FormalizerWorkspace.Cheat

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := FormalizerWorkspace.cheat

end FormalizerSubmission
""",
        auxiliary_sources=workspace.sources,
    )

    assert result.exit_code == 0
    assert not result.accepted
    assert result.verification_error is not None
    assert "FormalizerWorkspace.cheat" in result.verification_error


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
async def test_solution_using_sorry_is_rejected() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=TRUSTED_PROBLEM,
    )

    result = await checker.check(
        """\
import FormalizerProblem

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := by
  sorry

end FormalizerSubmission
"""
    )

    assert result.exit_code == 0
    assert not result.accepted
    assert result.verification_error is not None
    assert "sorryAx" in result.verification_error


@pytest.mark.integration
async def test_solution_using_submission_defined_axiom_is_rejected() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=TRUSTED_PROBLEM,
    )

    result = await checker.check(
        """\
import FormalizerProblem

namespace FormalizerSubmission

axiom cheat : FormalizerProblem.Target

theorem solution : FormalizerProblem.Target :=
  cheat

end FormalizerSubmission
"""
    )

    assert result.exit_code == 0
    assert not result.accepted
    assert result.verification_error is not None
    assert "FormalizerSubmission.cheat" in result.verification_error


@pytest.mark.integration
async def test_solution_using_standard_classical_axioms_is_accepted() -> None:
    checker = DockerLeanChecker(
        SandboxSettings(),
        problem_code=CLASSICAL_PROBLEM,
    )

    result = await checker.check(
        """\
import FormalizerProblem

namespace FormalizerSubmission

theorem solution : FormalizerProblem.Target := by
  intro p
  exact Classical.em p

end FormalizerSubmission
"""
    )

    assert result.accepted


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
