import pytest

from formalizer.lean import (
    DockerLeanChecker,
    InvalidLeanWorkspacePath,
    LeanResult,
    LeanWorkspace,
)
from formalizer.settings import SandboxSettings


def test_zero_exit_is_accepted() -> None:
    result = LeanResult(
        stdout="",
        stderr="",
        exit_code=0,
        duration_s=0.1,
    )

    assert result.accepted


def test_nonzero_exit_is_rejected() -> None:
    result = LeanResult(
        stdout="",
        stderr="proof failed",
        exit_code=1,
        duration_s=0.1,
    )

    assert not result.accepted


def test_timeout_is_not_accepted() -> None:
    result = LeanResult(
        stdout="",
        stderr="",
        exit_code=None,
        duration_s=30,
        timed_out=True,
    )

    assert not result.accepted


def test_workspace_saves_overwrites_and_deletes_auxiliary_modules() -> None:
    workspace = LeanWorkspace()

    first = workspace.save("theorem value : True := trivial\n", "Logic/Helper.lean")

    assert first.filename == "Logic/Helper.lean"
    assert first.module == "FormalizerWorkspace.Logic.Helper"
    assert first.revision == 1
    assert workspace.sources == {
        "FormalizerWorkspace/Logic/Helper.lean": "theorem value : True := trivial\n"
    }

    overwritten = workspace.save("theorem value : True := by trivial\n", "Logic/Helper.lean")

    assert overwritten.revision == 2
    assert workspace.sources == {
        "FormalizerWorkspace/Logic/Helper.lean": "theorem value : True := by trivial\n"
    }

    deleted = workspace.delete("Logic/Helper.lean")
    missing = workspace.delete("Logic/Helper.lean")

    assert deleted.deleted
    assert deleted.revision == 3
    assert not missing.deleted
    assert missing.revision == 3
    assert workspace.sources == {}


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "Main.lean",
        "FormalizerProblem.lean",
        "FormalizerCheck.lean",
        "/Helper.lean",
        "../Helper.lean",
        "Logic/../Helper.lean",
        "./Helper.lean",
        "FormalizerWorkspace/Helper.lean",
        "Helper.txt",
        "bad-name.lean",
        "Logic//Helper.lean",
    ],
)
def test_workspace_rejects_unsafe_or_invalid_module_names(filename: str) -> None:
    workspace = LeanWorkspace()

    with pytest.raises(InvalidLeanWorkspacePath):
        workspace.save("theorem value : True := by trivial\n", filename)

    assert workspace.sources == {}
    assert workspace.revision == 0


def test_workspaces_do_not_share_files() -> None:
    first = LeanWorkspace()
    second = LeanWorkspace()

    first.save("theorem value : True := by trivial\n", "Helper.lean")

    assert second.sources == {}
    assert second.revision == 0


@pytest.mark.parametrize(
    "archive_path",
    [
        "Helper.lean",
        "../FormalizerWorkspace/Helper.lean",
        "/FormalizerWorkspace/Helper.lean",
        "FormalizerWorkspace//Helper.lean",
    ],
)
async def test_checker_rejects_invalid_auxiliary_archive_paths_before_docker(
    archive_path: str,
) -> None:
    checker = DockerLeanChecker(SandboxSettings(), problem_code="def Target : Prop := True")

    with pytest.raises(InvalidLeanWorkspacePath):
        await checker.check(
            "theorem solution : True := by trivial",
            auxiliary_sources={archive_path: "theorem helper : True := by trivial"},
        )
