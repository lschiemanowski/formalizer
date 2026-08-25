from dataclasses import dataclass
from pathlib import Path

import pytest

import formalizer.cli as cli_module
from formalizer.agent import VerifiedSubmission
from formalizer.settings import Settings

PROBLEM = "Prove that 1 + 1 = 2."
VALID_CODE = """import Mathlib

example : 1 + 1 = 2 := by norm_num
"""


@dataclass(frozen=True, slots=True)
class FakeRunResult:
    output: VerifiedSubmission


@pytest.mark.parametrize(
    "argv",
    [
        [],
        [PROBLEM],
        ["--model", "test:model"],
    ],
)
def test_cli_requires_model_and_problem(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(argv)

    assert exc_info.value.code == 2


def test_cli_translates_arguments_into_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "formalizer-runs"
    calls: list[tuple[str, Settings]] = []

    async def record_formalize(problem: str, settings: Settings) -> FakeRunResult:
        calls.append((problem, settings))
        return FakeRunResult(output=VerifiedSubmission(code=VALID_CODE))

    monkeypatch.setattr(cli_module, "formalize", record_formalize, raising=False)

    exit_code = cli_module.main(
        [
            "--model",
            "test:model",
            "--runs-dir",
            str(runs_dir),
            PROBLEM,
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    problem, settings = calls[0]
    assert problem == PROBLEM
    assert settings.model_name == "test:model"
    assert settings.run.runs_dir == runs_dir


@pytest.mark.parametrize(
    ("code", "expected_stdout"),
    [
        (VALID_CODE, VALID_CODE),
        (VALID_CODE.rstrip("\n"), VALID_CODE),
    ],
)
def test_cli_prints_verified_lean_code_with_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    code: str,
    expected_stdout: str,
) -> None:
    async def succeed(problem: str, settings: Settings) -> FakeRunResult:
        return FakeRunResult(output=VerifiedSubmission(code=code))

    monkeypatch.setattr(cli_module, "formalize", succeed, raising=False)

    exit_code = cli_module.main(["--model", "test:model", PROBLEM])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == expected_stdout
    assert captured.err == ""


def test_cli_reports_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(problem: str, settings: Settings) -> FakeRunResult:
        raise RuntimeError("model request failed")

    monkeypatch.setattr(cli_module, "formalize", fail, raising=False)

    exit_code = cli_module.main(["--model", "test:model", PROBLEM])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "RuntimeError" in captured.err
    assert "model request failed" in captured.err


def test_cli_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def interrupt(problem: str, settings: Settings) -> FakeRunResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "formalize", interrupt, raising=False)

    exit_code = cli_module.main(["--model", "test:model", PROBLEM])

    captured = capsys.readouterr()
    assert exit_code == 130
    assert captured.out == ""
    assert "interrupted" in captured.err.lower()
