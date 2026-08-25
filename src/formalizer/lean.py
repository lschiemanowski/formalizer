import asyncio
import io
import re
import tarfile
from contextlib import suppress
from dataclasses import dataclass, replace
from time import monotonic
from typing import Protocol
from uuid import uuid4

from formalizer.settings import SandboxSettings

_VERIFICATION_CODE = """\
import FormalizerProblem
import Main

example : FormalizerProblem.Target :=
  FormalizerSubmission.solution

#print axioms FormalizerSubmission.solution
"""

_PROBLEM_VALIDATION_CODE = """\
import FormalizerProblem

#check FormalizerProblem.Target
"""

_ALLOWED_AXIOMS = frozenset(
    {
        "Classical.choice",
        "propext",
        "Quot.sound",
    }
)
_AXIOM_DEPENDENCIES_PATTERN = re.compile(
    r"'FormalizerSubmission\.solution' depends on axioms:\s*\[(?P<axioms>.*?)\]",
    re.DOTALL,
)
_NO_AXIOM_DEPENDENCIES_PATTERN = re.compile(
    r"'FormalizerSubmission\.solution' does not depend on any axioms"
)

_STANDALONE_COMMAND = "cat > /workspace/Main.lean && exec lake env lean /workspace/Main.lean"

_PROBLEM_CHECK_COMMAND = """\
tar -xf - -C /workspace &&
cd /opt/mathlib &&
exec lake env sh -c '
  export LEAN_PATH="/workspace${LEAN_PATH:+:$LEAN_PATH}"
  lean --root=/workspace -o /workspace/FormalizerProblem.olean /workspace/FormalizerProblem.lean &&
  lean --root=/workspace -o /workspace/Main.olean /workspace/Main.lean &&
  lean --root=/workspace /workspace/FormalizerCheck.lean
'
"""

_PROBLEM_VALIDATION_COMMAND = """\
tar -xf - -C /workspace &&
cd /opt/mathlib &&
exec lake env sh -c '
  export LEAN_PATH="/workspace${LEAN_PATH:+:$LEAN_PATH}"
  lean --root=/workspace -o /workspace/FormalizerProblem.olean /workspace/FormalizerProblem.lean &&
  lean --root=/workspace /workspace/FormalizerValidation.lean
'
"""


def _sources_archive(sources: dict[str, str]) -> bytes:
    buffer = io.BytesIO()

    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, source in sources.items():
            data = source.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))

    return buffer.getvalue()


def _source_archive(problem_code: str, solution_code: str) -> bytes:
    return _sources_archive(
        {
            "FormalizerProblem.lean": problem_code,
            "Main.lean": solution_code,
            "FormalizerCheck.lean": _VERIFICATION_CODE,
        }
    )


def _problem_validation_archive(problem_code: str) -> bytes:
    return _sources_archive(
        {
            "FormalizerProblem.lean": problem_code,
            "FormalizerValidation.lean": _PROBLEM_VALIDATION_CODE,
        }
    )


def _reported_axioms(stdout: str) -> frozenset[str]:
    match = _AXIOM_DEPENDENCIES_PATTERN.search(stdout)
    if match is not None:
        return frozenset(name.strip() for name in match.group("axioms").split(",") if name.strip())

    if _NO_AXIOM_DEPENDENCIES_PATTERN.search(stdout) is not None:
        return frozenset()

    raise LeanInfrastructureError("Lean completed without a recognizable axiom report")


@dataclass(frozen=True, slots=True)
class LeanResult:
    stdout: str
    stderr: str
    exit_code: int | None
    duration_s: float
    timed_out: bool = False
    verification_error: str | None = None

    @property
    def accepted(self) -> bool:
        return not self.timed_out and self.exit_code == 0 and self.verification_error is None


class LeanInfrastructureError(RuntimeError):
    """Lean could not be invoked reliably."""


class InvalidLeanProblem(ValueError):
    """The trusted problem source is not a valid formalization problem."""


class LeanChecker(Protocol):
    async def check(self, code: str) -> LeanResult: ...


class DockerLeanChecker:
    def __init__(
        self,
        settings: SandboxSettings,
        *,
        problem_code: str | None = None,
    ) -> None:
        self.settings = settings
        self.problem_code = problem_code

    async def validate_problem(self) -> None:
        if self.problem_code is None:
            raise ValueError("Cannot validate without problem_code")

        result = await self._run(
            _PROBLEM_VALIDATION_COMMAND,
            _problem_validation_archive(self.problem_code),
        )

        if result.timed_out:
            raise LeanInfrastructureError("Trusted problem validation timed out")

        if result.exit_code != 0:
            diagnostic = "\n".join(
                output.strip() for output in (result.stdout, result.stderr) if output.strip()
            )
            raise InvalidLeanProblem(diagnostic or f"Lean exited with status {result.exit_code}")

    async def check(self, code: str) -> LeanResult:
        if self.problem_code is None:
            command = _STANDALONE_COMMAND
            stdin = code.encode()
        else:
            command = _PROBLEM_CHECK_COMMAND
            stdin = _source_archive(self.problem_code, code)

        result = await self._run(command, stdin)

        if self.problem_code is not None and result.exit_code == 0:
            disallowed_axioms = sorted(_reported_axioms(result.stdout) - _ALLOWED_AXIOMS)
            if disallowed_axioms:
                return replace(
                    result,
                    verification_error=(
                        "Solution depends on disallowed axioms: " + ", ".join(disallowed_axioms)
                    ),
                )

        return result

    async def _run(self, command: str, stdin: bytes) -> LeanResult:
        container_name = f"formalizer-lean-{uuid4().hex}"
        docker_argv = (
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--pull",
            "never",
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--log-driver",
            "none",
            "--user",
            "10001:10001",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.settings.pids_limit),
            "--cpus",
            str(self.settings.cpus),
            "--memory",
            f"{self.settings.memory_mib}m",
            "--memory-swap",
            f"{self.settings.memory_mib}m",
            "--tmpfs",
            (
                "/workspace:rw,nosuid,nodev,noexec,"
                f"size={self.settings.workspace_mib}m,"
                "uid=10001,gid=10001,mode=0700"
            ),
            "--tmpfs",
            (f"/tmp:rw,nosuid,nodev,size={self.settings.tmp_mib}m,uid=10001,gid=10001,mode=1777"),
            "--workdir",
            "/opt/mathlib",
            self.settings.docker_image,
            "sh",
            "-c",
            command,
        )

        started_at = monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise LeanInfrastructureError("Could not start Docker") from error

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(stdin),
                timeout=self.settings.lean_timeout_s,
            )

        except TimeoutError:
            stdout_bytes, stderr_bytes = await self._terminate_and_remove(process, container_name)

            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            timeout_message = f"Lean check timed out after {self.settings.lean_timeout_s} seconds"
            stderr = f"{stderr.rstrip()}\n{timeout_message}" if stderr else timeout_message

            return LeanResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=None,
                duration_s=monotonic() - started_at,
                timed_out=True,
            )

        except asyncio.CancelledError:
            await asyncio.shield(
                self._terminate_and_remove(
                    process,
                    container_name,
                )
            )
            raise

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        duration_s = monotonic() - started_at

        exit_code = process.returncode
        if exit_code is None:
            raise LeanInfrastructureError("Docker exited without a return code")

        if exit_code in {125, 126, 127}:
            diagnostic = stderr or stdout or f"Docker exited with status {exit_code}"
            raise LeanInfrastructureError(diagnostic)

        return LeanResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_s=duration_s,
        )

    async def _container_exists(self, container_name: str) -> bool:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"name=^/{container_name}$",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            diagnostic = stderr.decode(errors="replace")
            raise LeanInfrastructureError(f"Could not verify container cleanup: {diagnostic}")

        return bool(stdout.strip())

    async def _remove_container(self, container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10,
            )
        except (OSError, TimeoutError) as error:
            raise LeanInfrastructureError(f"Could not remove container {container_name}") from error

        if await self._container_exists(container_name):
            diagnostic = stderr.decode(errors="replace")
            raise LeanInfrastructureError(
                f"Container {container_name} still exists after removal: {diagnostic}"
            )

    async def _terminate_and_remove(
        self,
        process: asyncio.subprocess.Process,
        container_name: str,
    ) -> tuple[bytes, bytes]:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()

        stdout, stderr = await process.communicate()
        await self._remove_container(container_name)

        return stdout, stderr
