import asyncio
import io
import re
import tarfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
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
_MODULE_COMPONENT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IMPORT_LINE_PATTERN = re.compile(r"^\s*(?:public\s+)?import\s+(?P<modules>[^\n]+)$", re.MULTILINE)
_WORKSPACE_MODULE_PATTERN = re.compile(r"\bFormalizerWorkspace(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_RESERVED_WORKSPACE_FILENAMES = frozenset(
    {
        "FormalizerCheck.lean",
        "FormalizerProblem.lean",
        "Main.lean",
    }
)
_WORKSPACE_PREFIX = PurePosixPath("FormalizerWorkspace")

_STANDALONE_COMMAND = "cat > /workspace/Main.lean && exec lake env lean /workspace/Main.lean"

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


def _source_archive(
    problem_code: str,
    solution_code: str,
    auxiliary_sources: Mapping[str, str],
) -> bytes:
    return _sources_archive(
        {
            "FormalizerProblem.lean": problem_code,
            **auxiliary_sources,
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


def _workspace_path(filename: str) -> PurePosixPath:
    if not filename or filename in _RESERVED_WORKSPACE_FILENAMES:
        raise InvalidLeanWorkspacePath(f"Cannot save auxiliary Lean file {filename!r}")

    path = PurePosixPath(filename)
    if (
        path.is_absolute()
        or str(path) != filename
        or ".." in path.parts
        or path.suffix != ".lean"
        or path.parts[0] == str(_WORKSPACE_PREFIX)
    ):
        raise InvalidLeanWorkspacePath(f"Invalid auxiliary Lean filename: {filename!r}")

    module_parts = (*path.parts[:-1], path.stem)
    if not module_parts or any(
        _MODULE_COMPONENT_PATTERN.fullmatch(part) is None for part in module_parts
    ):
        raise InvalidLeanWorkspacePath(f"Invalid auxiliary Lean filename: {filename!r}")

    return _WORKSPACE_PREFIX / path


def _workspace_module(path: PurePosixPath) -> str:
    return ".".join(path.with_suffix("").parts)


def _validate_auxiliary_archive_path(archive_path: str) -> None:
    path = PurePosixPath(archive_path)
    if str(path) != archive_path:
        raise InvalidLeanWorkspacePath(f"Invalid auxiliary Lean archive path: {archive_path!r}")

    try:
        relative_path = path.relative_to(_WORKSPACE_PREFIX)
    except ValueError as error:
        raise InvalidLeanWorkspacePath(
            f"Invalid auxiliary Lean archive path: {archive_path!r}"
        ) from error

    if _workspace_path(str(relative_path)) != path:
        raise InvalidLeanWorkspacePath(f"Invalid auxiliary Lean archive path: {archive_path!r}")


def _imported_workspace_modules(source: str) -> set[str]:
    return {
        imported
        for match in _IMPORT_LINE_PATTERN.finditer(source)
        for imported in _WORKSPACE_MODULE_PATTERN.findall(match.group("modules"))
    }


def _ordered_auxiliary_paths(
    auxiliary_sources: Mapping[str, str],
    root_source: str,
) -> list[str]:
    modules_by_path = {path: _workspace_module(PurePosixPath(path)) for path in auxiliary_sources}
    paths_by_module = {module: path for path, module in modules_by_path.items()}
    dependencies: dict[str, set[str]] = {}

    for path, source in auxiliary_sources.items():
        module = modules_by_path[path]
        dependencies[module] = _imported_workspace_modules(source) & paths_by_module.keys()

    ordered_modules: list[str] = []
    state: dict[str, int] = {}

    def visit(module: str) -> None:
        if state.get(module) == 2:
            return
        if state.get(module) == 1:
            return

        state[module] = 1
        for dependency in sorted(dependencies[module]):
            visit(dependency)
        state[module] = 2
        ordered_modules.append(module)

    root_modules = _imported_workspace_modules(root_source) & paths_by_module.keys()
    for module in sorted(root_modules):
        visit(module)

    return [paths_by_module[module] for module in ordered_modules]


def _problem_check_command(auxiliary_sources: Mapping[str, str], solution_code: str) -> str:
    compile_commands = [
        (
            "lean --root=/workspace -o /workspace/FormalizerProblem.olean "
            "/workspace/FormalizerProblem.lean"
        ),
        *(
            f"lean --root=/workspace -o /workspace/{path.removesuffix('.lean')}.olean "
            f"/workspace/{path}"
            for path in _ordered_auxiliary_paths(auxiliary_sources, solution_code)
        ),
        "lean --root=/workspace -o /workspace/Main.olean /workspace/Main.lean",
        "lean --root=/workspace /workspace/FormalizerCheck.lean",
    ]
    compiled = " &&\n  ".join(compile_commands)
    return f"""\
tar -xf - -C /workspace &&
cd /opt/mathlib &&
exec lake env sh -c '
  export LEAN_PATH="/workspace${{LEAN_PATH:+:$LEAN_PATH}}"
  {compiled}
'
"""


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


class InvalidLeanWorkspacePath(ValueError):
    """An auxiliary Lean filename is unsafe or not a valid module path."""


@dataclass(frozen=True, slots=True)
class SavedLeanFile:
    filename: str
    module: str
    revision: int


@dataclass(frozen=True, slots=True)
class DeletedLeanFile:
    filename: str
    deleted: bool
    revision: int


class LeanWorkspace:
    def __init__(self) -> None:
        self._sources: dict[str, str] = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def sources(self) -> dict[str, str]:
        return self._sources.copy()

    def save(self, code: str, filename: str) -> SavedLeanFile:
        path = _workspace_path(filename)
        archive_path = str(path)
        if self._sources.get(archive_path) != code:
            self._sources[archive_path] = code
            self._revision += 1

        return SavedLeanFile(
            filename=filename,
            module=_workspace_module(path),
            revision=self._revision,
        )

    def delete(self, filename: str) -> DeletedLeanFile:
        archive_path = str(_workspace_path(filename))
        deleted = archive_path in self._sources
        if deleted:
            del self._sources[archive_path]
            self._revision += 1

        return DeletedLeanFile(
            filename=filename,
            deleted=deleted,
            revision=self._revision,
        )


class LeanChecker(Protocol):
    async def check(
        self,
        code: str,
        *,
        auxiliary_sources: Mapping[str, str] | None = None,
    ) -> LeanResult: ...


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

    async def check(
        self,
        code: str,
        *,
        auxiliary_sources: Mapping[str, str] | None = None,
    ) -> LeanResult:
        resolved_auxiliary_sources = dict(auxiliary_sources or {})
        if self.problem_code is None:
            if resolved_auxiliary_sources:
                raise ValueError("Auxiliary Lean files require problem_code")
            command = _STANDALONE_COMMAND
            stdin = code.encode()
        else:
            for archive_path in resolved_auxiliary_sources:
                _validate_auxiliary_archive_path(archive_path)

            command = _problem_check_command(resolved_auxiliary_sources, code)
            stdin = _source_archive(self.problem_code, code, resolved_auxiliary_sources)

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
