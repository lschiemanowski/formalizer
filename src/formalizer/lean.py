import asyncio
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from typing import Protocol
from uuid import uuid4

from formalizer.settings import SandboxSettings


@dataclass(frozen=True, slots=True)
class LeanResult:
    stdout: str
    stderr: str
    exit_code: int | None
    duration_s: float
    timed_out: bool = False
    output_truncated: bool = False

    @property
    def accepted(self) -> bool:
        return not self.timed_out and self.exit_code == 0


class LeanInfrastructureError(RuntimeError):
    """Lean could not be invoked reliably."""


class LeanChecker(Protocol):
    async def check(self, code: str) -> LeanResult: ...


class DockerLeanChecker:
    def __init__(self, settings: SandboxSettings) -> None:
        self.settings = settings

    async def check(self, code: str) -> LeanResult:
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
            ("cat > /workspace/Main.lean && exec lake env lean /workspace/Main.lean"),
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
                process.communicate(code.encode()),
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
