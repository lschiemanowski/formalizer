import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from types import TracebackType
from typing import Protocol, Self
from uuid import uuid4

from formalizer.settings import SandboxSettings

_DOCKER_CLEANUP_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class SearchHit:
    name: str
    type: str
    module: str | None
    doc: str | None

    @classmethod
    def from_loogle_payload(cls, payload: object) -> Self:
        try:
            if not isinstance(payload, dict):
                raise TypeError("hit is not a JSON object")

            name = payload["name"]
            type_ = payload["type"]
            module = payload["module"]
            doc = payload["doc"]

            if not isinstance(name, str):
                raise TypeError("hit name is not a string")
            if not isinstance(type_, str):
                raise TypeError("hit type is not a string")
            if module is not None and not isinstance(module, str):
                raise TypeError("hit module is neither a string nor null")
            if doc is not None and not isinstance(doc, str):
                raise TypeError("hit doc is neither a string nor null")

            return cls(
                name=name,
                type=type_,
                module=module,
                doc=doc,
            )
        except (KeyError, TypeError) as error:
            raise SearchInfrastructureError("Loogle returned an invalid search hit") from error


@dataclass
class SearchResult:
    query: str
    hits: tuple[SearchHit, ...]
    total_count: int
    header: str
    error: str | None = None
    suggestions: tuple[str, ...] = ()
    duration_s: float = 0.0

    @classmethod
    def from_loogle_json(
        cls,
        *,
        query: str,
        response_line: bytes,
        duration_s: float,
    ) -> Self:
        try:
            payload = json.loads(response_line)
            if not isinstance(payload, dict):
                raise TypeError("response is not a JSON object")

            raw_suggestions = payload.get("suggestions", [])
            if not isinstance(raw_suggestions, list) or not all(
                isinstance(suggestion, str) for suggestion in raw_suggestions
            ):
                raise TypeError("suggestions is not a list of strings")
            suggestions = tuple(raw_suggestions)

            if "error" in payload:
                error = payload["error"]
                if not isinstance(error, str):
                    raise TypeError("error is not a string")

                return cls(
                    query=query,
                    hits=(),
                    total_count=0,
                    header="",
                    error=error,
                    suggestions=suggestions,
                    duration_s=duration_s,
                )

            header = payload["header"]
            total_count = payload["count"]
            raw_hits = payload["hits"]

            if not isinstance(header, str):
                raise TypeError("header is not a string")
            if not isinstance(total_count, int) or isinstance(total_count, bool):
                raise TypeError("count is not an integer")
            if total_count < 0:
                raise TypeError("count is negative")
            if not isinstance(raw_hits, list):
                raise TypeError("hits is not a list")

            return cls(
                query=query,
                hits=tuple(SearchHit.from_loogle_payload(hit) for hit in raw_hits),
                total_count=total_count,
                header=header,
                suggestions=suggestions,
                duration_s=duration_s,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
            raise SearchInfrastructureError("Loogle returned an invalid JSON response") from error


class SearchBackend(Protocol):
    async def search(self, query: str) -> SearchResult: ...


class SearchInfrastructureError(RuntimeError):
    """Loogle could not be queried reliably."""


class InvalidSearchQuery(ValueError):
    """The query cannot be represented as one Loogle request."""


class DockerLoogleBackend:
    def __init__(
        self,
        settings: SandboxSettings,
    ) -> None:
        self._settings = settings
        self._process: asyncio.subprocess.Process | None = None
        self._container_name: str | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr = bytearray()
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def start(self) -> None:
        if self._process is not None:
            return

        container_name = f"formalizer-loogle-{uuid4().hex}"
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
            str(self._settings.pids_limit),
            "--cpus",
            str(self._settings.cpus),
            "--memory",
            f"{self._settings.memory_mib}m",
            "--memory-swap",
            f"{self._settings.memory_mib}m",
            "--tmpfs",
            (f"/tmp:rw,nosuid,nodev,size={self._settings.tmp_mib}m,uid=10001,gid=10001,mode=1777"),
            "--workdir",
            "/opt/mathlib",
            self._settings.docker_image,
            "lake",
            "env",
            "/opt/loogle/.lake/build/bin/loogle",
            "--interactive",
            "--json",
            "--module",
            "Mathlib",
            "--index-mode",
            "read",
            "--index-file",
            "/opt/loogle/mathlib.loogle-index",
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise SearchInfrastructureError("Could not start Docker") from error

        stdout = process.stdout
        stderr = process.stderr
        assert stdout is not None
        assert stderr is not None

        self._process = process
        self._container_name = container_name
        self._stderr.clear()
        self._stderr_task = asyncio.create_task(self._drain_stderr(stderr))

        try:
            ready_line = await asyncio.wait_for(
                stdout.readline(),
                timeout=self._settings.search_timeout_s,
            )
        except TimeoutError as error:
            await asyncio.shield(self.close())
            raise SearchInfrastructureError("Loogle did not become ready in time") from error
        except asyncio.CancelledError:
            await asyncio.shield(self.close())
            raise

        readiness = ready_line.decode(errors="replace").strip()
        if readiness != "Loogle is ready.":
            await self.close()
            diagnostic = readiness or self._stderr_text() or "Loogle exited before becoming ready"
            raise SearchInfrastructureError(f"Unexpected Loogle startup response: {diagnostic}")

    async def close(self) -> None:
        process = self._process
        if process is None:
            return

        if process.stdin is not None:
            process.stdin.close()

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=self._settings.search_timeout_s,
            )
        except TimeoutError:
            await self._retire()
            return

        await self._finalize_process(process)

    async def _retire(self) -> None:
        process = self._process
        if process is None:
            return

        try:
            await self._remove_container()
        finally:
            await self._finalize_process(process)

    async def _finalize_process(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()

        stderr_task = self._stderr_task
        if stderr_task is not None:
            await stderr_task

        if self._process is process:
            self._process = None
            self._container_name = None
            self._stderr_task = None

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        while chunk := await stderr.read(8192):
            self._stderr.extend(chunk)

    def _stderr_text(self) -> str:
        return self._stderr.decode(errors="replace").strip()

    async def _remove_container(self) -> None:
        container_name = self._container_name
        if container_name is None:
            return

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
                timeout=_DOCKER_CLEANUP_TIMEOUT_S,
            )
        except OSError as error:
            raise SearchInfrastructureError(
                f"Could not remove Loogle container {container_name}"
            ) from error
        except TimeoutError as error:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.communicate()
            raise SearchInfrastructureError(
                f"Could not remove Loogle container {container_name}"
            ) from error

        if process.returncode != 0:
            diagnostic = stderr.decode(errors="replace").strip()
            raise SearchInfrastructureError(
                f"Could not remove Loogle container {container_name}: {diagnostic}"
            )

    async def search(self, query: str) -> SearchResult:
        query = self._validate_query(query)

        async with self._lock:
            process = self._process

            if process is None:
                raise SearchInfrastructureError("Loogle backend is not running")

            if process.returncode is not None:
                return_code = process.returncode
                await self.close()
                raise SearchInfrastructureError(
                    f"Loogle process exited with status {return_code}: {self._stderr_text()}"
                )

            stdin = process.stdin
            stdout = process.stdout

            if stdin is None or stdout is None:
                await asyncio.shield(self._retire())
                raise SearchInfrastructureError("Loogle protocol pipes are unavailable")

            started_at = monotonic()
            try:
                async with asyncio.timeout(self._settings.search_timeout_s):
                    stdin.write(f"{query}\n".encode())
                    await stdin.drain()
                    response_line = await stdout.readline()
            except TimeoutError as error:
                try:
                    await asyncio.shield(self._retire())
                except SearchInfrastructureError as cleanup_error:
                    raise SearchInfrastructureError(
                        f"Loogle query timed out and cleanup failed: {cleanup_error}"
                    ) from error
                raise SearchInfrastructureError("Loogle query timed out") from error
            except asyncio.CancelledError as error:
                try:
                    await asyncio.shield(self._retire())
                except SearchInfrastructureError as cleanup_error:
                    error.add_note(f"Loogle cleanup failed: {cleanup_error}")
                raise
            except (BrokenPipeError, ConnectionResetError) as error:
                await asyncio.shield(self._retire())
                raise SearchInfrastructureError("Lost connection to Loogle") from error

            if not response_line:
                await asyncio.shield(self._retire())
                raise SearchInfrastructureError("Loogle exited without returning a response")

            duration_s = monotonic() - started_at

            try:
                return SearchResult.from_loogle_json(
                    query=query,
                    response_line=response_line,
                    duration_s=duration_s,
                )
            except SearchInfrastructureError:
                await asyncio.shield(self._retire())
                raise

    def _validate_query(self, query: str) -> str:
        if "\n" in query or "\r" in query:
            raise InvalidSearchQuery("Search query must contain exactly one line")

        query = query.strip()
        if not query:
            raise InvalidSearchQuery("Search query must not be blank")

        return query
