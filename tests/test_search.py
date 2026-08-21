import asyncio

import pytest

from formalizer.search import (
    DockerLoogleBackend,
    InvalidSearchQuery,
    SearchHit,
    SearchInfrastructureError,
    SearchResult,
)
from formalizer.settings import SandboxSettings


@pytest.fixture
def backend() -> DockerLoogleBackend:
    return DockerLoogleBackend(
        settings=SandboxSettings(),
    )


async def formalizer_loogle_container_ids() -> set[str]:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "--all",
        "--quiet",
        "--filter",
        "name=formalizer-loogle-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace"))

    return set(stdout.decode().splitlines())


async def docker_container_command(*args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace"))


async def remove_containers(container_ids: set[str]) -> None:
    for container_id in container_ids:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "--force",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()


def test_successful_loogle_response_is_decoded() -> None:
    result = SearchResult.from_loogle_json(
        query="Nat.add_comm",
        response_line=b"""
        {
            "header": "Found 2 declarations",
            "heartbeats": 37,
            "count": 2,
            "hits": [
                {
                    "name": "Nat.add_comm",
                    "type": "Nat.add_comm (n m : Nat) : n + m = m + n",
                    "module": "Mathlib.Data.Nat.Basic",
                    "doc": "Addition is commutative."
                },
                {
                    "name": "Example.unattributed",
                    "type": "True",
                    "module": null,
                    "doc": null
                }
            ],
            "suggestions": ["Nat.add_assoc"]
        }
        """,
        duration_s=0.25,
    )

    assert result == SearchResult(
        query="Nat.add_comm",
        hits=(
            SearchHit(
                name="Nat.add_comm",
                type="Nat.add_comm (n m : Nat) : n + m = m + n",
                module="Mathlib.Data.Nat.Basic",
                doc="Addition is commutative.",
            ),
            SearchHit(
                name="Example.unattributed",
                type="True",
                module=None,
                doc=None,
            ),
        ),
        total_count=2,
        header="Found 2 declarations",
        suggestions=("Nat.add_assoc",),
        duration_s=0.25,
    )


def test_loogle_query_error_is_decoded_as_a_result() -> None:
    result = SearchResult.from_loogle_json(
        query="this_should_not_exist_zzzz",
        response_line=(
            b'{"error": "Could not parse query", "suggestions": '
            b'["\\"this_should_not_exist_zzzz\\""], "heartbeats": 12}'
        ),
        duration_s=0.1,
    )

    assert result == SearchResult(
        query="this_should_not_exist_zzzz",
        hits=(),
        total_count=0,
        header="",
        error="Could not parse query",
        suggestions=('"this_should_not_exist_zzzz"',),
        duration_s=0.1,
    )


@pytest.mark.parametrize(
    "response_line",
    [
        b"not JSON",
        b"[]",
        b'{"header": "Found declarations", "count": 1}',
        (
            b'{"header": "Found declarations", "count": 1, "hits": '
            b'[{"name": "bad", "type": "True", "module": 42, "doc": null}]}'
        ),
        b'{"error": "Bad query", "suggestions": "not a list"}',
    ],
)
def test_malformed_loogle_response_is_an_infrastructure_error(
    response_line: bytes,
) -> None:
    with pytest.raises(SearchInfrastructureError):
        SearchResult.from_loogle_json(
            query="Nat.add_comm",
            response_line=response_line,
            duration_s=0.1,
        )


@pytest.mark.integration
async def test_loogle_finds_known_declaration(
    backend: DockerLoogleBackend,
) -> None:
    async with backend as search:
        result = await search.search("Nat.add_comm")

    assert result.error is None
    assert any(hit.name == "Nat.add_comm" for hit in result.hits)


@pytest.mark.integration
async def test_persistent_loogle_handles_repeated_queries(
    backend: DockerLoogleBackend,
) -> None:
    async with backend:
        addition = await backend.search("Nat.add_comm")
        multiplication = await backend.search("Nat.mul_comm")

    assert any(hit.name == "Nat.add_comm" for hit in addition.hits)
    assert any(hit.name == "Nat.mul_comm" for hit in multiplication.hits)


@pytest.mark.integration
async def test_loogle_valid_empty_result(
    backend: DockerLoogleBackend,
) -> None:
    async with backend as search:
        result = await search.search('"this_should_not_exist_zzzz"')

    assert result.error is None
    assert result.total_count == 0
    assert result.hits == ()


@pytest.mark.integration
async def test_loogle_invalid_query(
    backend: DockerLoogleBackend,
) -> None:
    async with backend as search:
        result = await search.search("this_should_not_exist_zzzz")

    assert result.error is not None
    assert result.suggestions


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "one\ntwo",
        "one\rtwo",
    ],
)
async def test_invalid_search_input_is_rejected(
    backend: DockerLoogleBackend,
    query: str,
) -> None:
    with pytest.raises(InvalidSearchQuery):
        await backend.search(query)


@pytest.mark.integration
async def test_concurrent_queries_receive_their_own_results(
    backend: DockerLoogleBackend,
) -> None:
    async with backend:
        addition, multiplication = await asyncio.gather(
            backend.search("Nat.add_comm"),
            backend.search("Nat.mul_comm"),
        )

    assert any(hit.name == "Nat.add_comm" for hit in addition.hits)
    assert any(hit.name == "Nat.mul_comm" for hit in multiplication.hits)


@pytest.mark.integration
async def test_missing_image_is_an_infrastructure_error() -> None:
    containers_before = await formalizer_loogle_container_ids()
    backend = DockerLoogleBackend(
        SandboxSettings(
            docker_image="formalizer-image-that-does-not-exist",
        )
    )

    with pytest.raises(SearchInfrastructureError):
        await backend.start()

    containers_after = await formalizer_loogle_container_ids()
    assert containers_after <= containers_before


@pytest.mark.integration
async def test_context_exit_removes_loogle_container(
    backend: DockerLoogleBackend,
) -> None:
    containers_before = await formalizer_loogle_container_ids()

    async with backend:
        containers_during = await formalizer_loogle_container_ids()
        assert containers_during - containers_before

    containers_after = await formalizer_loogle_container_ids()
    assert containers_after <= containers_before


@pytest.mark.integration
async def test_timed_out_query_removes_loogle_container(
    backend: DockerLoogleBackend,
) -> None:
    containers_before = await formalizer_loogle_container_ids()
    await backend.start()
    new_containers = (await formalizer_loogle_container_ids()) - containers_before
    assert len(new_containers) == 1

    container_id = next(iter(new_containers))
    backend._settings = backend._settings.model_copy(update={"search_timeout_s": 0.2})
    await docker_container_command("pause", container_id)

    try:
        with pytest.raises(SearchInfrastructureError, match="timed out"):
            await backend.search("Nat.add_comm")

        containers_after = await formalizer_loogle_container_ids()
        assert container_id not in containers_after
    finally:
        await remove_containers(new_containers)
        await backend.close()


@pytest.mark.integration
async def test_cancelled_query_removes_loogle_container(
    backend: DockerLoogleBackend,
) -> None:
    containers_before = await formalizer_loogle_container_ids()
    await backend.start()
    new_containers = (await formalizer_loogle_container_ids()) - containers_before
    assert len(new_containers) == 1

    container_id = next(iter(new_containers))
    backend._settings = backend._settings.model_copy(update={"search_timeout_s": 0.2})
    await docker_container_command("pause", container_id)
    task = asyncio.create_task(backend.search("Nat.add_comm"))

    try:
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        containers_after = await formalizer_loogle_container_ids()
        assert container_id not in containers_after
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await remove_containers(new_containers)
        await backend.close()


@pytest.mark.integration
async def test_process_death_during_query_is_an_infrastructure_error(
    backend: DockerLoogleBackend,
) -> None:
    containers_before = await formalizer_loogle_container_ids()
    await backend.start()
    new_containers = (await formalizer_loogle_container_ids()) - containers_before
    assert len(new_containers) == 1

    container_id = next(iter(new_containers))
    await docker_container_command("pause", container_id)
    task = asyncio.create_task(backend.search("Nat.add_comm"))

    try:
        await asyncio.sleep(0.05)
        await docker_container_command("rm", "--force", container_id)

        with pytest.raises(SearchInfrastructureError):
            await task

        containers_after = await formalizer_loogle_container_ids()
        assert containers_after <= containers_before
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await remove_containers(new_containers)
        await backend.close()
