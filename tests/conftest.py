from collections.abc import Iterator

import pytest
from pydantic_ai import models


@pytest.fixture(autouse=True)
def block_unintended_model_requests() -> Iterator[None]:
    with models.override_allow_model_requests(False):
        yield
