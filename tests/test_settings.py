import math

import pytest
from httpx import Timeout
from pydantic import ValidationError

from formalizer.settings import Settings


@pytest.mark.parametrize("model_name", ["", " ", "\t\n"])
def test_model_name_must_not_be_blank(model_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(model_name=model_name)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sandbox": {"cpus": 0}},
        {"sandbox": {"cpus": math.inf}},
        {"sandbox": {"memory_mib": 255}},
        {"sandbox": {"pids_limit": 0}},
        {"sandbox": {"workspace_mib": 15}},
        {"sandbox": {"tmp_mib": 15}},
        {"sandbox": {"lean_timeout_s": 0}},
        {"sandbox": {"search_timeout_s": 0}},
        {"sandbox": {"search_max_results": 0}},
        {"sandbox": {"search_max_results": 101}},
        {"run": {"run_timeout_s": 0}},
    ],
)
def test_unsafe_resource_limits_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "model_name": "test-model",
                **overrides,
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"model_name": "test-model", "sandox": {}},
        {
            "model_name": "test-model",
            "model_settings": {"temprature": 0.0},
        },
    ],
)
def test_unknown_configuration_keys_are_rejected(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_settings_cannot_change_during_a_run() -> None:
    settings = Settings(model_name="test-model")

    with pytest.raises(ValidationError):
        settings.sandbox.cpus = 4  # ty: ignore[invalid-assignment]


def test_loogle_backend_result_ceiling_defaults_to_one_hundred() -> None:
    settings = Settings(model_name="test-model")

    assert settings.sandbox.search_max_results == 100


def test_model_http_timeout_can_use_the_provider_timeout_type() -> None:
    timeout = Timeout(60)

    settings = Settings(
        model_name="test-model",
        model_settings={"timeout": timeout},
    )

    assert settings.model_settings["timeout"] is timeout
