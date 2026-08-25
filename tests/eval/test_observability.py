import pytest

import formalizer.eval.observability as observability_module
from formalizer.eval import configure_logfire


def test_logfire_configuration_is_optional_and_instruments_pydantic_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_configure(**kwargs: object) -> None:
        calls.append(("configure", kwargs))

    def fake_instrument_pydantic_ai() -> None:
        calls.append(("instrument_pydantic_ai", None))

    monkeypatch.setattr(observability_module.logfire, "configure", fake_configure)
    monkeypatch.setattr(
        observability_module.logfire,
        "instrument_pydantic_ai",
        fake_instrument_pydantic_ai,
    )

    configure_logfire()

    assert calls == [
        (
            "configure",
            {
                "send_to_logfire": "if-token-present",
                "service_name": "formalizer",
            },
        ),
        ("instrument_pydantic_ai", None),
    ]
