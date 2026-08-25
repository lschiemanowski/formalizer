import logfire


def configure_logfire() -> None:
    logfire.configure(
        send_to_logfire="if-token-present",
        service_name="formalizer",
    )
    logfire.instrument_pydantic_ai()
