from formalizer.lean import LeanResult


def test_zero_exit_is_accepted() -> None:
    result = LeanResult(
        stdout="",
        stderr="",
        exit_code=0,
        duration_s=0.1,
    )

    assert result.accepted


def test_nonzero_exit_is_rejected() -> None:
    result = LeanResult(
        stdout="",
        stderr="proof failed",
        exit_code=1,
        duration_s=0.1,
    )

    assert not result.accepted


def test_timeout_is_not_accepted() -> None:
    result = LeanResult(
        stdout="",
        stderr="",
        exit_code=None,
        duration_s=30,
        timed_out=True,
    )

    assert not result.accepted
