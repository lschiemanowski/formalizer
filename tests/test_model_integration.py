import os
from pathlib import Path

import pytest
from pydantic_ai import ModelSettings, UsageLimits, models

from formalizer.agent import VerifiedSubmission
from formalizer.run import RunManifest, formalize
from formalizer.settings import RunSettings, Settings

_MODEL_NAME = "openrouter:deepseek/deepseek-v4-flash-0731"
_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
_PROBLEM = "Prove that 1 + 1 = 2."


@pytest.mark.integration
@pytest.mark.model
async def test_deepseek_v4_flash_produces_a_persisted_verified_submission(
    tmp_path: Path,
) -> None:
    if not os.environ.get(_API_KEY_ENV_VAR):
        pytest.skip(f"{_API_KEY_ENV_VAR} is not configured")

    settings = Settings(
        model_name=_MODEL_NAME,
        model_settings=ModelSettings(max_tokens=4096),
        run=RunSettings(
            runs_dir=tmp_path,
            run_timeout_s=300,
            usage_limits=UsageLimits(
                request_limit=8,
                output_tokens_limit=20_000,
                total_tokens_limit=30_000,
            ),
        ),
    )

    with models.override_allow_model_requests(True):
        result = await formalize(_PROBLEM, settings)

    run_dir = tmp_path / result.run_id
    manifest = RunManifest.model_validate_json((run_dir / "run.json").read_bytes())

    assert isinstance(result.output, VerifiedSubmission)
    assert result.output.code
    assert str(manifest.run_id) == result.run_id
    assert manifest.problem == _PROBLEM
    assert manifest.status == "verified"
    assert (run_dir / "messages.json").stat().st_size > 0
    assert (run_dir / "final.lean").read_text() == result.output.code
