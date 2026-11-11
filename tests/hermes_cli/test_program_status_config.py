"""The Ink status preference survives the profile config and native projection."""

from gateway.session_config import _client_config
from hermes_cli.config import load_config
from hermes_cli.config_effective import load_user_config_effective


def test_program_status_defaults_on_and_preserves_explicit_profile_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert load_config()["display"]["program_status"] is True

    config_path = tmp_path / "config.yaml"
    config_path.write_text("display:\n  program_status: false\n", encoding="utf-8")
    assert load_config()["display"]["program_status"] is False
    # config.get(full), consumed by Ink, uses effective user config rather than defaults.
    projected = _client_config(load_user_config_effective(config_path))
    assert projected["display"]["program_status"] is False
