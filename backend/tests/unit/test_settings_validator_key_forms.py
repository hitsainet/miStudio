"""The key forms a `mode="before"` validator receives, pinned against the INSTALLED pydantic-settings
(042 FTID §3.2, TI-1, TI-2; FTASKS 3.1).

`core/config.apply_legacy_env` decides "is the new field present?" by these forms, and reads an old name
from a `.env` file under its lower-case form. They were MEASURED (2026-11-08, pydantic-settings 2.16.1),
not documented by the library; the image may ship another version than the dev venv (suite memory
`dev-venv-vs-image-version-drift`). If a release moves a form, this test turns red instead of a fallback
silently disappearing.
"""
from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core import config, legacy_names

SEEN: list[dict[str, Any]] = []


class _Two(BaseSettings):
    """Two fields shaped like the production ones: one aliased, one plain."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")
    bypass_settings_pin: bool = Field(default=False,
                                      validation_alias=AliasChoices("MIWORKBENCH_BYPASS_PIN", "bypass_settings_pin"))
    miserver_api_url: str = ""

    @model_validator(mode="before")
    @classmethod
    def _seen(cls, data):
        SEEN.append(dict(data) if isinstance(data, dict) else {"<not a dict>": data})
        return config.apply_legacy_env(data)


def _keys(monkeypatch, tmp_path, env=None, dotenv=None, **init):
    for name in ("MIWORKBENCH_BYPASS_PIN", "BYPASS_SETTINGS_PIN", "MISTUDIO_BYPASS_PIN", "MISERVER_API_URL",
                 "MILLM_API_URL"):
        monkeypatch.delenv(name, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    env_file = None
    if dotenv is not None:
        env_file = tmp_path / ".env"
        env_file.write_text("".join(f"{k}={v}\n" for k, v in dotenv.items()))
    SEEN.clear()
    model = _Two(_env_file=env_file, **init)
    return set(SEEN[-1]), model


def test_measured_key_forms(monkeypatch, tmp_path):
    legacy_names.reset()
    # 1. an aliased field set by EITHER choice arrives keyed by the FIRST alias
    assert _keys(monkeypatch, tmp_path, env={"BYPASS_SETTINGS_PIN": "1"})[0] == {"MIWORKBENCH_BYPASS_PIN"}
    assert _keys(monkeypatch, tmp_path, env={"MIWORKBENCH_BYPASS_PIN": "1"})[0] == {"MIWORKBENCH_BYPASS_PIN"}
    # 2. a plain field from the environment arrives by its lower-case field name
    assert _keys(monkeypatch, tmp_path, env={"MISERVER_API_URL": "http://a"})[0] == {"miserver_api_url"}
    # 3. an init keyword arrives under the field name
    assert _keys(monkeypatch, tmp_path, bypass_settings_pin=True)[0] == {"bypass_settings_pin"}
    # 4. an OLD name kept in a .env file arrives under its lower-case name (extra="ignore" notwithstanding)
    keys, model = _keys(monkeypatch, tmp_path, dotenv={"MILLM_API_URL": "http://old"})
    assert keys == {"millm_api_url"}
    assert model.miserver_api_url == "", "the row is retired since ENH-009 P7: the old value is not read"
    # ... and an old name in the process environment does NOT arrive at all (os.environ must be read)
    keys, model = _keys(monkeypatch, tmp_path, env={"MILLM_API_URL": "http://old-env"})
    assert keys == set()
    assert model.miserver_api_url == ""
