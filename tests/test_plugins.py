from importlib.metadata import EntryPoint

import pytest

from vardex import cli, plugins
from vardex.config import Settings
from vardex.llm import ModelError
from vardex.sql import QueryError


def test_a_model_reference_names_its_provider_or_means_the_default():
    assert plugins.split_model("claude-sonnet-5-5") == ("anthropic", "claude-sonnet-5-5")
    assert plugins.split_model("ollama/gemma4:12b") == ("ollama", "gemma4:12b")
    # Everything after the first slash is the model's own name
    assert plugins.split_model("openai/Qwen/Qwen3.5-9B") == ("openai", "Qwen/Qwen3.5-9B")
    with pytest.raises(plugins.PluginError):
        plugins.split_model("ollama/")


def test_vardex_declares_its_own_connectors_as_entry_points():
    assert {"anthropic", "openai", "ollama"} <= set(plugins.available("providers"))
    assert {"duckdb", "postgresql", "postgres"} <= set(plugins.available("warehouses"))


def test_an_unknown_plugin_lists_the_ones_installed():
    with pytest.raises(plugins.PluginError) as caught:
        plugins.load("providers", "mistral")
    assert "No model provider called 'mistral'" in str(caught.value)
    assert "anthropic" in str(caught.value)


def test_a_plugin_whose_package_is_missing_says_which_extra_to_install(monkeypatch):
    missing = EntryPoint("gemini", "vardex.gemini_client_not_installed:connect", "vardex.providers")
    monkeypatch.setattr(plugins, "entry_points", lambda group: [missing])
    with pytest.raises(plugins.PluginError) as caught:
        plugins.load("providers", "gemini")
    assert "pip install 'vardex[gemini]'" in str(caught.value)


def test_connect_finds_the_provider_the_model_names(monkeypatch):
    made = []

    def fake_provider(settings, model):
        made.append(model)
        return "a client"

    monkeypatch.setattr(plugins, "load", lambda kind, name: fake_provider)
    assert cli.connect(Settings(model="ollama/gemma4:12b")) == "a client"
    assert made == ["gemma4:12b"]


def test_a_provider_that_is_not_installed_is_a_model_error():
    with pytest.raises(ModelError):
        cli.connect(Settings(model="mistral/large"))


def test_a_warehouse_is_found_by_its_scheme_or_its_suffix(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(plugins, "load", lambda kind, name: lambda url, s: asked.append(name))
    settings = Settings(model="m")
    cli.connect_warehouse("postgresql://reader@db/warehouse", tmp_path, settings)
    cli.connect_warehouse(str(tmp_path / "copy.duckdb"), tmp_path, settings)
    assert asked == ["postgresql", "duckdb"]


def test_a_warehouse_nobody_offers_is_a_query_error(tmp_path):
    with pytest.raises(QueryError) as caught:
        cli.connect_warehouse("snowflake://account/db", tmp_path, Settings(model="m"))
    assert "No warehouse called 'snowflake'" in str(caught.value)
