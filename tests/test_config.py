"""配置加载与路径解析测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from medscholar.config import (
    AppConfig,
    _deep_merge,
    _env_overrides,
    data_home,
)


class TestDefaults:
    def test_sane_defaults(self):
        cfg = AppConfig()
        assert cfg.llm.provider == "ollama"
        assert cfg.embedding.provider == "ollama"
        assert cfg.embedding.dim == 768
        assert cfg.retrieval.rrf_k == 60
        assert cfg.agent.require_approval is True

    def test_rate_limits_match_published_quotas(self):
        """速率默认值必须贴合各 API 的公开限制，否则会被限流甚至封禁。"""
        sources = AppConfig().sources
        assert sources.pubmed.rps == 3.0, "无 Key 时 NCBI 限制为 3 次/秒"
        assert sources.europepmc.rps <= 10.0
        assert sources.semantic_scholar.rps <= 0.34, "无 Key 约 100 次/5 分钟"
        assert sources.arxiv.rps <= 0.34, "arXiv 要求间隔 ≥3 秒"
        assert sources.cnki.enabled is False, "CNKI 公开检索已不可用，默认关闭"

    def test_pubmed_rate_raised_with_key(self):
        from medscholar.api.pubmed_client import PubMedClient

        plain = PubMedClient()
        with_key = PubMedClient(
            AppConfig().sources.pubmed.model_copy(update={"api_key": "abc", "rps": 3.0})
        )
        assert plain.bucket.rps == 3.0
        assert with_key.bucket.rps == 10.0

    def test_cnki_disabled_by_default(self):
        assert AppConfig().sources.get("cnki").enabled is False


class TestPaths:
    def test_home_follows_data_dir(self, tmp_path):
        cfg = AppConfig(data_dir=str(tmp_path))
        assert cfg.home == tmp_path.resolve()

    def test_db_path_under_home(self, tmp_path):
        cfg = AppConfig(data_dir=str(tmp_path))
        assert cfg.db_path.parent == tmp_path.resolve()
        assert cfg.db_path.name == "medscholar.db"

    def test_ensure_dirs_creates_all(self, tmp_path):
        cfg = AppConfig(data_dir=str(tmp_path / "nested"))
        cfg.ensure_dirs()
        for path in (cfg.home, cfg.fulltext_dir, cfg.export_dir, cfg.upload_dir):
            assert path.is_dir()

    def test_data_home_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEDSCHOLAR_HOME", str(tmp_path / "custom"))
        assert data_home() == (tmp_path / "custom").resolve()


class TestPublicDict:
    def test_no_api_key_leak(self):
        cfg = AppConfig()
        cfg.llm.api_key = "sk-secret-value"
        cfg.sources.pubmed.api_key = "ncbi-secret"
        payload = cfg.public_dict()

        serialized = str(payload)
        assert "sk-secret-value" not in serialized
        assert "ncbi-secret" not in serialized
        assert payload["llm"]["has_api_key"] is True
        assert payload["sources"]["pubmed"]["has_api_key"] is True

    def test_contains_expected_sections(self):
        payload = AppConfig().public_dict()
        for key in ("app_name", "offline", "db_path", "llm", "embedding", "retrieval", "sources"):
            assert key in payload


class TestMerge:
    def test_deep_merge_nested(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"c": 9}, "e": 5}
        merged = _deep_merge(base, override)
        assert merged == {"a": {"b": 1, "c": 9}, "d": 3, "e": 5}

    def test_override_replaces_non_dict(self):
        assert _deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}


class TestEnvOverrides:
    def test_api_keys_picked_up(self, monkeypatch):
        monkeypatch.setenv("NCBI_API_KEY", "k1")
        monkeypatch.setenv("S2_API_KEY", "k2")
        monkeypatch.setenv("OPENALEX_EMAIL", "me@example.org")
        cfg = _env_overrides({})
        assert cfg["sources"]["pubmed"]["api_key"] == "k1"
        assert cfg["sources"]["semantic_scholar"]["api_key"] == "k2"
        assert cfg["sources"]["openalex"]["email"] == "me@example.org"

    def test_deepseek_key_does_not_hijack_provider(self, monkeypatch):
        """**核心回归**：环境里存在 DEEPSEEK_API_KEY，不得改变 provider。

        真实故障：用户机器上别的项目把 DEEPSEEK_API_KEY 设成了用户级环境变量，
        早期版本因此静默把后端从 ollama 切成 deepseek，却沿用 config.yaml 里的
        model: qwen3:8b，最终把 Ollama 的模型名发给了云端网关，得到 HTTP 400。
        结论：环境里有某个 Key，不等于用户想让本程序用它。
        """
        monkeypatch.delenv("MEDSCHOLAR_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        cfg = _env_overrides({})
        assert "provider" not in cfg.get("llm", {}), "环境变量不得自动切换 provider"
        assert cfg["llm"]["api_key"] == "sk-x", "Key 本身仍应可用"

    def test_explicit_provider_wins(self, monkeypatch):
        monkeypatch.setenv("MEDSCHOLAR_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        cfg = _env_overrides({})
        assert cfg["llm"]["provider"] == "ollama"

    def test_deepseek_base_url_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.example.com/v1")
        assert _env_overrides({})["llm"]["base_url"] == "https://gateway.example.com/v1"

    def test_offline_flag(self, monkeypatch):
        monkeypatch.setenv("MEDSCHOLAR_OFFLINE", "true")
        assert _env_overrides({})["offline"] is True


class TestLlmConsistency:
    """provider 与 model 的配套校验。

    对应真实故障：provider=deepseek 却配着 model=qwen3:8b，请求发到云端必然
    HTTP 400，而云端返回的是英文错误、也不告诉你该怎么改。
    """

    @pytest.mark.parametrize(
        "provider,model,should_error",
        [
            ("ollama", "qwen3:8b", False),
            ("ollama", "llama3.2:3b", False),
            ("ollama", "nomic-embed-text:latest", False),
            ("deepseek", "deepseek-chat", False),
            ("deepseek", "deepseek-flash", False),
            ("deepseek", "deepseek-v4-pro", False),
            ("openai-compatible", "gpt-4o", False),
            ("deepseek", "qwen3:8b", True),
            ("deepseek", "llama3.2:3b", True),
            ("openai-compatible", "qwen3:8b", True),
            ("deepseek", "deepseek-r1:7b", True),
        ],
    )
    def test_consistency(self, provider, model, should_error):
        from medscholar.config import LLMSettings

        message = LLMSettings(provider=provider, model=model).consistency_error()
        assert bool(message) is should_error

    def test_error_message_is_actionable(self):
        from medscholar.config import LLMSettings

        message = LLMSettings(provider="deepseek", model="qwen3:8b").consistency_error()
        assert "llm.provider" in message and "llm.model" in message
        assert "ollama" in message, "必须告诉用户可以改回 ollama"
        assert "HTTP 400" in message, "必须说明云端会拒绝"

    def test_empty_model_flagged(self):
        from medscholar.config import LLMSettings

        assert LLMSettings(provider="ollama", model="").consistency_error()

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("qwen3:8b", True),
            ("deepseek-r1:7b", True),
            ("nomic-embed-text:latest", True),
            ("llama3.2:3b", True),
            ("deepseek-chat", False),
            ("deepseek-flash", False),
            ("deepseek-v4-pro", False),
            ("gpt-4o", False),
        ],
    )
    def test_ollama_model_detection(self, name, expected):
        from medscholar.config import _looks_like_ollama_model

        assert _looks_like_ollama_model(name) is expected

    async def test_client_start_raises_readable_error(self):
        from medscholar.config import LLMSettings
        from medscholar.llm.client import LLMClient, LLMError

        client = LLMClient(LLMSettings(provider="deepseek", model="qwen3:8b", api_key="sk-x"))
        with pytest.raises(LLMError, match="llm.provider"):
            await client.start()

    async def test_health_reports_consistency_error(self):
        from medscholar.config import LLMSettings
        from medscholar.llm.client import LLMClient

        client = LLMClient(LLMSettings(provider="deepseek", model="qwen3:8b", api_key="sk-x"))
        ok, message = await client.health()
        assert ok is False
        assert "配置不一致" in message


class TestOllamaKeepAlive:
    """keep_alive 必须下发：Ollama 默认 5 分钟就卸载模型，重新加载要 60~120 秒。"""

    def test_default_keep_alive_is_not_ollama_default(self):
        from medscholar.config import LLMSettings

        settings = LLMSettings()
        assert settings.keep_alive, "必须显式设置，否则会落到 Ollama 的 5 分钟默认值"
        assert settings.keep_alive != "5m"

    async def test_payload_carries_keep_alive(self, monkeypatch):
        import asyncio
        import httpx
        import json as _json

        from medscholar.config import LLMSettings
        from medscholar.llm.client import LLMClient

        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(_json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "message": {"content": '{"ok": true}'},
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )

        settings = LLMSettings(provider="ollama", model="qwen3:8b", keep_alive="45m")
        client = LLMClient(settings)
        await client.start()
        # _ensure_client 会按事件循环缓存客户端，必须连 loop 一起换掉，
        # 否则注入的 MockTransport 会在下一次调用时被重建的 httpx 客户端覆盖。
        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=httpx.MockTransport(handler),
        )
        client._client_loop = asyncio.get_running_loop()

        await client.chat(
            [{"role": "user", "content": "hi"}], max_tokens=16
        )

        assert seen, "没有发出请求"
        assert seen[0].get("keep_alive") == "45m"
        assert seen[0]["options"]["num_predict"] == 16
        await client.close()


class TestYamlLoading:
    def test_config_file_parsed(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    "llm": {"provider": "deepseek", "model": "deepseek-chat"},
                    "retrieval": {"rrf_k": 30},
                    "sources": {"pubmed": {"rps": 9.0}},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("MEDSCHOLAR_CONFIG", str(cfg_file))
        from medscholar.config import reload_config

        cfg = reload_config()
        assert cfg.llm.provider == "deepseek"
        assert cfg.retrieval.rrf_k == 30
        assert cfg.sources.pubmed.rps == 9.0
        # 未覆盖的字段保持默认
        assert cfg.embedding.dim == 768

    def test_broken_yaml_raises_readable_error(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.yaml"
        bad.write_text("llm: [unclosed", encoding="utf-8")
        monkeypatch.setenv("MEDSCHOLAR_CONFIG", str(bad))
        from medscholar.config import reload_config

        with pytest.raises(RuntimeError, match="配置文件解析失败"):
            reload_config()

    def test_invalid_rps_clamped(self):
        from medscholar.config import SourceSettings

        assert SourceSettings(rps=0.0).rps >= 0.05
        assert SourceSettings(rps=-5).rps >= 0.05


class TestValidateAgainstSample:
    def test_example_config_is_valid(self):
        """config.example.yaml 必须能真正被解析（否则用户复制后无法启动）。"""
        path = Path(__file__).resolve().parent.parent / "config.example.yaml"
        if not path.exists():
            pytest.skip("config.example.yaml 不存在")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = AppConfig.model_validate(raw)
        assert cfg.sources.pubmed.rps == 3.0
        assert cfg.sources.cnki.enabled is False

    def test_dotenv_example_exists_and_has_no_real_secrets(self):
        path = Path(__file__).resolve().parent.parent / ".env.example"
        if not path.exists():
            pytest.skip(".env.example 不存在")
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                key, _, value = stripped.partition("=")
                assert value.strip() == "", f"{key} 在模板里不应含真实值"
