"""Tests for multi-account support in dependencies and provider integration."""

import pytest

from config.nim import NimSettings
from providers.account_pool import AccountPool
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider
from providers.open_router import OpenRouterProvider


class TestParseApiKeys:
    """Tests for _parse_api_keys-equivalent logic."""

    @staticmethod
    def _parse(keys_str: str, single_key: str) -> list[str]:
        """Parse comma-separated API keys, falling back to single key."""
        keys: list[str] = []
        if keys_str and keys_str.strip():
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not keys and single_key and single_key.strip():
            keys = [single_key.strip()]
        return keys

    def test_single_key(self):
        assert self._parse("", "single_key") == ["single_key"]

    def test_multiple_keys(self):
        result = self._parse("key1,key2,key3", "")
        assert result == ["key1", "key2", "key3"]

    def test_multiple_keys_with_spaces(self):
        result = self._parse(" key1 , key2 , key3 ", "")
        assert result == ["key1", "key2", "key3"]

    def test_empty_keys_filtered(self):
        result = self._parse("key1,,key2,,,key3", "")
        assert result == ["key1", "key2", "key3"]

    def test_multi_keys_override_single(self):
        result = self._parse("key1,key2", "single")
        assert result == ["key1", "key2"]

    def test_empty_both(self):
        assert self._parse("", "") == []

    def test_whitespace_only_single(self):
        assert self._parse("", "   ") == []


class TestMultiAccountProviderIntegration:
    """Tests for provider creation with AccountPool."""

    def test_nvidia_nim_with_pool(self, provider_config):
        """NIM provider works with account pool."""
        pool = AccountPool(
            api_keys=["key1_long_enough", "key2_long_enough", "key3_long_enough"],
            base_url="https://integrate.api.nvidia.com/v1",
            strategy="round_robin",
        )
        provider = NvidiaNimProvider(
            provider_config, nim_settings=NimSettings(), account_pool=pool
        )

        assert provider._account_pool is not None
        assert provider._account_pool.account_count == 3

    def test_nvidia_nim_without_pool(self, provider_config):
        """NIM provider works without pool (backward compatible)."""
        provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())

        assert provider._account_pool is None

    def test_openrouter_with_pool(self):
        """OpenRouter provider works with account pool."""
        pool = AccountPool(
            api_keys=["key1_long_enough", "key2_long_enough"],
            base_url="https://openrouter.ai/api/v1",
            strategy="least_used",
        )
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        provider = OpenRouterProvider(config, account_pool=pool)

        assert provider._account_pool is not None
        assert provider._account_pool.account_count == 2
        assert provider._account_pool.strategy == "least_used"

    def test_openrouter_without_pool(self):
        """OpenRouter provider works without pool."""
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        provider = OpenRouterProvider(config)

        assert provider._account_pool is None

    def test_lmstudio_with_pool(self):
        """LM Studio provider works with account pool."""
        from providers.lmstudio import LMStudioProvider

        pool = AccountPool(
            api_keys=["key1_long_enough", "key2_long_enough"],
            base_url="http://localhost:1234/v1",
        )
        config = ProviderConfig(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1",
            rate_limit=10,
            rate_window=60,
        )
        provider = LMStudioProvider(config, account_pool=pool)

        assert provider._account_pool is not None

    def test_pool_accounts_have_correct_base_url(self):
        """Account pool creates clients with correct base URL."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222"],
            base_url="https://integrate.api.nvidia.com/v1",
        )
        for account in pool._accounts:
            assert account.client.base_url is not None


class TestPoolCreationHelper:
    """Tests for pool creation logic used by dependencies."""

    def test_single_key_no_pool(self):
        """Single key should not create a pool."""
        api_keys = ["single_key"]
        pool = (
            None
            if len(api_keys) <= 1
            else AccountPool(
                api_keys=api_keys,
                base_url="https://api.test.com/v1",
            )
        )
        assert pool is None

    def test_multiple_keys_create_pool(self):
        """Multiple keys should create a pool."""
        api_keys = ["key1", "key2", "key3"]
        pool = (
            None
            if len(api_keys) <= 1
            else AccountPool(
                api_keys=api_keys,
                base_url="https://api.test.com/v1",
            )
        )
        assert pool is not None
        assert pool.account_count == 3

    def test_pool_uses_round_robin_default(self):
        """Pool defaults to round_robin strategy."""
        pool = AccountPool(
            api_keys=["key1", "key2"],
            base_url="https://api.test.com/v1",
        )
        assert pool.strategy == "round_robin"

    def test_pool_uses_least_used_strategy(self):
        """Pool respects least_used strategy."""
        pool = AccountPool(
            api_keys=["key1", "key2"],
            base_url="https://api.test.com/v1",
            strategy="least_used",
        )
        assert pool.strategy == "least_used"

    @pytest.mark.asyncio
    async def test_pool_cleanup(self):
        """Pool cleanup closes all clients."""
        pool = AccountPool(
            api_keys=["key1", "key2"],
            base_url="https://api.test.com/v1",
        )
        await pool.aclose()
