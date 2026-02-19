"""Dependency injection for FastAPI."""

from fastapi import HTTPException
from loguru import logger

from config.settings import NVIDIA_NIM_BASE_URL, Settings
from config.settings import get_settings as _get_settings
from providers.account_pool import AccountPool
from providers.base import BaseProvider, ProviderConfig

# Global provider instance (singleton)
_provider: BaseProvider | None = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def _parse_api_keys(keys_str: str, single_key: str) -> list[str]:
    """Parse comma-separated API keys, falling back to single key.

    Returns a list of non-empty, stripped API keys.
    """
    keys: list[str] = []
    if keys_str and keys_str.strip():
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not keys and single_key and single_key.strip():
        keys = [single_key.strip()]
    return keys


def _create_account_pool(
    api_keys: list[str],
    base_url: str,
    settings: Settings,
) -> AccountPool | None:
    """Create an AccountPool if multiple API keys are provided."""
    if len(api_keys) <= 1:
        return None
    return AccountPool(
        api_keys=api_keys,
        base_url=base_url,
        strategy=settings.account_selection_strategy,
        rate_limit=settings.provider_rate_limit,
        rate_window=float(settings.provider_rate_window),
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
    )


def get_provider() -> BaseProvider:
    """Get or create the provider instance based on settings.provider_type."""
    global _provider
    if _provider is None:
        settings = get_settings()

        if settings.provider_type == "nvidia_nim":
            api_keys = _parse_api_keys(
                settings.nvidia_nim_api_keys, settings.nvidia_nim_api_key
            )
            if not api_keys:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "NVIDIA_NIM_API_KEY is not set. Add it to your .env file. "
                        "Get a key at https://build.nvidia.com/settings/api-keys"
                    ),
                )
            from providers.nvidia_nim import NvidiaNimProvider

            pool = _create_account_pool(api_keys, NVIDIA_NIM_BASE_URL, settings)
            config = ProviderConfig(
                api_key=api_keys[0],
                base_url=NVIDIA_NIM_BASE_URL,
                rate_limit=settings.provider_rate_limit,
                rate_window=settings.provider_rate_window,
                http_read_timeout=settings.http_read_timeout,
                http_write_timeout=settings.http_write_timeout,
                http_connect_timeout=settings.http_connect_timeout,
            )
            _provider = NvidiaNimProvider(
                config, nim_settings=settings.nim, account_pool=pool
            )
            key_info = (
                f"{len(api_keys)} account(s)" if len(api_keys) > 1 else "1 account"
            )
            logger.info(
                "Provider initialized: %s (%s)", settings.provider_type, key_info
            )
        elif settings.provider_type == "open_router":
            api_keys = _parse_api_keys(
                settings.open_router_api_keys, settings.open_router_api_key
            )
            if not api_keys:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "OPENROUTER_API_KEY is not set. Add it to your .env file. "
                        "Get a key at https://openrouter.ai/keys"
                    ),
                )
            from providers.open_router import OpenRouterProvider

            base_url = "https://openrouter.ai/api/v1"
            pool = _create_account_pool(api_keys, base_url, settings)
            config = ProviderConfig(
                api_key=api_keys[0],
                base_url=base_url,
                rate_limit=settings.provider_rate_limit,
                rate_window=settings.provider_rate_window,
                http_read_timeout=settings.http_read_timeout,
                http_write_timeout=settings.http_write_timeout,
                http_connect_timeout=settings.http_connect_timeout,
            )
            _provider = OpenRouterProvider(config, account_pool=pool)
            key_info = (
                f"{len(api_keys)} account(s)" if len(api_keys) > 1 else "1 account"
            )
            logger.info(
                "Provider initialized: %s (%s)", settings.provider_type, key_info
            )
        elif settings.provider_type == "lmstudio":
            from providers.lmstudio import LMStudioProvider

            config = ProviderConfig(
                api_key="lm-studio",
                base_url=settings.lm_studio_base_url,
                rate_limit=settings.provider_rate_limit,
                rate_window=settings.provider_rate_window,
                http_read_timeout=settings.http_read_timeout,
                http_write_timeout=settings.http_write_timeout,
                http_connect_timeout=settings.http_connect_timeout,
            )
            _provider = LMStudioProvider(config)
            logger.info("Provider initialized: %s", settings.provider_type)
        else:
            logger.error(
                "Unknown provider_type: '%s'. Supported: 'nvidia_nim', 'open_router', 'lmstudio'",
                settings.provider_type,
            )
            raise ValueError(
                f"Unknown provider_type: '{settings.provider_type}'. "
                f"Supported: 'nvidia_nim', 'open_router', 'lmstudio'"
            )
    return _provider


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider:
        pool = getattr(_provider, "_account_pool", None)
        if pool is not None:
            await pool.aclose()
        else:
            client = getattr(_provider, "_client", None)
            if client and hasattr(client, "aclose"):
                await client.aclose()
    _provider = None
    logger.debug("Provider cleanup completed")
