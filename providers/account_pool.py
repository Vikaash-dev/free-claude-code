"""Account pool for multi-account load balancing across API keys."""

import asyncio
import random
import time
from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
import openai
from loguru import logger
from openai import AsyncOpenAI

T = TypeVar("T")


class AccountRateLimiter:
    """Per-account rate limiter with proactive throttling and reactive blocking.

    Unlike GlobalRateLimiter, this is not a singleton — each account gets its own
    instance so rate limits are tracked independently.
    """

    def __init__(self, rate_limit: int, rate_window: float):
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")

        self._rate_limit = rate_limit
        self._rate_window = float(rate_window)
        self._request_times: deque[float] = deque()
        self._blocked_until: float = 0
        self._lock = asyncio.Lock()

    async def wait_if_blocked(self) -> bool:
        """Wait if currently rate limited or throttle to meet quota."""
        waited_reactively = False
        now = time.monotonic()
        if now < self._blocked_until:
            wait_time = self._blocked_until - now
            await asyncio.sleep(wait_time)
            waited_reactively = True

        await self._acquire_proactive_slot()
        return waited_reactively

    async def _acquire_proactive_slot(self) -> None:
        """Acquire a proactive slot enforcing a strict rolling window."""
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._rate_window

                while self._request_times and self._request_times[0] <= cutoff:
                    self._request_times.popleft()

                if len(self._request_times) < self._rate_limit:
                    self._request_times.append(now)
                    return

                oldest = self._request_times[0]
                wait_time = max(0.0, (oldest + self._rate_window) - now)

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    def set_blocked(self, seconds: float) -> None:
        """Set block for specified seconds (reactive)."""
        self._blocked_until = time.monotonic() + seconds

    def is_blocked(self) -> bool:
        """Check if currently reactively blocked."""
        return time.monotonic() < self._blocked_until

    def remaining_wait(self) -> float:
        """Get remaining reactive wait time in seconds."""
        return max(0.0, self._blocked_until - time.monotonic())


class Account:
    """Represents a single API account with its own client and rate limiter."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        rate_limit: int,
        rate_window: float,
        *,
        http_read_timeout: float = 300.0,
        http_write_timeout: float = 10.0,
        http_connect_timeout: float = 2.0,
    ):
        self.api_key = api_key
        self.rate_limiter = AccountRateLimiter(rate_limit, rate_window)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=httpx.Timeout(
                http_read_timeout,
                connect=http_connect_timeout,
                read=http_read_timeout,
                write=http_write_timeout,
            ),
        )
        self.request_count: int = 0
        self.last_used: float = 0.0
        # Masked key for logging (show first 4 and last 4 chars)
        if len(api_key) > 8:
            self.masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        else:
            self.masked_key = "***"

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.close()


class AccountPool:
    """Manages multiple API accounts with load balancing strategies.

    Inspired by antigravity-claude-proxy's multi-account load balancing:
    - round_robin: Cycles through accounts sequentially
    - least_used: Picks the account with the fewest recent requests

    When an account is rate-limited, it is skipped and the next available
    account is used automatically (failover).
    """

    def __init__(
        self,
        api_keys: list[str],
        base_url: str,
        strategy: str = "round_robin",
        rate_limit: int = 40,
        rate_window: float = 60.0,
        *,
        http_read_timeout: float = 300.0,
        http_write_timeout: float = 10.0,
        http_connect_timeout: float = 2.0,
    ):
        if not api_keys:
            raise ValueError("At least one API key is required")

        unique_keys = list(dict.fromkeys(api_keys))
        if len(unique_keys) < len(api_keys):
            logger.warning(
                "Duplicate API keys removed: %d -> %d",
                len(api_keys),
                len(unique_keys),
            )

        self._accounts = [
            Account(
                api_key=key,
                base_url=base_url,
                rate_limit=rate_limit,
                rate_window=rate_window,
                http_read_timeout=http_read_timeout,
                http_write_timeout=http_write_timeout,
                http_connect_timeout=http_connect_timeout,
            )
            for key in unique_keys
        ]
        self._strategy = strategy
        self._round_robin_index = 0
        self._lock = asyncio.Lock()

        logger.info(
            "AccountPool initialized: %d account(s), strategy=%s, "
            "rate_limit=%d/%ds per account",
            len(self._accounts),
            strategy,
            rate_limit,
            rate_window,
        )

    @property
    def account_count(self) -> int:
        """Number of accounts in the pool."""
        return len(self._accounts)

    @property
    def strategy(self) -> str:
        """Current selection strategy."""
        return self._strategy

    async def get_account(self) -> Account:
        """Select the next available account based on strategy.

        Skips accounts that are currently rate-limited.
        If all accounts are blocked, waits on the one with the shortest wait.
        """
        async with self._lock:
            available = [a for a in self._accounts if not a.rate_limiter.is_blocked()]

            if not available:
                # All blocked — pick the one with the shortest remaining wait
                best = min(
                    self._accounts, key=lambda a: a.rate_limiter.remaining_wait()
                )
                logger.warning(
                    "All accounts rate-limited, waiting on account %s (%.1fs)",
                    best.masked_key,
                    best.rate_limiter.remaining_wait(),
                )
                return best

            if self._strategy == "least_used":
                account = min(available, key=lambda a: a.request_count)
            else:
                # round_robin (default)
                self._round_robin_index = self._round_robin_index % len(self._accounts)
                # Find next non-blocked account starting from current index
                account = None
                for i in range(len(self._accounts)):
                    idx = (self._round_robin_index + i) % len(self._accounts)
                    if not self._accounts[idx].rate_limiter.is_blocked():
                        account = self._accounts[idx]
                        self._round_robin_index = idx + 1
                        break
                if account is None:
                    account = available[0]

            account.request_count += 1
            account.last_used = time.monotonic()
            return account

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable with multi-account retry on 429.

        On rate limit errors, marks the current account as blocked and
        tries the next available account in the pool.
        """
        last_exc: Exception | None = None

        for attempt in range(1 + max_retries):
            account = await self.get_account()
            await account.rate_limiter.wait_if_blocked()

            try:
                return await fn(account.client, *args, **kwargs)
            except openai.RateLimitError as e:
                last_exc = e
                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, jitter)

                logger.warning(
                    "Account %s rate limited (429), attempt %d/%d. "
                    "Blocking for %.1fs and trying next account...",
                    account.masked_key,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                account.rate_limiter.set_blocked(delay)

                if attempt >= max_retries:
                    break

        assert last_exc is not None
        raise last_exc

    def get_stats(self) -> list[dict[str, Any]]:
        """Get status of all accounts for monitoring."""
        return [
            {
                "key": account.masked_key,
                "request_count": account.request_count,
                "is_blocked": account.rate_limiter.is_blocked(),
                "remaining_wait": round(account.rate_limiter.remaining_wait(), 1),
            }
            for account in self._accounts
        ]

    async def aclose(self) -> None:
        """Close all account clients."""
        for account in self._accounts:
            await account.aclose()
