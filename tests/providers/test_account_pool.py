"""Tests for providers.account_pool multi-account load balancing."""

import asyncio
import time

import pytest

from providers.account_pool import Account, AccountPool, AccountRateLimiter


class TestAccountRateLimiter:
    """Tests for per-account rate limiter."""

    @pytest.mark.asyncio
    async def test_proactive_throttling(self):
        """Per-account rate limiter throttles within rolling window."""
        limiter = AccountRateLimiter(rate_limit=1, rate_window=0.25)

        start = time.time()
        for _ in range(3):
            await limiter.wait_if_blocked()
        duration = time.time() - start

        # 3 requests at 1/0.25s should take ~0.5s
        assert duration >= 0.4

    @pytest.mark.asyncio
    async def test_reactive_blocking(self):
        """set_blocked pauses requests for the specified duration."""
        limiter = AccountRateLimiter(rate_limit=100, rate_window=60)
        limiter.set_blocked(0.5)

        assert limiter.is_blocked()

        start = time.time()
        await limiter.wait_if_blocked()
        duration = time.time() - start

        assert duration >= 0.4

    @pytest.mark.asyncio
    async def test_remaining_wait(self):
        """remaining_wait returns correct value."""
        limiter = AccountRateLimiter(rate_limit=100, rate_window=60)
        assert limiter.remaining_wait() == 0

        limiter.set_blocked(2.0)
        assert limiter.remaining_wait() > 1.5

    @pytest.mark.asyncio
    async def test_not_blocked_initially(self):
        """Fresh limiter is not blocked."""
        limiter = AccountRateLimiter(rate_limit=100, rate_window=60)
        assert limiter.is_blocked() is False

    def test_invalid_rate_limit(self):
        """rate_limit <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="rate_limit must be > 0"):
            AccountRateLimiter(rate_limit=0, rate_window=60)

    def test_invalid_rate_window(self):
        """rate_window <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="rate_window must be > 0"):
            AccountRateLimiter(rate_limit=10, rate_window=0)


class TestAccount:
    """Tests for Account class."""

    def test_account_creation(self):
        """Account initializes with correct attributes."""
        account = Account(
            api_key="test_key_123456",
            base_url="https://api.test.com/v1",
            rate_limit=40,
            rate_window=60,
        )
        assert account.api_key == "test_key_123456"
        assert account.request_count == 0
        assert account.last_used == 0.0
        assert account.masked_key == "test...3456"

    def test_short_key_masked(self):
        """Short API keys are fully masked."""
        account = Account(
            api_key="short",
            base_url="https://api.test.com/v1",
            rate_limit=40,
            rate_window=60,
        )
        assert account.masked_key == "***"

    @pytest.mark.asyncio
    async def test_account_aclose(self):
        """Account can close its client."""
        account = Account(
            api_key="test_key",
            base_url="https://api.test.com/v1",
            rate_limit=40,
            rate_window=60,
        )
        await account.aclose()


class TestAccountPool:
    """Tests for AccountPool multi-account load balancing."""

    def test_pool_creation(self):
        """Pool initializes with correct account count."""
        pool = AccountPool(
            api_keys=["key1_longkey", "key2_longkey", "key3_longkey"],
            base_url="https://api.test.com/v1",
        )
        assert pool.account_count == 3
        assert pool.strategy == "round_robin"

    def test_pool_empty_keys_raises(self):
        """Empty key list raises ValueError."""
        with pytest.raises(ValueError, match="At least one API key"):
            AccountPool(api_keys=[], base_url="https://api.test.com/v1")

    def test_pool_deduplicates_keys(self):
        """Duplicate keys are removed."""
        pool = AccountPool(
            api_keys=["key1_longkey", "key2_longkey", "key1_longkey"],
            base_url="https://api.test.com/v1",
        )
        assert pool.account_count == 2

    @pytest.mark.asyncio
    async def test_round_robin_selection(self):
        """Round-robin cycles through accounts."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222", "key_cccc_3333"],
            base_url="https://api.test.com/v1",
            strategy="round_robin",
        )

        keys_used = []
        for _ in range(6):
            account = await pool.get_account()
            keys_used.append(account.api_key)

        # Should cycle: key1, key2, key3, key1, key2, key3
        assert keys_used[0] == "key_aaaa_1111"
        assert keys_used[1] == "key_bbbb_2222"
        assert keys_used[2] == "key_cccc_3333"
        assert keys_used[3] == "key_aaaa_1111"
        assert keys_used[4] == "key_bbbb_2222"
        assert keys_used[5] == "key_cccc_3333"

    @pytest.mark.asyncio
    async def test_least_used_selection(self):
        """Least-used picks account with fewest requests."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222", "key_cccc_3333"],
            base_url="https://api.test.com/v1",
            strategy="least_used",
        )

        # All start at 0, so first pick can be any (min picks first found)
        a1 = await pool.get_account()
        assert a1.request_count == 1

        a2 = await pool.get_account()
        # Should pick one of the others (count=0)
        assert a2.api_key != a1.api_key
        assert a2.request_count == 1

        a3 = await pool.get_account()
        # Should pick the remaining one (count=0)
        assert a3.request_count == 1
        used_keys = {a1.api_key, a2.api_key, a3.api_key}
        assert len(used_keys) == 3

    @pytest.mark.asyncio
    async def test_skip_blocked_account(self):
        """Blocked accounts are skipped in round-robin."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222", "key_cccc_3333"],
            base_url="https://api.test.com/v1",
            strategy="round_robin",
        )

        # Block the first account
        pool._accounts[0].rate_limiter.set_blocked(10.0)

        account = await pool.get_account()
        assert account.api_key == "key_bbbb_2222"

        account = await pool.get_account()
        assert account.api_key == "key_cccc_3333"

    @pytest.mark.asyncio
    async def test_all_blocked_picks_shortest_wait(self):
        """When all accounts blocked, picks the one with shortest wait."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222"],
            base_url="https://api.test.com/v1",
        )

        pool._accounts[0].rate_limiter.set_blocked(10.0)
        pool._accounts[1].rate_limiter.set_blocked(5.0)

        account = await pool.get_account()
        # Should pick key_bbbb_2222 (shorter wait)
        assert account.api_key == "key_bbbb_2222"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """get_stats returns correct account information."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222"],
            base_url="https://api.test.com/v1",
        )

        await pool.get_account()

        stats = pool.get_stats()
        assert len(stats) == 2
        assert stats[0]["key"] == "key_...1111"
        assert stats[0]["request_count"] == 1
        assert stats[0]["is_blocked"] is False
        assert stats[1]["request_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """execute_with_retry calls function with account's client."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111"],
            base_url="https://api.test.com/v1",
        )

        async def mock_fn(client, **kwargs):
            return "result"

        result = await pool.execute_with_retry(mock_fn, max_retries=1)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_with_retry_failover(self):
        """On 429, execute_with_retry tries next account."""
        import openai
        from httpx import Request, Response

        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222"],
            base_url="https://api.test.com/v1",
            strategy="round_robin",
        )

        call_count = 0
        clients_seen: list[object] = []

        async def mock_fn(client, **kwargs):
            nonlocal call_count
            clients_seen.append(client)
            call_count += 1
            if call_count == 1:
                raise openai.RateLimitError(
                    "rate limited",
                    response=Response(429, request=Request("POST", "http://x")),
                    body={},
                )
            return "ok"

        result = await pool.execute_with_retry(
            mock_fn, max_retries=2, base_delay=0.01, max_delay=0.1, jitter=0
        )
        assert result == "ok"
        assert call_count == 2
        # Should have used different clients (different accounts)
        assert len(clients_seen) == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """When all retries exhausted, raises last exception."""
        import openai
        from httpx import Request, Response

        pool = AccountPool(
            api_keys=["key_aaaa_1111"],
            base_url="https://api.test.com/v1",
        )

        async def fail(client, **kwargs):
            raise openai.RateLimitError(
                "rate limited",
                response=Response(429, request=Request("POST", "http://x")),
                body={},
            )

        with pytest.raises(openai.RateLimitError):
            await pool.execute_with_retry(
                fail, max_retries=1, base_delay=0.01, max_delay=0.1, jitter=0
            )

    @pytest.mark.asyncio
    async def test_single_key_pool(self):
        """Pool with single key works correctly."""
        pool = AccountPool(
            api_keys=["single_key_long"],
            base_url="https://api.test.com/v1",
        )
        assert pool.account_count == 1

        account = await pool.get_account()
        assert account.api_key == "single_key_long"

    @pytest.mark.asyncio
    async def test_aclose_all(self):
        """aclose closes all account clients."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222"],
            base_url="https://api.test.com/v1",
        )
        await pool.aclose()

    @pytest.mark.asyncio
    async def test_concurrent_get_account(self):
        """Concurrent get_account calls are safe."""
        pool = AccountPool(
            api_keys=["key_aaaa_1111", "key_bbbb_2222", "key_cccc_3333"],
            base_url="https://api.test.com/v1",
            strategy="round_robin",
        )

        accounts = await asyncio.gather(*(pool.get_account() for _ in range(9)))

        # All 9 should succeed
        assert len(accounts) == 9
        # Each key should be used 3 times (round-robin)
        key_counts: dict[str, int] = {}
        for a in accounts:
            key_counts[a.api_key] = key_counts.get(a.api_key, 0) + 1
        assert key_counts["key_aaaa_1111"] == 3
        assert key_counts["key_bbbb_2222"] == 3
        assert key_counts["key_cccc_3333"] == 3
