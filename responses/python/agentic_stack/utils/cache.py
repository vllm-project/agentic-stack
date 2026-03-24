from contextlib import asynccontextmanager, suppress
from typing import AsyncGenerator

import orjson
from pottery import AIORedlock, ReleaseUnlockedLock
from redis import Redis
from redis.asyncio import Redis as RedisAsync
from redis.backoff import EqualJitterBackoff
from redis.exceptions import ConnectionError, TimeoutError
from redis.retry import Retry


class Cache:
    def __init__(
        self,
        *,
        redis_url: str,
        cache_expiration: int = 5 * 60,  # 5 minutes
    ):
        self._redis_kwargs = dict(
            url=redis_url,
            # https://redis.io/kb/doc/22wxq63j93/how-to-manage-client-reconnections-in-case-of-errors-with-redis-py
            retry=Retry(EqualJitterBackoff(cap=10, base=1), 5),
            retry_on_error=[ConnectionError, TimeoutError, ConnectionResetError],
            health_check_interval=15,
            decode_responses=True,
        )
        self._redis = Redis.from_url(**self._redis_kwargs)
        self._redis_async = RedisAsync.from_url(**self._redis_kwargs)
        self.cache_expiration = int(cache_expiration)
        # try:
        #     self._redis.ping()
        # except ConnectionError as e:
        #     logger.error(f"Failed to connect to Redis: {repr(e)}")
        #     raise

    def __getitem__(self, key: str) -> str | None:
        """
        Getter method.
        ```
        cache = Cache(...)
        value = cache["key"]
        ```

        Args:
            key (str): Key.

        Returns:
            value (str | None): Value.
        """
        return self._redis.get(key)

    def __setitem__(self, key: str, value: str) -> None:
        """
        Setter method.
        ```
        cache = Cache(...)
        cache["key"] = value
        ```

        Args:
            key (str): Key.
            value (str): Value.
        """
        if not isinstance(value, str):
            raise TypeError(f"`value` must be a str, received: {type(value)}")
        self._redis.set(key, value)

    def __delitem__(self, key) -> None:
        """
        Delete method.
        ```
        cache = Cache(...)
        del cache["key"]
        ```

        Args:
            key (str): Key.
        """
        self._redis.delete(key)

    def __contains__(self, key) -> bool:
        self._redis.exists(key)

    def purge(self):
        self._redis.flushdb()

    async def aclose(self):
        self._redis.close()
        await self._redis_async.aclose()

    async def get(self, key: str) -> str | None:
        return await self._redis_async.get(key)

    async def set(self, key: str, value: str, **kwargs) -> None:
        if not isinstance(value, str):
            raise TypeError(f"`value` must be a str, received: {type(value)}")
        await self._redis_async.set(key, value, **kwargs)

    async def get_json(self, key: str) -> object | None:
        value = await self.get(key)
        if value is None:
            return None
        return orjson.loads(value)

    async def set_json(self, key: str, value: object, **kwargs) -> None:
        await self.set(key, orjson.dumps(value).decode("utf-8"), **kwargs)

    async def delete(self, key: str) -> None:
        await self._redis_async.delete(key)

    async def exists(self, *keys: str) -> int:
        return await self._redis_async.exists(*keys)

    @asynccontextmanager
    async def alock(
        self,
        key: str,
        blocking: bool = True,
        expire: float = 60.0,
    ) -> AsyncGenerator[bool, None]:
        lock = AIORedlock(
            key=key,
            masters={self._redis_async},
            auto_release_time=max(1.0, expire),
        )
        lock_acquired = await lock.acquire(blocking=blocking)
        try:
            yield lock_acquired
        finally:
            if lock_acquired:
                with suppress(ReleaseUnlockedLock):
                    await lock.release()

    async def clear_all_async(self) -> None:
        pass
