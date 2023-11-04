import asyncio
from typing import TypeVar
from collections.abc import Callable, Awaitable
import functools


T = TypeVar("T")


def run_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Decorator to run a synchronous function in a separate thread, making it awaitable."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper
