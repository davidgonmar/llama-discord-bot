import asyncio
import typing
import functools


T = typing.TypeVar("T")

def run_async(func: typing.Callable[..., T]) -> typing.Callable[..., typing.Awaitable[T]]:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper