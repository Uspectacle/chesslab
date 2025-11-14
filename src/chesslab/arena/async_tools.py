import asyncio
import threading
from typing import Any, Coroutine


def run_async_background(coro: Coroutine[Any, Any, Any]):
    try:
        # If there is already an event loop running, use it
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # No event loop running → make one in a new thread
        def runner():
            asyncio.run(coro)

        threading.Thread(target=runner, daemon=True).start()
