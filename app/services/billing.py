import asyncio
import structlog

logger = structlog.get_logger()


billing_queue = asyncio.Queue()


async def billing_worker():
    while True:
        billing_data = await billing_queue.get()
        logger.info("billing", **billing_data)
        billing_queue.task_done()


async def drain_billing_queue():
    # there's potential for data loss if the server is shut
    # down in a non-graceful way.
    print("draining billing")
    while not billing_queue.empty():
        billing_data = await billing_queue.get()
        logger.info("billing", **billing_data)
        billing_queue.task_done()