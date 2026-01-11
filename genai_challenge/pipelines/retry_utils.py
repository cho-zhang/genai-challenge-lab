import asyncio

import aiohttp
import litellm


def is_retryable_exception(exception: BaseException) -> bool:
    """Determine if an error should be retried.

    Retries:
    - Timeouts
    - Rate limiting (429)
    - Server errors (5xx)
    - Network/connection errors

    Do not retry:
    - 4xx client errors (except 429)
    """
    if isinstance(exception, asyncio.TimeoutError):
        return True
    elif isinstance(exception, aiohttp.ClientResponseError):
        return (
            exception.status == 429
            or exception.status >= 500
            or isinstance(exception, aiohttp.ContentTypeError)
        )
    elif isinstance(exception, litellm.exceptions.RateLimitError):
        return True
    elif isinstance(exception, litellm.exceptions.InternalServerError):
        return True
    return isinstance(exception, aiohttp.ClientError)
