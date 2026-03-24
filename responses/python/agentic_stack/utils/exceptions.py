from functools import partial, wraps
from inspect import iscoroutinefunction
from typing import Any, Callable, TypeVar, overload

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from loguru import logger
from sqlalchemy.exc import IntegrityError

from agentic_stack.entrypoints._state import get_request_id


def docstring_message(cls):
    """
    Decorates an exception to make its docstring its default message.
    https://stackoverflow.com/a/66491013
    """
    # Must use cls_init name, not cls.__init__ itself, in closure to avoid recursion
    cls_init = cls.__init__

    @wraps(cls.__init__)
    def wrapped_init(self, msg=cls.__doc__, *args, **kwargs):
        cls_init(self, msg, *args, **kwargs)

    cls.__init__ = wrapped_init
    return cls


@docstring_message
class VRException(Exception):
    """Base exception class for errors."""


@docstring_message
class AuthorizationError(VRException):
    """You do not have the correct credentials."""


@docstring_message
class ExternalAuthError(VRException):
    """Authentication with external provider failed."""


@docstring_message
class ForbiddenError(VRException):
    """You do not have access to this resource."""


@docstring_message
class UpgradeTierError(VRException):
    """You have exhausted the allocations of your subscribed tier. Please upgrade."""


@docstring_message
class BudgetExceededError(VRException):
    """You have reached your budget limit. Wait for the next cycle or ask an administrator to increase it."""


@docstring_message
class InsufficientCreditsError(VRException):
    """Please ensure that you have sufficient credits."""


@docstring_message
class ResourceNotFoundError(VRException):
    """Resource with the specified name is not found."""


@docstring_message
class MethodNotAllowedError(VRException):
    """Method is not allowed."""


@docstring_message
class ResourceExistsError(VRException):
    """Resource with the specified name already exists."""


@docstring_message
class UnsupportedMediaTypeError(VRException):
    """This file type is unsupported."""


@docstring_message
class BadInputError(VRException):
    """Your input is invalid."""


class ResponsesAPIError(VRException):
    """Responses/OpenAI-style API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_type = str(error_type)
        self.param = param
        self.code = code


@docstring_message
class ModelCapabilityError(BadInputError):
    """No model has the specified capabilities."""


@docstring_message
class ContextOverflowError(VRException):
    """Model's context length has been exceeded."""


@docstring_message
class UnexpectedError(VRException):
    """We ran into an unexpected error."""


@docstring_message
class RateLimitExceedError(VRException):
    """The rate limit is exceeded."""

    def __init__(
        self,
        *args,
        limit: int,
        remaining: int,
        reset_at: int,
        used: int | None = None,
        retry_after: int | None = None,
        meta: dict[str, Any] | None = None,
    ):
        super().__init__(*args)
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
        self.used = used
        self.retry_after = retry_after
        self.meta = meta


@docstring_message
class UnavailableError(VRException):
    """The requested functionality is unavailable."""


@docstring_message
class ServerBusyError(VRException):
    """The server is busy."""


@docstring_message
class ModelOverloadError(VRException):
    """The model is overloaded."""


F = TypeVar("F", bound=Callable[..., Any])


@overload
def handle_exception(
    func: F,
    *,
    handler: Callable[..., Any] | None = None,
) -> F: ...


@overload
def handle_exception(
    *,
    handler: Callable[..., Any] | None = None,
) -> Callable[[F], F]: ...


def handle_exception(
    func: F | None = None,
    *,
    handler: Callable[..., Any] | None = None,
) -> Callable[[F], F] | F:
    """
    A decorator to handle exceptions for both synchronous and asynchronous functions.
    Its main purpose is to:
    - Produce shorter traceback (160 vs 500 lines) upon unexpected errors (such as `ValueError`).
    - Transform certain error classes, for example `IntegrityError` -> `ResourceExistsError`.

    It also allows you to specify a custom exception handler function.
    The handler function should accept a single positional argument (the exception instance)
    and all keyword arguments passed to the decorated function.

    Note that if a handler is provided, you are responsible to re-raise the exception if desired.

    Args:
        func (F | None): The function to be decorated. This can be either a synchronous or
            asynchronous function. When used as a decorator, leave this unset. Defaults to `None`.
        handler (Callable[..., None] | None): A custom exception handler function.
            The handler function should accept a positional argument (the exception instance)
            followed by all arguments passed to the decorated function.

    Returns:
        func (Callable[[F], F] | F): The decorated function with exception handling applied.

    Raises:
        VRException: If `VRException` is raised.
        RequestValidationError: If `fastapi.exceptions.RequestValidationError` is raised.
        ResourceExistsError: If `sqlalchemy.exc.IntegrityError` indicates a unique constraint violation in the database.
        UnexpectedError: For all other exception.
    """

    def _default_handler(e: Exception, *args, **kwargs):
        if isinstance(e, VRException):
            raise
        elif isinstance(e, RequestValidationError):
            raise
        # elif isinstance(e, ValidationError):
        #     raise RequestValidationError(errors=e.errors()) from e
        elif isinstance(e, IntegrityError):
            err_mssg: str = e.args[0]
            err_mssgs = err_mssg.split("UNIQUE constraint failed:")
            if len(err_mssgs) > 1:
                constraint = err_mssgs[1].strip()
                raise ResourceExistsError(f'DB item "{constraint}" already exists.') from e
            else:
                raise UnexpectedError(f"{e.__class__.__name__}: {e}") from e
        else:
            request: Request | None = kwargs.get("request", None)
            mssg = f"Failed to run {func.__name__}"
            mssg = f"{e.__class__.__name__}: {e} - {mssg} - kwargs={kwargs}"
            if request:
                request_id = get_request_id(request)
                logger.error(f"{request_id} - {mssg}")
            else:
                logger.error(mssg)
            raise UnexpectedError(f"{e.__class__.__name__}: {e}") from e

    if handler is None:
        handler = _default_handler

    if iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return handler(e, *args, **kwargs)

    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler(e, *args, **kwargs)

    return partial(handle_exception, handler=handler) if func is None else wrapper
