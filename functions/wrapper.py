from inspect import signature, Signature
from abc import abstractmethod
from typing import get_args, Union, TypeVar, Type, Generic, LiteralString, Callable, Optional, Awaitable
from pydantic.v1 import BaseModel
from langchain_community.tools import BaseTool, StructuredTool
from langchain.tools.base import ToolException
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.schema.runnable import Runnable, RunnableConfig, RunnableLambda
from functools import wraps


Input = TypeVar("Input", bound=BaseModel, contravariant=True)
Output = TypeVar("Output", bound=BaseModel, covariant=True)

OrigFunc = Callable[..., Output]
AsyncOrigFunc = Callable[..., Awaitable[Output]]

# tool function returns a json string dumped from Output
ToolFunc = Callable[..., str]
AsyncToolFunc = Callable[..., Awaitable[str]]

ToolCallback = Callable[[], Union[BaseModel, Awaitable[BaseModel]]]

RunnableFunc = Union[
    Callable[[Input], Output],
    Callable[[Input, RunnableConfig], Output],
    Callable[[Input, CallbackManagerForChainRun], Output],
    Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
]
AsyncRunnableFunc = Union[
    Callable[[Input], Awaitable[Output]],
    Callable[[Input, RunnableConfig], Awaitable[Output]],
    Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
    Callable[
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
        Awaitable[Output],
    ],
]


class FunctionWrapper(Generic[Input, Output]):
    """A wrapper which wraps a function to either a runnable or a tool."""

    return_direct: bool = False

    def __init__(self):
        if self.func:
            self._validate_func(self.func)
        if self.async_func:
            self._validate_func(self.async_func)
        if not self.func and not self.async_func:
            raise NotImplementedError("must define at least one process function")

    @classmethod
    def _validate_func(cls, func: Callable):
        input_fields = {
            name: (field.annotation, field.default)
            for name, field in cls.input_type().__fields__.items()
        }
        func_params = {
            name: (param.annotation, param.default if param.default is not Signature.empty else None)
            for name, param in signature(func).parameters.items() if name is not "callbacks"
        }
        if input_fields != func_params:
            raise TypeError(f"input fields {input_fields} and function parameters {func_params} don't match")

    @classmethod
    @abstractmethod
    def name(cls) -> LiteralString:
        raise NotImplementedError("must define a name")

    @classmethod
    @abstractmethod
    def description(cls) -> LiteralString:
        raise NotImplementedError("must define a description")

    @classmethod
    @abstractmethod
    def notification(cls) -> str:
        raise NotImplementedError("must define a notification")

    @classmethod
    def input_type(cls) -> Type[Input]:
        """The type of input this runnable accepts specified as a type annotation."""
        for c in cls.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(c)
            if type_args and len(type_args) == 2:
                tp = type_args[0]
                if not issubclass(tp, BaseModel):
                    raise TypeError(f"input type {tp} must be a subclass of BaseModel")
                return tp
        raise TypeError(f"{cls.__name__} doesn't have an inferable input type")

    @classmethod
    def output_type(cls) -> Type[Output]:
        """The type of output this runnable accepts specified as a type annotation."""
        for c in cls.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(c)
            if type_args and len(type_args) == 2:
                return type_args[1]
        raise TypeError(f"{cls.__name__} doesn't have an inferable output type")

    @property
    def func(self) -> Optional[OrigFunc]:
        return None

    @property
    def async_func(self) -> Optional[AsyncOrigFunc]:
        return None

    @property
    def tool_func(self) -> Optional[ToolFunc]:
        if self.func:
            @wraps(self.func)
            def _tool_func(*args, **kwargs) -> str:
                assert self.func is not None
                try:
                    output = self.func(*args, **kwargs)
                    return output.json()
                except Exception as e:
                    raise ToolException(e)

            return _tool_func
        else:
            return None

    @property
    def async_tool_func(self) -> Optional[AsyncToolFunc]:
        if self.async_func:
            @wraps(self.async_func)
            async def _async_tool_func(*args, **kwargs) -> str:
                assert self.async_func is not None
                try:
                    output = await self.async_func(*args, **kwargs)
                    return output.json()
                except Exception as e:
                    raise ToolException(e)

            return _async_tool_func
        else:
            return None

    @property
    def tool_callback(self) -> Optional[ToolCallback]:
        return None

    @property
    def runnable_func(self) -> Optional[RunnableFunc]:
        if self.func:
            def _runnable_func(_input: Input) -> Output:
                assert self.func is not None
                return self.func(**_input.dict())

            return _runnable_func
        else:
            return None

    @property
    def async_runnable_func(self) -> Optional[AsyncRunnableFunc]:
        if self.async_func:
            async def _async_runnable_func(_input: Input) -> Output:
                assert self.async_func is not None
                return await self.async_func(**_input.dict())

            return _async_runnable_func
        else:
            return None

    @property
    def runnable(self) -> Runnable[Input, Output]:
        if self.runnable_func:
            return RunnableLambda(self.runnable_func, afunc=self.async_runnable_func)
        elif self.async_runnable_func:
            return RunnableLambda[Input, Output](self.async_runnable_func)
        else:
            raise NotImplementedError("must define at least one process function")

    def tool(self, **kwargs) -> BaseTool:
        return StructuredTool.from_function(
            func=self.tool_func,
            coroutine=self.async_tool_func,
            name=self.name(),
            description=self.description(),
            return_direct=self.return_direct,
            args_schema=self.input_type(),
            metadata={
                "notification": self.notification()
            },
            handle_tool_error=True,
            **kwargs,
        )
