"""
OpenAI types.

Chat (legacy): [Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create)
Chat: [Responses API](https://platform.openai.com/docs/api-reference/responses/create)
Embedding: [Embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create)
"""

from __future__ import annotations

import json
from collections import defaultdict
from time import time
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, Union

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from pydantic import (
    AliasChoices,
    BaseModel,
    Discriminator,
    Field,
    WrapValidator,
    field_validator,
)
from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ExternalToolset,
    FunctionToolset,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    Tool,
    ToolCallPart,
    ToolDefinition,
    ToolReturnPart,
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_extra_types.timezone_name import TimeZoneName

from agentic_stack.mcp.gateway_toolset import McpGatewayToolset, ResolvedMcpTool
from agentic_stack.mcp.resolver import (
    ResolvedMcpServerTools,
    build_request_remote_toolset,
    resolve_mcp_declarations,
)
from agentic_stack.mcp.types import McpToolInfo, McpToolRef
from agentic_stack.mcp.utils import (
    HOSTED_MCP_INTERNAL_PREFIX,
    build_internal_mcp_tool_name,
    normalize_mcp_input_schema,
)
from agentic_stack.tools import TOOLS
from agentic_stack.tools.ids import CODE_INTERPRETER_TOOL, WEB_SEARCH_TOOL
from agentic_stack.utils import uuid7_str
from agentic_stack.utils.exceptions import BadInputError

if TYPE_CHECKING:
    from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient


class OpenAIConversation(BaseModel):
    """The conversation that this response belongs to."""

    id: Annotated[
        str,
        Field(description="The unique ID of the conversation."),
    ]


class OpenAIReasoningEffort(BaseModel):
    effort: Annotated[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        Field(
            description=(
                "Constrains effort on reasoning for reasoning models. "
                "Currently supported values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`. "
                "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response."
            ),
        ),
    ] = "medium"
    summary: Annotated[
        Literal["auto", "concise", "detailed"] | None,
        Field(
            description=(
                "A summary of the reasoning performed by the model. "
                "This can be useful for debugging and understanding the model's reasoning process. "
                "One of `auto`, `concise`, or `detailed`."
            ),
        ),
    ] = None


# ---- Responses `include[]` (Open Responses schema) ----

# Keep this as a finite set (enum-like) so we fail fast on typos while staying spec-aligned.
# Note: The gateway currently acts upon `web_search_call.action.sources` and
# `code_interpreter_call.outputs` (others are parsed but ignored).
OpenAIResponsesInclude = Literal[
    "web_search_call.action.sources",
    "code_interpreter_call.outputs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
]


class _OpenAIMessage(BaseModel):
    role: Annotated[
        Literal["user", "assistant", "system", "developer"],
        Field(
            description=(
                "The role of the message input. "
                "One of `user`, `assistant`, `system`, or `developer`."
            )
        ),
    ]
    type: Annotated[
        Literal["message"], Field(description="The type of the message input. Always `message`.")
    ] = "message"


class OpenAITextContent(BaseModel):
    """A text input to the model."""

    text: Annotated[
        str,
        Field(description=("The text input to the model.")),
    ]
    type: Annotated[
        Literal["input_text"],
        Field(description=("The type of the input item. Always `input_text`.")),
    ] = "input_text"


class OpenAIImageContent(BaseModel):
    """An image input to the model."""

    detail: Annotated[
        Literal["high", "low", "auto"],
        Field(
            description=(
                "The detail level of the image to be sent to the model. "
                "One of `high`, `low`, or `auto`. Defaults to `auto`."
            ),
        ),
    ] = "auto"
    type: Annotated[
        Literal["input_image"],
        Field(description=("The type of the input item. Always `input_image`.")),
    ] = "input_image"
    # file_id: Annotated[
    #     str | None,
    #     Field(
    #         description=("The ID of the file to be sent to the model."),
    #     ),
    # ] = None
    image_url: Annotated[
        # str | None,
        str,
        Field(
            description=(
                "The URL of the image to be sent to the model. "
                "A fully qualified URL or base64 encoded image in a data URL."
            ),
        ),
    ]


# class OpenAIFileContent(BaseModel):
#     """A file input to the model."""

#     type: Annotated[
#         Literal["input_file"],
#         Field(description=("The type of the input item. Always `input_file`.")),
#     ] = "input_file"
#     file_data: Annotated[
#         str | None,
#         Field(
#             description=("The content of the file to be sent to the model."),
#         ),
#     ] = None
#     file_id: Annotated[
#         str | None,
#         Field(
#             description=("The ID of the file to be sent to the model."),
#         ),
#     ] = None
#     file_url: Annotated[
#         str | None,
#         Field(
#             description=("The URL of the file to be sent to the model."),
#         ),
#     ] = None
#     filename: Annotated[
#         str | None,
#         Field(
#             description=("The name of the file to be sent to the model."),
#         ),
#     ] = None


class OpenAIInputMessage(_OpenAIMessage):
    content: Annotated[
        # str | list[OpenAITextContent | OpenAIImageContent | OpenAIFileContent]
        str | list[OpenAITextContent | OpenAIImageContent],
        Field(
            description=(
                "Text, image, or audio input to the model, used to generate a response. "
                "Can also contain previous assistant responses."
            ),
        ),
    ]


class OpenAIInputItem(_OpenAIMessage):
    content: Annotated[
        # list[OpenAITextContent | OpenAIImageContent | OpenAIFileContent]
        list[OpenAITextContent | OpenAIImageContent],
        Field(
            description=(
                "Text, image, or audio input to the model, used to generate a response. "
                "Can also contain previous assistant responses."
            ),
        ),
    ]
    role: Annotated[
        Literal["user", "system", "developer"],
        Field(
            description=("The role of the message input. One of `user`, `system`, or `developer`.")
        ),
    ]
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(
            description=(
                "The status of item. "
                "One of `in_progress`, `completed`, or `incomplete`. "
                "Populated when items are returned via API."
            ),
            exclude=True,
        ),
    ] = None


class OpenAIOutputTextContent(BaseModel):
    """A text output from the model."""

    annotations: Annotated[
        # TODO: Define annotation structure
        list[dict[str, Any]],
        Field(description=("The annotations of the text output.")),
    ] = Field(default_factory=list)
    logprobs: Annotated[
        # TODO: Define logprob structure
        list[dict[str, Any]],
        Field(description=("Log probability information.")),
    ] = Field(default_factory=list)
    text: Annotated[
        str,
        Field(description=("The text output from the model.")),
    ]
    type: Annotated[
        Literal["output_text"],
        Field(description=("The type of the output text. Always `output_text`.")),
    ] = "output_text"


class OpenAIOutputRefusal(BaseModel):
    """A refusal from the model."""

    refusal: Annotated[
        str,
        Field(description=("The refusal explanation from the model.")),
    ]
    type: Annotated[
        Literal["refusal"],
        Field(description=("The type of the refusal. Always `refusal`.")),
    ] = "refusal"


class OpenAIOutputItem(_OpenAIMessage):
    content: Annotated[
        list[OpenAIOutputTextContent | OpenAIOutputRefusal],
        Field(description=("The content of the output message.")),
    ]
    id: Annotated[
        str,
        Field(description=("The unique ID of the output message.")),
    ]
    role: Annotated[
        Literal["assistant"],
        Field(description=("The role of the output message. Always `assistant`.")),
    ] = "assistant"
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"],
        Field(
            description=(
                "The status of item. "
                "One of `in_progress`, `completed`, or `incomplete`. "
                "Populated when items are returned via API."
            ),
        ),
    ]


class OpenAIReasoningContent(BaseModel):
    """Reasoning text content."""

    text: Annotated[
        str,
        Field(description=("The reasoning text from the model.")),
    ]
    type: Annotated[
        Literal["reasoning_text"],
        Field(description=("The type of the reasoning text. Always `reasoning_text`.")),
    ] = "reasoning_text"


class OpenAIReasoningSummary(BaseModel):
    """Reasoning summary content."""

    text: Annotated[
        str,
        Field(description=("A summary of the reasoning output from the model so far.")),
    ]
    type: Annotated[
        Literal["summary_text"],
        Field(description=("The type of the summary text. Always `summary_text`.")),
    ] = "summary_text"


class OpenAIReasoningItem(BaseModel):
    """
    A description of the chain of thought used by a reasoning model while generating a response.
    Be sure to include these items in your input to the Responses API
    for subsequent turns of a conversation if you are manually managing context.
    """

    id: Annotated[
        str,
        Field(description=("The unique identifier of the reasoning content.")),
    ]
    summary: Annotated[
        list[OpenAIReasoningSummary],
        Field(description=("Reasoning summary content.")),
    ] = Field(default_factory=list)
    type: Annotated[
        Literal["reasoning"],
        Field(description=("The type of the object. Always `reasoning`.")),
    ] = "reasoning"
    content: Annotated[
        list[OpenAIReasoningContent],
        Field(description=("Reasoning text content.")),
    ] = Field(default_factory=list)
    encrypted_content: Annotated[
        str | None,
        Field(
            description=(
                "The encrypted content of the reasoning item - "
                "populated when a response is generated with `reasoning.encrypted_content` in the `include` parameter."
            ),
            exclude=True,
        ),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(
            description=(
                "The status of item. "
                "One of `in_progress`, `completed`, or `incomplete`. "
                "Populated when items are returned via API."
            ),
            exclude=True,
        ),
    ] = None


class OpenAIFunctionToolCall(BaseModel):
    """A tool call to run a function."""

    arguments: Annotated[
        str,
        Field(description=("A JSON string of the arguments to pass to the function.")),
    ]
    call_id: Annotated[
        str,
        Field(description=("The unique ID of the function tool call generated by the model.")),
    ]
    name: Annotated[
        str,
        Field(description=("The name of the function to run.")),
    ]
    type: Annotated[
        Literal["function_call"],
        Field(description=("The type of the function tool call. Always `function_call`.")),
    ] = "function_call"
    id: Annotated[
        str | None,
        Field(description=("The unique ID of the function tool call.")),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(
            description=(
                "The status of item. "
                "One of `in_progress`, `completed`, or `incomplete`. "
                "Populated when items are returned via API."
            ),
        ),
    ] = None


class OpenAIMcpToolCall(BaseModel):
    """An MCP tool call executed by the gateway."""

    id: Annotated[
        str,
        Field(description=("The unique ID of the MCP tool call.")),
    ]
    server_label: Annotated[
        str,
        Field(description=("The MCP server label for this call.")),
    ]
    name: Annotated[
        str,
        Field(description=("The MCP tool name.")),
    ]
    arguments: Annotated[
        str,
        Field(description=("A JSON string of MCP tool arguments.")),
    ]
    status: Annotated[
        Literal["in_progress", "completed", "incomplete", "failed"],
        Field(
            description=(
                "The status of the MCP call item. "
                "One of `in_progress`, `completed`, `incomplete`, or `failed`."
            )
        ),
    ]
    output: Annotated[
        str | None,
        Field(description=("The tool output text for successful MCP calls.")),
    ] = None
    error: Annotated[
        str | None,
        Field(description=("The error text for failed MCP calls.")),
    ] = None
    type: Annotated[
        Literal["mcp_call"],
        Field(description=("The type of the MCP tool call. Always `mcp_call`.")),
    ] = "mcp_call"


class OpenAIFunctionToolOutput(BaseModel):
    """The output of a function tool call."""

    call_id: Annotated[
        str,
        Field(description=("The unique ID of the function tool call generated by the model.")),
    ]
    output: Annotated[
        # str | list[OpenAITextContent | OpenAIImageContent | OpenAIFileContent]
        str | list[OpenAITextContent | OpenAIImageContent],
        Field(description=("Text, image, or file output of the function tool call.")),
    ]
    type: Annotated[
        Literal["function_call_output"],
        Field(
            description=(
                "The type of the function tool call output. Always `function_call_output`."
            )
        ),
    ] = "function_call_output"
    id: Annotated[
        str | None,
        Field(
            description=(
                "The unique ID of the function tool call output. "
                "Populated when this item is returned via API."
            )
        ),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete"] | None,
        Field(
            description=(
                "The status of item. "
                "One of `in_progress`, `completed`, or `incomplete`. "
                "Populated when items are returned via API."
            ),
        ),
    ] = None


class OpenAICodeOutputLog(BaseModel):
    """The logs output from the code interpreter."""

    logs: Annotated[
        str | None,
        Field(description=("The logs output from the code interpreter.")),
    ]
    type: Annotated[
        Literal["logs"],
        Field(description=("The type of the output. Always `logs`")),
    ] = "logs"


class OpenAICodeOutputImage(BaseModel):
    """The image output from the code interpreter."""

    type: Annotated[
        Literal["image"],
        Field(description=("The type of the output. Always `image`")),
    ] = "image"
    url: Annotated[
        str,
        Field(description=("The URL of the image output from the code interpreter.")),
    ]


class OpenAICodeToolCall(BaseModel):
    """Code interpreter tool call."""

    code: Annotated[
        str | None,
        Field(description=("The code to run, or null if not available.")),
    ]
    container_id: Annotated[
        str,
        Field(description=("The ID of the container used to run the code.")),
    ]
    id: Annotated[
        str,
        Field(description=("The unique ID of the code interpreter tool call.")),
    ]
    outputs: Annotated[
        list[OpenAICodeOutputLog | OpenAICodeOutputImage] | None,
        Field(
            description=(
                "The outputs generated by the code interpreter, such as logs or images. "
                "Can be null if no outputs are available."
            )
        ),
    ] = None
    status: Annotated[
        Literal["in_progress", "completed", "incomplete", "interpreting", "failed"],
        Field(
            description=(
                "The status of the code interpreter tool call. "
                "Valid values are `in_progress`, `completed`, `incomplete`, `interpreting`, and `failed`."
            )
        ),
    ]
    type: Annotated[
        Literal["code_interpreter_call"],
        Field(
            description=(
                "The type of the code interpreter tool call. Always `code_interpreter_call`."
            )
        ),
    ] = "code_interpreter_call"


class OpenAIWebSearchSource(BaseModel):
    type: Annotated[
        Literal["url"],
        Field(description="The type of source. Always `url`."),
    ] = "url"
    url: Annotated[
        str,
        Field(description="The canonical URL used by the web search tool."),
    ]


class OpenAIWebSearchActionSearch(BaseModel):
    type: Annotated[
        Literal["search"],
        Field(description="The web search action type."),
    ] = "search"
    query: Annotated[
        str,
        Field(description="The primary search query executed by the tool."),
    ]
    queries: Annotated[
        list[str] | None,
        Field(description="Optional related or rewritten queries used by the tool."),
    ] = None
    sources: Annotated[
        list[OpenAIWebSearchSource] | None,
        Field(description="Optional normalized source URLs used by the search action."),
    ] = None


class OpenAIWebSearchActionOpenPage(BaseModel):
    type: Annotated[
        Literal["open_page"],
        Field(description="The web search action type."),
    ] = "open_page"
    url: Annotated[
        str | None,
        Field(description="The page URL opened by the tool."),
    ] = None


class OpenAIWebSearchActionFindInPage(BaseModel):
    type: Annotated[
        Literal["find_in_page"],
        Field(description="The web search action type."),
    ] = "find_in_page"
    pattern: Annotated[
        str,
        Field(description="The text pattern searched within the page."),
    ]
    url: Annotated[
        str | None,
        Field(description="The page URL searched by the tool."),
    ] = None


class OpenAIWebSearchToolCall(BaseModel):
    action: (
        Annotated[
            OpenAIWebSearchActionSearch
            | OpenAIWebSearchActionOpenPage
            | OpenAIWebSearchActionFindInPage,
            Field(discriminator="type", description="The completed web search action."),
        ]
        | None
    ) = None
    id: Annotated[
        str,
        Field(description="The unique ID of the web search tool call."),
    ]
    status: Annotated[
        Literal["in_progress", "completed", "incomplete", "failed"],
        Field(description="The status of the web search tool call."),
    ]
    type: Annotated[
        Literal["web_search_call"],
        Field(description="The type of the web search tool call."),
    ] = "web_search_call"


vLLMInput = Union[
    OpenAIInputMessage,
    OpenAIInputItem,
    OpenAIOutputItem,
    Annotated[
        Union[
            OpenAIReasoningItem,
            OpenAIFunctionToolCall,
            OpenAIFunctionToolOutput,
            OpenAIMcpToolCall,
            OpenAICodeToolCall,
            OpenAIWebSearchToolCall,
        ],
        Discriminator("type"),
    ],
]
vLLMOutput = Annotated[
    Union[
        OpenAIOutputItem,
        OpenAIReasoningItem,
        OpenAIFunctionToolCall,
        OpenAIMcpToolCall,
        OpenAICodeToolCall,
        OpenAIWebSearchToolCall,
    ],
    Discriminator("type"),
]


class OpenAIPromptTemplate(BaseModel):
    id: Annotated[str, Field(description=("The unique identifier of the prompt template to use."))]
    variables: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional map of values to substitute in for variables in your prompt. "
                "The substitution values can either be strings, or other Response input types like images or files."
            ),
        ),
    ] = None
    version: Annotated[
        str | None,
        Field(
            description=("Optional version of the prompt template."),
        ),
    ] = None


class OpenAITextFormat(BaseModel):
    type: Annotated[
        Literal["text"],
        Field(
            description=("The type of response format being defined. Always `text`."),
        ),
    ] = "text"


class OpenAIJsonSchemaFormat(BaseModel):
    name: Annotated[
        str,
        Field(
            pattern=r"^[a-zA-Z0-9_-]{1,64}$",
            description=(
                "The name of the response format. "
                "Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64."
            ),
        ),
    ]
    schema_: Annotated[
        dict[str, Any],
        Field(
            alias="schema",
            description=("The schema for the response format, described as a JSON Schema object."),
        ),
    ]
    type: Annotated[
        Literal["json_schema"],
        Field(
            description=("The type of response format being defined. Always `json_schema`."),
        ),
    ] = "json_schema"
    description: Annotated[
        str | None,
        Field(
            description=(
                "A description of what the response format is for, used by the model to determine how to respond in the format."
            ),
        ),
    ] = None
    strict: Annotated[
        bool,
        Field(
            description=(
                "Whether to enable strict schema adherence when generating the output. "
                "If set to `true`, the model will always follow the exact schema defined in the schema field. "
                "Only a subset of JSON Schema is supported when `strict` is `true`."
            ),
        ),
    ] = False


class OpenAIJsonObjectFormat(BaseModel):
    type: Annotated[
        Literal["json_object"],
        Field(
            description=("The type of response format being defined. Always `json_object`."),
        ),
    ] = "json_object"


class vLLMTextConfig(BaseModel):
    format: Annotated[
        OpenAITextFormat | None,
        Field(
            description=(
                "An object specifying the format that the model must output. "
                'Configuring `{ "type": "json_schema" }` enables Structured Outputs, '
                "which ensures the model will match your supplied JSON schema. "
                'The default format is `{ "type": "text" }` with no additional options. '
                'Setting to `{ "type": "json_object" }` enables the older JSON mode, '
                "which ensures the message the model generates is valid JSON. "
                "Using `json_schema` is preferred for models that support it."
            ),
        ),
    ] = None


class OpenAITextConfig(BaseModel):
    format: Annotated[
        OpenAITextFormat | OpenAIJsonSchemaFormat | OpenAIJsonObjectFormat | None,
        Field(
            description=(
                "An object specifying the format that the model must output. "
                'Configuring `{ "type": "json_schema" }` enables Structured Outputs, '
                "which ensures the model will match your supplied JSON schema. "
                'The default format is `{ "type": "text" }` with no additional options. '
                'Setting to `{ "type": "json_object" }` enables the older JSON mode, '
                "which ensures the message the model generates is valid JSON. "
                "Using `json_schema` is preferred for models that support it."
            ),
        ),
    ] = None
    verbosity: Annotated[
        Literal["low", "medium", "high"] | None,
        Field(
            description=(
                "Constrains the verbosity of the model's response. "
                "Lower values will result in more concise responses, "
                "while higher values will result in more verbose responses. "
                "Currently supported values are `low`, `medium`, and `high`."
            ),
        ),
    ] = None


class OpenAIHostedToolChoice(BaseModel):
    """
    Indicates that the model should use a built-in tool to generate a response.
    """

    type: Annotated[
        Literal[
            CODE_INTERPRETER_TOOL,
            "web_search",
            "web_search_preview",
            "web_search_2025_08_26",
        ],
        Field(
            description=(
                "The type of hosted tool the model should to use. Allowed values are:\n"
                f"- `{CODE_INTERPRETER_TOOL}`\n"
                "- `web_search`\n"
                "- `web_search_preview`\n"
                "- `web_search_2025_08_26`"
            ),
        ),
    ]


class OpenAIFunctionToolChoice(BaseModel):
    """
    Use this option to force the model to call a specific function.
    """

    name: Annotated[
        str,
        Field(
            description="The name of the function to call.",
        ),
    ]
    type: Annotated[
        Literal["function"],
        Field(
            description="For function calling, the type is always `function`.",
        ),
    ] = "function"


class OpenAIMcpToolChoice(BaseModel):
    """
    Use this option to force the model to call a specific tool on a remote MCP server.
    """

    server_label: Annotated[
        str,
        Field(
            description="The label of the MCP server to use.",
        ),
    ]
    type: Annotated[
        Literal["mcp"],
        Field(
            description="For MCP tools, the type is always `mcp`.",
        ),
    ] = "mcp"
    name: Annotated[
        str | None,
        Field(
            description="The name of the tool to call on the server.",
        ),
    ] = None


class OpenAIAllowedToolsChoice(BaseModel):
    """
    Constrains the tools available to the model to a pre-defined set.
    """

    mode: Annotated[
        Literal["auto", "required"],
        Field(
            description=(
                "Constrains the tools available to the model to a pre-defined set. "
                "`auto` allows the model to pick from among the allowed tools and generate a message. "
                "`required` requires the model to call one or more of the allowed tools."
            ),
        ),
    ]
    tools: Annotated[
        list[OpenAIHostedToolChoice | OpenAIFunctionToolChoice | OpenAIMcpToolChoice],
        Field(
            description=(
                "A list of tool definitions that the model should be allowed to call. "
                "For the Responses API, the list of tool definitions might look like: \n"
                "```\n"
                "[\n"
                '  { "type": "function", "name": "get_weather" },\n'
                '  { "type": "mcp", "server_label": "deepwiki" },\n'
                '  { "type": "image_generation" }\n'
                "]\n"
                "```"
            ),
        ),
    ]
    type: Annotated[
        Literal["allowed_tools"],
        Field(
            description="Allowed tool configuration type. Always `allowed_tools`.",
        ),
    ] = "allowed_tools"

    def validate_vllm_hosted_tools(self) -> None:
        if any(
            isinstance(tool, OpenAIHostedToolChoice)
            and _normalize_hosted_tool_type(tool.type)
            not in {CODE_INTERPRETER_TOOL, WEB_SEARCH_TOOL}
            for tool in self.tools
        ):
            raise ValueError(
                "vLLM does not support hosted tools other than code interpreter and web_search."
            )


def _normalize_hosted_tool_type(tool_type: str) -> str:
    if tool_type in {"web_search_preview", "web_search_2025_08_26"}:
        return WEB_SEARCH_TOOL
    return tool_type


def _apply_allowed_tools_choice(
    *,
    choice: OpenAIAllowedToolsChoice,
    builtin_tools: list[Tool],
    deferred_tools: list[ToolDefinition],
    mcp_servers: dict[str, ResolvedMcpServerTools],
) -> tuple[
    list[Tool],
    list[ToolDefinition],
    dict[str, ResolvedMcpServerTools],
    dict[str, dict[str, McpToolInfo]],
]:
    allowed_builtin_names = {
        _normalize_hosted_tool_type(tool.type)
        for tool in choice.tools
        if isinstance(tool, OpenAIHostedToolChoice)
    }
    allowed_function_names = {
        tool.name for tool in choice.tools if isinstance(tool, OpenAIFunctionToolChoice)
    }
    allowed_mcp_names_by_server: dict[str, set[str] | None] = {}
    for tool in choice.tools:
        if not isinstance(tool, OpenAIMcpToolChoice):
            continue
        if tool.server_label not in allowed_mcp_names_by_server:
            if tool.name is None:
                allowed_mcp_names_by_server[tool.server_label] = None
            else:
                allowed_mcp_names_by_server[tool.server_label] = {tool.name}
            continue
        existing = allowed_mcp_names_by_server[tool.server_label]
        if existing is None:
            continue
        if tool.name is None:
            allowed_mcp_names_by_server[tool.server_label] = None
            continue
        existing.add(tool.name)

    builtin_by_name = {tool.name: tool for tool in builtin_tools}
    missing_builtin_names = sorted(allowed_builtin_names - builtin_by_name.keys())
    if missing_builtin_names:
        raise BadInputError(
            "`tool_choice.allowed_tools` references built-in tools not present in effective tools: "
            f"{missing_builtin_names!r}."
        )

    deferred_by_name = {tool.name: tool for tool in deferred_tools}
    missing_function_names = sorted(allowed_function_names - deferred_by_name.keys())
    if missing_function_names:
        raise BadInputError(
            "`tool_choice.allowed_tools` references function tools not present in effective tools: "
            f"{missing_function_names!r}."
        )

    selected_mcp_tool_infos_by_server: dict[str, dict[str, McpToolInfo]] = {}
    filtered_mcp_servers: dict[str, ResolvedMcpServerTools] = {}
    for server_label, allowed_names in allowed_mcp_names_by_server.items():
        server_runtime = mcp_servers.get(server_label)
        if server_runtime is None:
            raise BadInputError(
                "`tool_choice.allowed_tools` references an MCP `server_label` "
                "not present in effective tools."
            )
        if allowed_names is None:
            filtered_mcp_servers[server_label] = server_runtime
            continue
        missing_names = sorted(
            name for name in allowed_names if name not in server_runtime.allowed_tool_infos
        )
        if missing_names:
            raise BadInputError(
                "`tool_choice.allowed_tools` references MCP tools not present in effective tools: "
                f"server={server_label!r} names={missing_names!r}."
            )
        selected_mcp_tool_infos_by_server[server_label] = {
            name: tool_info
            for name, tool_info in server_runtime.allowed_tool_infos.items()
            if name in allowed_names
        }
        filtered_mcp_servers[server_label] = server_runtime

    return (
        [tool for tool in builtin_tools if tool.name in allowed_builtin_names],
        [tool for tool in deferred_tools if tool.name in allowed_function_names],
        filtered_mcp_servers,
        selected_mcp_tool_infos_by_server,
    )


def _build_required_tool_choice_instruction(
    *,
    kind: Literal["any", "allowed", "builtin", "function", "mcp"],
    builtin_name: str | None = None,
    function_name: str | None = None,
    server_label: str | None = None,
    mcp_tool_name: str | None = None,
) -> str:
    if kind == "builtin":
        assert builtin_name is not None
        return (
            f"You must call the `{builtin_name}` tool before producing the final answer. "
            "Do not answer directly without calling it."
        )
    if kind == "function":
        assert function_name is not None
        return (
            f"You must call the function `{function_name}` before producing the final answer. "
            "Do not answer directly without calling it."
        )
    if kind == "mcp":
        assert server_label is not None
        if mcp_tool_name is None:
            return (
                f"You must call a tool from the MCP server `{server_label}` before producing "
                "the final answer. Do not answer directly without a tool call."
            )
        return (
            f"You must call the MCP tool `{server_label}:{mcp_tool_name}` before producing "
            "the final answer. Do not answer directly without calling it."
        )
    if kind == "allowed":
        return (
            "You must call at least one of the allowed tools before producing the final answer. "
            "Do not answer directly without a tool call."
        )
    return (
        "You must call at least one available tool before producing the final answer. "
        "Do not answer directly without a tool call."
    )


def _merge_internal_instructions(
    *,
    user_instructions: str | None,
    internal_instruction: str | None,
) -> str | None:
    if not internal_instruction:
        return user_instructions
    if not user_instructions:
        return internal_instruction
    return f"{user_instructions.rstrip()}\n\n{internal_instruction}"


OpenAIToolChoice = Union[
    Literal["none", "auto", "required"],
    Annotated[
        Union[
            OpenAIAllowedToolsChoice,
            OpenAIHostedToolChoice,
            OpenAIFunctionToolChoice,
            OpenAIMcpToolChoice,
        ],
        Discriminator("type"),
    ],
]


def _json_schema_validator(value: Any, handler):
    if isinstance(value, dict):
        try:
            Draft202012Validator.check_schema(value)
        except jsonschema_exceptions.SchemaError as exc:
            raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc
        return value
    return handler(value)


JsonSchema = Annotated[dict[str, Any], WrapValidator(_json_schema_validator)]


class AgentRunSettings(TypedDict):
    """
    Subset of kwargs passed to `Agent.run_stream_events(...)`.

    Keep this aligned with the public `pydantic_ai` run signature to avoid untyped
    `dict[str, Any]` plumbing across LM engine boundaries.
    Reference: `.venv/.../pydantic_ai/agent/abstract.py` `run_stream_events(...)`
    keyword parameters (`message_history`, `instructions`, `toolsets`, `usage_limits`).
    """

    message_history: list[ModelMessage]
    instructions: str | None
    toolsets: list[AbstractToolset[Any]] | None
    usage_limits: UsageLimits


def _normalize_mcp_input_schema(
    *,
    server_label: str,
    tool_name: str,
    input_schema: dict[str, object],
) -> dict[str, object]:
    try:
        normalized = normalize_mcp_input_schema(input_schema)
    except (TypeError, ValueError) as exc:
        raise BadInputError(
            f"MCP tool {server_label!r}:{tool_name!r} has invalid `input_schema`: {exc}"
        ) from exc

    if normalized["type"] != "object":
        raise BadInputError(
            f"MCP tool {server_label!r}:{tool_name!r} has invalid `input_schema`; "
            "root `type` must be `object`."
        )

    try:
        Draft202012Validator.check_schema(normalized)
    except jsonschema_exceptions.SchemaError as exc:
        raise BadInputError(
            f"MCP tool {server_label!r}:{tool_name!r} has invalid `input_schema`: {exc.message}"
        ) from exc

    return normalized


class OpenAIResponsesFunctionTool(BaseModel):
    """
    Defines a function in your own code the model can choose to call.
    """

    name: Annotated[
        str,
        Field(
            description="The name of the function to call.",
        ),
    ]
    parameters: Annotated[
        # TODO: Perhaps we should skip validation?
        JsonSchema,
        Field(
            description="A JSON schema object describing the parameters of the function.",
        ),
    ]
    strict: Annotated[
        bool,
        Field(
            description="Whether to enforce strict parameter validation. Defaults to `true`.",
        ),
    ] = True
    type: Annotated[
        Literal["function"],
        Field(
            description="The type of the function tool. Always `function`.",
        ),
    ] = "function"
    description: Annotated[
        str | None,
        Field(
            description=(
                "A description of the function. "
                "Used by the model to determine whether or not to call the function."
            ),
        ),
    ] = None


class OpenAIResponsesMcpTool(BaseModel):
    """MCP server declaration for gateway-side execution."""

    model_config = {"extra": "forbid"}

    type: Annotated[
        Literal["mcp"],
        Field(description="The MCP tool type. Always `mcp`."),
    ] = "mcp"
    server_label: Annotated[
        str,
        Field(description="The request-visible MCP server label."),
    ]
    server_url: Annotated[
        str | None,
        Field(
            description=(
                "Optional request-remote MCP endpoint URL. "
                "When omitted, the declaration resolves to hosted mode by `server_label`."
            )
        ),
    ] = None
    connector_id: Annotated[
        str | None,
        Field(description="Connector-based MCP mode is unsupported by this gateway."),
    ] = None
    headers: Annotated[
        dict[str, str] | None,
        Field(
            description=(
                "Optional request-scoped headers for request-remote MCP declarations. "
                "Hosted MCP declarations do not accept `headers`."
            )
        ),
    ] = None
    authorization: Annotated[
        str | None,
        Field(
            description=(
                "Optional request-scoped bearer token for request-remote MCP. "
                "Mapped to outbound `Authorization: Bearer <token>`."
            )
        ),
    ] = None
    allowed_tools: Annotated[
        list[str] | None,
        Field(description="Optional request-scoped tool allowlist for this MCP declaration."),
    ] = None
    require_approval: Annotated[
        str | None,
        Field(description="Approval policy for MCP calls. Only `never` is supported."),
    ] = None


class OpenAIResponsesWebSearchFilters(BaseModel):
    allowed_domains: Annotated[
        list[str],
        Field(
            description=(
                "Allowed domains for the search. "
                "If not provided, all domains are allowed. "
                "Subdomains of the provided domains are allowed as well."
            ),
            examples=["pubmed.ncbi.nlm.nih.gov"],
        ),
    ] = []


class OpenAIResponsesWebSearchUserLocation(BaseModel):
    city: Annotated[
        str | None,
        Field(
            description="Free text input for the city of the user, e.g. `San Francisco`.",
        ),
    ] = None
    country: Annotated[
        str | None,
        Field(
            max_length=2,
            description="The two-letter ISO country code of the user, e.g. `US`.",
        ),
    ] = None
    region: Annotated[
        str | None,
        Field(
            description="Free text input for the region of the user, e.g. `California`.",
        ),
    ] = None
    timezone: Annotated[
        TimeZoneName | None,
        Field(
            description="The IANA timezone of the user, e.g. `America/Los_Angeles`.",
        ),
    ] = None
    type: Annotated[
        Literal["approximate"],
        Field(
            description="The type of location approximation. Always `approximate`.",
        ),
    ] = "approximate"


class OpenAIResponsesWebSearchTool(BaseModel):
    """
    Search the Internet for sources related to the prompt.
    """

    type: Annotated[
        Literal["web_search", "web_search_preview", "web_search_2025_08_26"],
        Field(description="The type of tool."),
    ] = "web_search"
    filters: Annotated[
        OpenAIResponsesWebSearchFilters | None,
        Field(
            description="Filters for the search.",
        ),
    ] = None
    search_context_size: Annotated[
        Literal["low", "medium", "high"],
        Field(
            description=(
                "High level guidance for the amount of context window space to use for the search. "
                "One of `low`, `medium`, or `high`. `medium` is the default."
            ),
        ),
    ] = "medium"
    user_location: Annotated[
        OpenAIResponsesWebSearchUserLocation | None,
        Field(
            description="The approximate location of the user.",
        ),
    ] = None


class OpenAIResponsesCodeContainer(BaseModel):
    """
    Configuration for a code interpreter container. Optionally specify the IDs of the files to run the code on.
    """

    type: Annotated[
        Literal["auto"],
        Field(description=("Always `auto`.")),
    ] = "auto"
    file_ids: Annotated[
        list[str] | None,
        Field(
            description=("An optional list of uploaded files to make available to your code."),
        ),
    ] = None
    memory_limit: Annotated[
        str | None,
        Field(
            description=("The memory limit for the code interpreter container."),
        ),
    ] = None


class vLLMResponsesCodeTool(BaseModel):
    type: Annotated[
        Literal[CODE_INTERPRETER_TOOL],
        Field(description="The type of tool."),
    ] = CODE_INTERPRETER_TOOL


class OpenAIResponsesCodeTool(vLLMResponsesCodeTool):
    container: Annotated[
        str | OpenAIResponsesCodeContainer,
        Field(
            description=(
                "The code interpreter container. "
                "Can be a container ID or an object that specifies uploaded file IDs to make available to your code, "
                "along with an optional `memory_limit` setting."
            ),
        ),
    ] = OpenAIResponsesCodeContainer()


vLLMResponsesTool = Annotated[
    vLLMResponsesCodeTool
    | OpenAIResponsesWebSearchTool
    | OpenAIResponsesFunctionTool
    | OpenAIResponsesMcpTool,
    Field(discriminator="type", description="The type of tool."),
]
OpenAIResponsesTool = Annotated[
    OpenAIResponsesWebSearchTool
    | OpenAIResponsesCodeTool
    | OpenAIResponsesFunctionTool
    | OpenAIResponsesMcpTool,
    Field(discriminator="type", description="The type of tool."),
]


class OpenAIStreamOptions(BaseModel):
    include_obfuscation: Annotated[
        bool,
        Field(
            description=(
                "When `true`, stream obfuscation will be enabled. "
                "Stream obfuscation adds random characters to an obfuscation field on streaming delta events to normalize payload sizes as a mitigation to certain side-channel attacks. "
                "These obfuscation fields are included by default, but add a small amount of overhead to the data stream. "
                "You can set include_obfuscation to false to optimize for bandwidth if you trust the network links between your application and the API."
            ),
        ),
    ] = True


class vLLMResponsesRequest(BaseModel):
    input: Annotated[
        str | list[vLLMInput],
        Field(
            description="Text, image, or file inputs to the model, used to generate a response."
        ),
    ]
    instructions: Annotated[
        str | None,
        Field(
            description=(
                "A system (or developer) message inserted into the model's context. "
                "When using along with `previous_response_id`, the instructions from a previous response will not be carried over to the next response. "
                "This makes it simple to swap out system (or developer) messages in new responses."
            ),
        ),
    ] = None
    # OpenAI Responses API reference: `instructions` are not carried over when using `previous_response_id`.
    # https://platform.openai.com/docs/guides/text#message-roles-and-instruction-following
    max_output_tokens: Annotated[
        int | None,
        Field(
            description=(
                "An upper bound for the number of tokens that can be generated for a response, "
                "including visible output tokens and reasoning tokens."
            ),
        ),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(
            description=(
                "The maximum number of total calls to built-in tools that can be processed in a response. "
                "This maximum number applies across all built-in tool calls, not per individual tool. "
                "Any further attempts to call a tool by the model will be ignored. "
                "As of 2026-02-22, observed OpenAI runtime behavior is not always consistent with this documented rule across model/tool paths."
            ),
        ),
    ] = None
    metadata: Annotated[
        dict[str, Any] | None,
        Field(
            max_length=16,
            description=(
                "Set of key-value pairs that can be attached to an object. "
                "This can be useful for storing additional information about the object in a structured format."
            ),
        ),
    ] = None
    include: Annotated[
        list[OpenAIResponsesInclude] | None,
        Field(
            description=(
                "Specify additional output data to include in the model response. "
                "MVP: only `code_interpreter_call.outputs` is currently acted upon by the gateway."
            ),
        ),
    ] = None
    model: Annotated[
        str,
        Field(
            description=(
                "Model ID used to generate the response. "
                "Note: Open Responses allows additional ways to specify prompts; this gateway currently requires `model`."
            ),
        ),
    ]
    parallel_tool_calls: Annotated[
        bool,
        Field(
            description="Whether to allow the model to run tool calls in parallel.",
        ),
    ] = True
    previous_response_id: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("previous_response_id", "previous_responses_id"),
            description=(
                "The unique ID of the previous response to the model. Use this to create multi-turn conversations."
            ),
        ),
    ] = None
    reasoning: Annotated[
        OpenAIReasoningEffort | None,
        Field(
            description=(
                "Configuration options for reasoning models. "
                "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response."
            )
        ),
    ] = None
    store: Annotated[
        bool,
        Field(
            description="Whether to store the generated model response for later retrieval via API.",
        ),
    ] = True
    stream: Annotated[
        bool,
        Field(
            description=(
                "If set to `true`, the model response data will be streamed to the client as it is generated using server-sent events."
            ),
        ),
    ] = False
    temperature: Annotated[
        float,
        Field(
            gt=0,
            lt=2,
            description=(
                "What sampling temperature to use, between 0 and 2. "
                "Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. "
                "We generally recommend altering this or `top_p` but not both."
            ),
        ),
    ] = 1.0
    text: Annotated[
        vLLMTextConfig | None,
        Field(
            description=(
                "Configuration options for a text response from the model. "
                "Can be plain text or structured JSON data."
            ),
        ),
    ] = None
    tool_choice: Annotated[
        OpenAIToolChoice,
        Field(
            description=(
                "Controls which (if any) tool is called by the model. "
                "`none` means the model will not call any tool and instead generates a message. "
                "`auto` means the model can pick between generating a message or calling one or more tools. "
                "`required` means the model must call one or more tools. "
                "Alternatively, you can provide an `allowed_tools` list to constrain the model to a specific set of tools."
            ),
        ),
    ] = "auto"
    tools: Annotated[
        list[vLLMResponsesTool] | None,
        Field(
            description=(
                "An array of tools the model may call while generating a response. "
                "You can specify which tool to use by setting the `tool_choice` parameter."
            ),
            min_length=1,
            examples=[
                [
                    vLLMResponsesCodeTool(),
                    OpenAIResponsesFunctionTool(
                        name="get_weather",
                        parameters=dict(
                            type="object",
                            properties=dict(
                                location=dict(
                                    type="string",
                                    description="City and country, e.g. `Bogotá, Colombia`",
                                )
                            ),
                            required=["location"],
                            additionalProperties=False,
                        ),
                        description="Get current temperature for a given location.",
                    ),
                ],
            ],
        ),
    ] = None
    top_logprobs: Annotated[
        int | None,
        Field(
            ge=0,
            description=(
                "An integer specifying the number of most likely tokens to return at each token position, "
                "each with an associated log probability."
            ),
        ),
    ] = None
    top_p: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description=(
                "An alternative to sampling with `temperature`, called nucleus sampling, "
                "where the model considers the results of the tokens with `top_p` probability mass. "
                "So 0.1 means only the tokens comprising the top 10% probability mass are considered. "
                "We generally recommend altering this or `temperature` but not both."
            ),
        ),
    ] = 1.0

    @field_validator("tools", mode="after")
    @classmethod
    def validate_tools(cls, v: list[vLLMResponsesTool] | None) -> list[vLLMResponsesTool] | None:
        if v:
            tool_count = defaultdict(int)
            for t in v:
                if isinstance(t, OpenAIResponsesFunctionTool):
                    tool_name = t.name
                elif isinstance(t, vLLMResponsesCodeTool):
                    tool_name = t.type
                elif isinstance(t, OpenAIResponsesWebSearchTool):
                    tool_name = WEB_SEARCH_TOOL
                elif isinstance(t, OpenAIResponsesMcpTool):
                    # MCP duplicate-by-server validation is handled in `as_run_settings(...)` where request-level
                    # validation can include runtime manager availability checks.
                    continue
                else:
                    raise ValueError(f"Invalid tool type: {type(t)}")
                tool_count[tool_name] += 1
            if any(count > 1 for count in tool_count.values()):
                raise ValueError(
                    f"Duplicate tool names found in tools: {[tool for tool, count in tool_count.items() if count > 1]}."
                )
        return v

    @field_validator("tool_choice", mode="after")
    @classmethod
    def validate_tool_choice(cls, v: OpenAIToolChoice) -> OpenAIToolChoice:
        if isinstance(v, OpenAIAllowedToolsChoice):
            v.validate_vllm_hosted_tools()
        return v

    async def as_run_settings(
        self,
        *,
        builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None = None,
        request_remote_enabled: bool,
        request_remote_url_checks_enabled: bool,
    ) -> tuple[AgentRunSettings, list[Tool], dict[str, McpToolRef]]:
        """
        Converts the request into a dictionary of run settings for Pydantic AI.

        Notes:

        If system prompt is "A" and instructions is "B", then pydantic ai will form this history:
        ```yaml
        messages:
        - content: A
        role: system
        - content: B
        role: system
        - content: Hi.
        role: user
        ```
        """
        mcp_tool_name_map: dict[str, McpToolRef] = {}
        mcp_internal_name_by_ref: dict[McpToolRef, str] = {}

        def _register_internal_mcp_tool(ref: McpToolRef) -> str:
            existing_name = mcp_internal_name_by_ref.get(ref)
            if existing_name is not None:
                return existing_name
            try:
                internal_tool_name = build_internal_mcp_tool_name(
                    server_label=ref.server_label,
                    tool_name=ref.tool_name,
                    existing_map=mcp_tool_name_map,
                )
            except ValueError as exc:
                raise BadInputError(str(exc)) from exc
            mcp_tool_name_map[internal_tool_name] = ref
            mcp_internal_name_by_ref[ref] = internal_tool_name
            return internal_tool_name

        if not isinstance(self.input, str):
            historical_mcp_refs = sorted(
                {
                    McpToolRef(server_label=msg.server_label, tool_name=msg.name)
                    for msg in self.input
                    if isinstance(msg, OpenAIMcpToolCall)
                },
                key=lambda ref: (ref.server_label, ref.tool_name),
            )
            for ref in historical_mcp_refs:
                _register_internal_mcp_tool(ref)

        # Process chat history
        # `previous_response_id` hydration is implemented in Layer 4 (ResponseStore). This method expects
        # `self.input` to already include any prior context/tool outputs that should be part of the upstream prompt.
        if isinstance(self.input, str):
            message_history = [
                ModelRequest(parts=[UserPromptPart(content=self.input)]),
            ]
        else:
            message_history = []
            for msg in self.input:
                if isinstance(msg, (OpenAIInputMessage, OpenAIInputItem)):
                    if msg.role in ("system", "developer"):
                        if not isinstance(msg.content, str):
                            # NOTE: Can system prompt be non-string?
                            raise BadInputError("System prompt must be a string.")
                        message_history.append(
                            ModelRequest(parts=[SystemPromptPart(content=msg.content)])
                        )
                        continue

                    if msg.role == "user":
                        if isinstance(msg.content, str):
                            message_history.append(
                                ModelRequest(parts=[UserPromptPart(content=msg.content)])
                            )
                        else:
                            message_history.append(
                                ModelRequest(
                                    parts=[
                                        UserPromptPart(
                                            content=[
                                                c.text
                                                if isinstance(c, OpenAITextContent)
                                                else ImageUrl(
                                                    url=c.image_url,
                                                    vendor_metadata=dict(detail=c.detail),
                                                )
                                                for c in msg.content
                                            ]
                                        )
                                    ]
                                )
                            )
                        continue

                    if msg.role == "assistant":
                        if not isinstance(msg.content, str):
                            raise BadInputError(
                                "Assistant message input must be a string. "
                                "To replay assistant outputs, use `OpenAIOutputItem`."
                            )
                        message_history.append(
                            ModelResponse(
                                parts=[TextPart(content=msg.content, id=None)],
                                # NOTE: Do we need to provide the provider name?
                                # provider_name="openai",
                            )
                        )
                        continue

                    raise BadInputError(f"Invalid message role: {msg.role}")

                if isinstance(msg, OpenAIOutputItem):
                    msg_id = msg.id
                    for content_item in msg.content:
                        if isinstance(content_item, OpenAIOutputTextContent):
                            message_history.append(
                                ModelResponse(
                                    parts=[TextPart(content=content_item.text, id=msg_id)],
                                    # provider_name="openai",
                                )
                            )
                        elif isinstance(content_item, OpenAIOutputRefusal):
                            message_history.append(
                                ModelResponse(
                                    parts=[TextPart(content=content_item.refusal, id=msg_id)],
                                    # provider_name="openai",
                                )
                            )
                        else:
                            raise BadInputError(
                                f"Invalid assistant content type: {type(content_item)}"
                            )
                    continue

                if isinstance(msg, OpenAIReasoningItem):
                    msg_id = msg.id
                    content = ""
                    if msg.content:
                        content = "".join(c.text for c in msg.content)
                    if (not msg.encrypted_content) and msg.summary:
                        content = "".join(c.text for c in msg.summary)
                    message_history.append(
                        ModelResponse(
                            parts=[
                                ThinkingPart(
                                    content=content,
                                    signature=msg.encrypted_content,
                                    id=msg_id,
                                    # provider_name="openai",
                                )
                            ],
                            # provider_name="openai",
                        )
                    )
                    continue

                if isinstance(msg, OpenAIFunctionToolCall):
                    msg_id = msg.id
                    message_history.append(
                        ModelResponse(
                            parts=[
                                ToolCallPart(
                                    tool_name=msg.name,
                                    args=msg.arguments,
                                    tool_call_id=msg.call_id,
                                    id=msg_id,
                                )
                            ],
                            # provider_name="openai",
                        )
                    )
                    continue

                if isinstance(msg, OpenAIFunctionToolOutput):
                    tool_name: str = next(
                        (
                            i.name
                            for i in self.input
                            if isinstance(i, OpenAIFunctionToolCall) and i.call_id == msg.call_id
                        ),
                        "",
                    )
                    tool_output: str
                    if isinstance(msg.output, str):
                        tool_output = msg.output
                    else:
                        tool_output = json.dumps(
                            [c.model_dump(mode="python") for c in msg.output],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    message_history.append(
                        ModelRequest(
                            parts=[
                                ToolReturnPart(
                                    tool_name=tool_name,
                                    # TODO: Handle list[OpenAITextContent | OpenAIImageContent]
                                    content=tool_output,
                                    tool_call_id=msg.call_id,
                                ),
                            ]
                        )
                    )
                    continue

                if isinstance(msg, OpenAICodeToolCall):
                    msg_id = msg.id
                    message_history.append(
                        ModelResponse(
                            parts=[
                                BuiltinToolCallPart(
                                    tool_name="code_execution",
                                    args={
                                        "container_id": msg.container_id,
                                        "code": msg.code,
                                    },
                                    tool_call_id=msg_id,
                                    # provider_name="openai",
                                ),
                                BuiltinToolReturnPart(
                                    tool_name="code_execution",
                                    content={"status": "completed"},
                                    tool_call_id=msg_id,
                                    # provider_name="openai",
                                ),
                            ],
                            # provider_name="openai",
                        )
                    )
                    continue

                if isinstance(msg, OpenAIWebSearchToolCall):
                    msg_id = msg.id
                    action_payload = (
                        msg.action.model_dump(mode="python", exclude_none=True)
                        if msg.action is not None
                        else {}
                    )
                    message_history.append(
                        ModelResponse(
                            parts=[
                                BuiltinToolCallPart(
                                    tool_name=WEB_SEARCH_TOOL,
                                    args=action_payload,
                                    tool_call_id=msg_id,
                                ),
                                BuiltinToolReturnPart(
                                    tool_name=WEB_SEARCH_TOOL,
                                    content={"status": msg.status, "action": action_payload},
                                    tool_call_id=msg_id,
                                ),
                            ]
                        )
                    )
                    continue

                if isinstance(msg, OpenAIMcpToolCall):
                    msg_id = msg.id
                    internal_tool_name = _register_internal_mcp_tool(
                        McpToolRef(server_label=msg.server_label, tool_name=msg.name)
                    )
                    mcp_args: dict[str, Any] = {}
                    try:
                        parsed_args = json.loads(msg.arguments)
                        if isinstance(parsed_args, dict):
                            mcp_args = parsed_args
                    except Exception:
                        mcp_args = {}
                    tool_return_content: dict[str, Any] = {"status": msg.status}
                    if msg.output is not None:
                        tool_return_content["output"] = msg.output
                    if msg.error is not None:
                        tool_return_content["error"] = msg.error
                    message_history.append(
                        ModelResponse(
                            parts=[
                                BuiltinToolCallPart(
                                    tool_name=internal_tool_name,
                                    args=mcp_args,
                                    tool_call_id=msg_id,
                                ),
                                BuiltinToolReturnPart(
                                    tool_name=internal_tool_name,
                                    content=tool_return_content,
                                    tool_call_id=msg_id,
                                ),
                            ]
                        )
                    )
                    continue

                raise BadInputError(f"Invalid message type: {type(msg)}")

        # Process tool list
        builtin_tools: list[Tool] = []
        deferred_tools: list[ToolDefinition] = []
        mcp_declarations: dict[str, OpenAIResponsesMcpTool] = {}
        mcp_servers: dict[str, ResolvedMcpServerTools] = {}
        selected_mcp_tool_infos_by_server: dict[str, dict[str, McpToolInfo]] = {}
        if self.tools:
            for t in self.tools:
                match t:
                    case OpenAIResponsesFunctionTool():
                        if t.name.startswith(HOSTED_MCP_INTERNAL_PREFIX):
                            raise BadInputError(
                                f"Function tool names starting with {HOSTED_MCP_INTERNAL_PREFIX!r} are reserved."
                            )
                        deferred_tools.append(
                            ToolDefinition(
                                name=t.name,
                                parameters_json_schema=t.parameters,
                                strict=t.strict,
                                description=t.description,
                            )
                        )
                    case vLLMResponsesCodeTool():
                        builtin_tools.append(TOOLS[CODE_INTERPRETER_TOOL])
                    case OpenAIResponsesWebSearchTool():
                        builtin_tools.append(TOOLS[WEB_SEARCH_TOOL])
                    case OpenAIResponsesMcpTool():
                        if t.server_label in mcp_declarations:
                            raise BadInputError(
                                "Duplicate MCP declarations are not allowed for the same "
                                f"`server_label`: {t.server_label!r}."
                            )
                        mcp_declarations[t.server_label] = t
                    case _:
                        raise BadInputError(f"Invalid tool type: {type(t)}")

        if mcp_declarations:
            mcp_servers = await resolve_mcp_declarations(
                declarations=mcp_declarations,
                builtin_mcp_runtime_client=builtin_mcp_runtime_client,
                request_remote_enabled=request_remote_enabled,
                request_remote_url_checks_enabled=request_remote_url_checks_enabled,
                request_remote_toolset_builder=build_request_remote_toolset,
            )

        internal_tool_choice_instruction: str | None = None

        if self.tool_choice == "auto":
            pass
        elif self.tool_choice == "none":
            builtin_tools = []
            deferred_tools = []
            mcp_servers = {}
            selected_mcp_tool_infos_by_server = {}
        elif self.tool_choice == "required":
            if not builtin_tools and not deferred_tools and not mcp_servers:
                raise BadInputError(
                    '`tool_choice="required"` requires at least one effective tool.'
                )
            internal_tool_choice_instruction = _build_required_tool_choice_instruction(kind="any")
        elif isinstance(self.tool_choice, OpenAIAllowedToolsChoice):
            (
                builtin_tools,
                deferred_tools,
                mcp_servers,
                selected_mcp_tool_infos_by_server,
            ) = _apply_allowed_tools_choice(
                choice=self.tool_choice,
                builtin_tools=builtin_tools,
                deferred_tools=deferred_tools,
                mcp_servers=mcp_servers,
            )
            if self.tool_choice.mode == "auto":
                pass
            else:
                if not builtin_tools and not deferred_tools and not mcp_servers:
                    raise BadInputError(
                        '`tool_choice.allowed_tools` with `mode="required"` '
                        "requires at least one effective tool."
                    )
                internal_tool_choice_instruction = _build_required_tool_choice_instruction(
                    kind="allowed"
                )
        elif isinstance(self.tool_choice, OpenAIHostedToolChoice):
            requested_builtin = _normalize_hosted_tool_type(self.tool_choice.type)
            builtin_tools = [t for t in builtin_tools if t.name == requested_builtin]
            if not builtin_tools:
                raise BadInputError(
                    f"`tool_choice.type={self.tool_choice.type!r}` references a built-in tool "
                    "not present in effective tools."
                )
            deferred_tools = []
            mcp_servers = {}
            selected_mcp_tool_infos_by_server = {}
            internal_tool_choice_instruction = _build_required_tool_choice_instruction(
                kind="builtin",
                builtin_name=requested_builtin,
            )
        elif isinstance(self.tool_choice, OpenAIFunctionToolChoice):
            builtin_tools = []
            deferred_tools = [t for t in deferred_tools if t.name == self.tool_choice.name]
            if not deferred_tools:
                raise BadInputError(
                    f"`tool_choice.name` {self.tool_choice.name!r} is not present in effective tools."
                )
            mcp_servers = {}
            selected_mcp_tool_infos_by_server = {}
            internal_tool_choice_instruction = _build_required_tool_choice_instruction(
                kind="function",
                function_name=self.tool_choice.name,
            )
        elif isinstance(self.tool_choice, OpenAIMcpToolChoice):
            if not mcp_servers:
                raise BadInputError(
                    '`tool_choice.type="mcp"` requires at least one MCP tool declaration.'
                )
            selected_server = mcp_servers.get(self.tool_choice.server_label)
            if selected_server is None:
                raise BadInputError(
                    '`tool_choice.type="mcp"` references a `server_label` '
                    "not present in effective tools."
                )
            selected_tools = selected_server.allowed_tool_infos
            if self.tool_choice.name is not None:
                if self.tool_choice.name not in selected_tools:
                    raise BadInputError(
                        f"`tool_choice.name` {self.tool_choice.name!r} is not allowed for "
                        f"server {self.tool_choice.server_label!r}."
                    )
                selected_tools = {self.tool_choice.name: selected_tools[self.tool_choice.name]}
            mcp_servers = {self.tool_choice.server_label: selected_server}
            selected_mcp_tool_infos_by_server = {self.tool_choice.server_label: selected_tools}
            builtin_tools = []
            deferred_tools = []
            internal_tool_choice_instruction = _build_required_tool_choice_instruction(
                kind="mcp",
                server_label=self.tool_choice.server_label,
                mcp_tool_name=self.tool_choice.name,
            )
        else:
            raise BadInputError(f"Invalid `tool_choice` provided: {self.tool_choice}.")

        mcp_resolved_tools: list[ResolvedMcpTool] = []
        for server_label, server_runtime in sorted(mcp_servers.items()):
            tool_infos = selected_mcp_tool_infos_by_server.get(
                server_label, server_runtime.allowed_tool_infos
            )
            for tool_name, tool_info in tool_infos.items():
                normalized_schema = _normalize_mcp_input_schema(
                    server_label=server_label,
                    tool_name=tool_name,
                    input_schema=tool_info.input_schema,
                )
                ref = McpToolRef(
                    server_label=server_label,
                    tool_name=tool_name,
                    mode=server_runtime.mode,
                )
                internal_tool_name = _register_internal_mcp_tool(ref)

                description = (
                    tool_info.description.strip()
                    if isinstance(tool_info.description, str) and tool_info.description.strip()
                    else f"MCP tool {server_label}:{tool_name}"
                )
                mcp_resolved_tools.append(
                    ResolvedMcpTool(
                        internal_name=internal_tool_name,
                        ref=ref,
                        mcp_toolset=server_runtime.mcp_toolset,
                        mcp_tool_name=tool_name,
                        description=description,
                        input_schema=normalized_schema,
                        schema_validator=Draft202012Validator(normalized_schema),
                        secret_values=server_runtime.secret_values,
                        mcp_tool=server_runtime.allowed_mcp_tools_by_name.get(tool_name),
                    )
                )
        toolsets: list[AbstractToolset[Any]] = []
        if builtin_tools:
            toolsets.append(FunctionToolset(tools=builtin_tools))
        if mcp_resolved_tools:
            toolsets.append(
                McpGatewayToolset(
                    tools=mcp_resolved_tools,
                    id="agentic_stack_mcp",
                )
            )
        if deferred_tools:
            toolsets.append(ExternalToolset(tool_defs=deferred_tools))
        run_settings: AgentRunSettings = {
            "message_history": message_history,
            "instructions": _merge_internal_instructions(
                user_instructions=self.instructions,
                internal_instruction=internal_tool_choice_instruction,
            ),
            "toolsets": toolsets or None,
            # Parity note: do not enforce request `max_tool_calls` in pydantic_ai runtime limits.
            # Keep `max_tool_calls` as request/response metadata only, matching observed OpenAI behavior.
            # Preserve pydantic_ai default request_limit safety guard for non-terminating loops.
            "usage_limits": UsageLimits(tool_calls_limit=None),
        }
        return run_settings, builtin_tools, mcp_tool_name_map

    def as_openai_chat_settings(self) -> OpenAIChatModelSettings:
        """
        These parameters are not supported by Responses API:
        - id
        - n
        - presence_penalty
        - frequency_penalty
        - logit_bias
        - stop / stop_sequences
        - seed

        These parameters are not supported by vLLM:
        - user / openai_user
        - service_tier / openai_service_tier
        - prediction / openai_prediction
        - safety_identifier
        - prompt_cache_key / openai_prompt_cache_key
        - prompt_cache_retention / openai_prompt_cache_retention

        Unused OpenAIChatModelSettings fields:
        - timeout: float | Timeout
        - extra_headers: dict[str, str]
        - extra_body: object

        vLLMResponses fields that are mapped in `as_run_settings`:
        - input
        - instructions
        - tools
        - tool_choice
        - previous_response_id

        vLLMResponses fields that are used when creating Agent:
        - model

        vLLMResponses fields that are not used:
        - metadata
        """
        kwargs: dict[str, Any] = self.model_dump(exclude_unset=True, exclude_none=True)

        max_output_tokens = kwargs.pop("max_output_tokens", None)
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens
        reasoning_effort = kwargs.pop("reasoning", {}).get("effort")
        if reasoning_effort:
            kwargs["openai_reasoning_effort"] = reasoning_effort
        top_logprobs = kwargs.pop("top_logprobs", None)
        if top_logprobs:
            kwargs["openai_logprobs"] = True
            kwargs["openai_top_logprobs"] = top_logprobs
        else:
            kwargs["openai_logprobs"] = False

        return OpenAIChatModelSettings(**kwargs)


# class OpenAIResponsesRequest(vLLMResponsesRequest):
#     background: Annotated[
#         bool,
#         Field(
#             description="Whether to run the model response in the background.",
#         ),
#     ] = False
#     conversation: Annotated[
#         str | OpenAIConversation | None,
#         Field(
#             description=(
#                 "The conversation that this response belongs to. "
#                 "Items from this conversation are prepended to `input_items` for this response request. "
#                 "Input items and output items from this response are automatically added to this conversation after this response completes."
#             )
#         ),
#     ] = None
#     include: Annotated[
#         list[
#             Literal[
#                 "web_search_call.action.sources",
#                 "code_interpreter_call.outputs",
#                 "computer_call_output.output.image_url",
#                 "file_search_call.results",
#                 "message.input_image.image_url",
#                 "message.output_text.logprobs",
#                 "reasoning.encrypted_content",
#             ]
#         ]
#         | None,
#         Field(
#             description=(
#                 "Specify additional output data to include in the model response. Currently supported values are:\n"
#                 "`web_search_call.action.sources`: Include the sources of the web search tool call.\n"
#                 "`code_interpreter_call.outputs`: Includes the outputs of python code execution in code interpreter tool call items.\n"
#                 "`computer_call_output.output.image_url`: Include image urls from the computer call output.\n"
#                 "`file_search_call.results`: Include the search results of the file search tool call.\n"
#                 "`message.input_image.image_url:` Include image urls from the input message.\n"
#                 "`message.output_text.logprobs`: Include logprobs with assistant messages.\n"
#                 "`reasoning.encrypted_content`: Includes an encrypted version of reasoning tokens in reasoning item outputs. "
#                 "This enables reasoning items to be used in multi-turn conversations when using the Responses API statelessly "
#                 "(like when the store parameter is set to false, or when an organization is enrolled in the zero data retention program)."
#             ),
#         ),
#     ] = None
#     prompt: Annotated[
#         OpenAIPromptTemplate | None,
#         Field(
#             description=("Reference to a prompt template and its variables."),
#         ),
#     ] = None
#     prompt_cache_key: Annotated[
#         str | None,
#         Field(
#             description=(
#                 "Used to cache responses for similar requests to optimize your cache hit rates. "
#                 "Replaces the `user` field."
#             ),
#         ),
#     ] = None
#     prompt_cache_retention: Annotated[
#         Literal["24h"] | None,
#         Field(
#             description=(
#                 "The retention policy for the prompt cache. "
#                 "Set to `24h` to enable extended prompt caching, "
#                 "which keeps cached prefixes active for longer, up to a maximum of 24 hours."
#             ),
#         ),
#     ] = None
#     safety_identifier: Annotated[
#         str | None,
#         Field(
#             description=(
#                 "A stable identifier used to help detect users of your application that may be violating usage policies. "
#                 "The IDs should be a string that uniquely identifies each user. "
#                 "We recommend hashing their username or email address, in order to avoid sending us any identifying information."
#             ),
#         ),
#     ] = None
#     service_tier: Annotated[
#         Literal["auto", "default", "flex", "priority"],
#         Field(
#             description=(
#                 "Specifies the processing type used for serving the request.\n\n"
#                 "- If set to 'auto', then the request will be processed with the service tier configured in the Project settings. "
#                 "Unless otherwise configured, the Project will use 'default'.\n"
#                 "- If set to 'default', then the request will be processed with the standard pricing and performance for the selected model.\n"
#                 "- If set to 'flex' or 'priority', then the request will be processed with the corresponding service tier.\n"
#                 "- When not set, the default behavior is 'auto'.\n"
#                 "When the `service_tier` parameter is set, the response body will include the `service_tier` value based on the processing mode actually used to serve the request. "
#                 "This response value may be different from the value set in the parameter."
#             ),
#         ),
#     ] = "auto"
#     stream_options: Annotated[
#         OpenAIStreamOptions | None,
#         Field(
#             description=(
#                 "Options for streaming responses. Only set this when you set `stream: true`."
#             ),
#         ),
#     ] = None
#     text: Annotated[
#         OpenAITextConfig | None,
#         Field(
#             description=(
#                 "Configuration options for a text response from the model. "
#                 "Can be plain text or structured JSON data."
#             ),
#         ),
#     ] = None
#     tools: Annotated[
#         list[OpenAIResponsesTool] | None,
#         Field(
#             description=(
#                 "An array of tools the model may call while generating a response. "
#                 "You can specify which tool to use by setting the `tool_choice` parameter."
#             ),
#             min_length=1,
#             examples=[
#                 [
#                     OpenAIResponsesWebSearchTool(),
#                     OpenAIResponsesCodeTool(),
#                     OpenAIResponsesFunctionTool(
#                         name="get_weather",
#                         parameters=dict(
#                             type="object",
#                             properties=dict(
#                                 location=dict(
#                                     type="string",
#                                     description="City and country, e.g. `Bogotá, Colombia`",
#                                 )
#                             ),
#                             required=["location"],
#                             additionalProperties=False,
#                         ),
#                         description="Get current temperature for a given location.",
#                     ),
#                 ],
#             ],
#         ),
#     ] = None
#     truncation: Annotated[
#         Literal["auto", "disabled"],
#         Field(
#             description=(
#                 "The truncation strategy to use for the model response. "
#                 "`auto`: If the input to this Response exceeds the model's context window size, "
#                 "the model will truncate the response to fit the context window by dropping items from the beginning of the conversation. "
#                 "`disabled` (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error."
#             ),
#         ),
#     ] = "disabled"
#     user: Annotated[
#         str | None,
#         Field(
#             description=(
#                 "This field is being replaced by `safety_identifier` and `prompt_cache_key`. "
#                 "Use `prompt_cache_key` instead to maintain caching optimizations. "
#                 "A stable identifier for your end-users. "
#                 "Used to boost cache hit rates by better bucketing similar requests and to help detect and prevent abuse."
#             ),
#         ),
#     ] = None

#     @field_validator("tool_choice", mode="after")
#     @classmethod
#     def validate_tool_choice(
#         cls, v: str | OpenAIAllowedToolsChoice
#     ) -> str | OpenAIAllowedToolsChoice:
#         return v


class OpenAIResponsesResponseError(BaseModel):
    code: Annotated[
        str,
        Field(description="The error code for the response."),
    ]
    message: Annotated[
        str,
        Field(description="A human-readable description of the error."),
    ]


class OpenAIResponsesIncompleteDetails(BaseModel):
    """Details about why the response is incomplete."""

    reason: Annotated[
        str,
        Field(description="The reason why the response is incomplete."),
    ]


class OpenAIInputTokenDetails(BaseModel):
    """A detailed breakdown of the input tokens."""

    cached_tokens: Annotated[
        int | None,
        Field(description="The number of tokens that were retrieved from the cache."),
    ] = None


class OpenAIOutputTokenDetails(BaseModel):
    """A detailed breakdown of the output tokens."""

    reasoning_tokens: Annotated[
        int | None,
        Field(description="The number of reasoning tokens."),
    ] = None


class OpenAIResponsesUsage(BaseModel):
    input_tokens: Annotated[
        int,
        Field(description="The number of input tokens."),
    ]
    input_tokens_details: Annotated[
        OpenAIInputTokenDetails | None,
        Field(description="A detailed breakdown of the input tokens."),
    ] = None
    output_tokens: Annotated[
        int,
        Field(description="The number of output tokens."),
    ]
    output_tokens_details: Annotated[
        OpenAIOutputTokenDetails | None,
        Field(description="A detailed breakdown of the output tokens."),
    ] = None
    total_tokens: Annotated[
        int,
        Field(description="The total number of tokens used."),
    ]


class OpenAIResponsesResponse(BaseModel):
    background: Annotated[
        bool,
        Field(
            description="Whether to run the model response in the background.",
        ),
    ] = False
    completed_at: Annotated[
        int | None,
        Field(
            description=("Unix timestamp (in seconds) of when this Response was completed."),
        ),
    ] = None
    conversation: Annotated[
        OpenAIConversation | None,
        Field(
            description=(
                "The conversation that this response belongs to. "
                "Input items and output items from this response are automatically added to this conversation."
            )
        ),
    ] = None
    created_at: Annotated[
        int,
        Field(
            default_factory=lambda: int(time()),
            description=("Unix timestamp (in seconds) of when this Response was created."),
        ),
    ]
    error: Annotated[
        OpenAIResponsesResponseError | None,
        Field(description="An error object returned when the model fails to generate a Response."),
    ] = None
    id: Annotated[
        str,
        Field(
            default_factory=lambda: uuid7_str("resp_"),
            description="Unique identifier for this Response.",
        ),
    ]
    incomplete_details: Annotated[
        OpenAIResponsesIncompleteDetails | None,
        Field(description="Details about why the response is incomplete."),
    ] = None
    instructions: Annotated[
        str | None,
        Field(
            description=(
                "A system (or developer) message inserted into the model's context. "
                "When using along with `previous_response_id`, the instructions from a previous response will not be carried over to the next response. "
                "This makes it simple to swap out system (or developer) messages in new responses."
            ),
        ),
    ] = None
    max_output_tokens: Annotated[
        int | None,
        Field(
            description=(
                "An upper bound for the number of tokens that can be generated for a response, "
                "including visible output tokens and reasoning tokens."
            ),
        ),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(
            description=(
                "The maximum number of total calls to built-in tools that can be processed in a response. "
                "This maximum number applies across all built-in tool calls, not per individual tool. "
                "Any further attempts to call a tool by the model will be ignored. "
                "As of 2026-02-22, observed OpenAI runtime behavior is not always consistent with this documented rule across model/tool paths."
            ),
        ),
    ] = None
    metadata: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Set of key-value pairs that can be attached to an object. "
                "This can be useful for storing additional information about the object in a structured format."
            ),
        ),
    ] = Field(default_factory=dict)
    model: Annotated[
        str,
        Field(description="Model ID used to generate the response."),
    ]
    object: Annotated[
        Literal["response"],
        Field(description="The object type of this resource. Always `response`."),
    ] = "response"
    output: Annotated[
        list[vLLMOutput],
        Field(
            description=(
                "An array of content items generated by the model. "
                "The length and order of items in the output array is dependent on the model's response."
            )
        ),
    ] = Field(default_factory=list)
    parallel_tool_calls: Annotated[
        bool,
        Field(
            description="Whether to allow the model to run tool calls in parallel.",
        ),
    ] = True
    previous_response_id: Annotated[
        str | None,
        Field(
            description=(
                "The unique ID of the previous response to the model. "
                "Use this to create multi-turn conversations. "
                "Cannot be used in conjunction with conversation."
            ),
        ),
    ] = None
    prompt: Annotated[
        OpenAIPromptTemplate | None,
        Field(
            description=("Reference to a prompt template and its variables."),
        ),
    ] = None
    prompt_cache_key: Annotated[
        str | None,
        Field(
            description=(
                "Used to cache responses for similar requests to optimize your cache hit rates. "
                "Replaces the `user` field."
            ),
        ),
    ] = None
    prompt_cache_retention: Annotated[
        Literal["24h"] | None,
        Field(
            description=(
                "The retention policy for the prompt cache. "
                "Set to `24h` to enable extended prompt caching, "
                "which keeps cached prefixes active for longer, up to a maximum of 24 hours."
            ),
        ),
    ] = None
    reasoning: Annotated[
        OpenAIReasoningEffort,
        Field(
            description=(
                "Configuration options for reasoning models. "
                "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response."
            )
        ),
    ] = Field(default_factory=OpenAIReasoningEffort)
    safety_identifier: Annotated[
        str | None,
        Field(
            description=(
                "A stable identifier used to help detect users of your application that may be violating usage policies. "
                "The IDs should be a string that uniquely identifies each user. "
                "We recommend hashing their username or email address, in order to avoid sending us any identifying information."
            ),
        ),
    ] = None
    service_tier: Annotated[
        Literal["auto", "default", "flex", "priority"],
        Field(
            description=("Specifies the processing type used for serving the request."),
        ),
    ] = "default"
    store: Annotated[
        bool,
        Field(
            description="Whether to store the generated model response for later retrieval via API."
        ),
    ] = True
    status: Annotated[
        Literal["completed", "failed", "in_progress", "cancelled", "queued", "incomplete"],
        Field(
            description=(
                "The status of the response generation. "
                "One of `completed`, `failed`, `in_progress`, `cancelled`, `queued`, or `incomplete`."
            ),
        ),
    ] = "in_progress"
    temperature: Annotated[
        float,
        Field(description=("What sampling temperature to use.")),
    ] = 1.0
    text: Annotated[
        OpenAITextConfig,
        Field(
            description=(
                "Configuration options for a text response from the model. "
                "Can be plain text or structured JSON data."
            ),
        ),
    ] = Field(
        default_factory=lambda: OpenAITextConfig(format=OpenAITextFormat(), verbosity="medium")
    )
    tool_choice: Annotated[
        OpenAIToolChoice,
        Field(description=("Controls which (if any) tool is called by the model.")),
    ] = "auto"
    tools: Annotated[
        list[vLLMResponsesTool],
        Field(description=("An array of tools the model may call while generating a response.")),
    ] = Field(default_factory=list)
    presence_penalty: Annotated[
        float,
        Field(
            description="Unused by Responses API; always 0 in this gateway (schema-compat default)."
        ),
    ] = 0.0
    frequency_penalty: Annotated[
        float,
        Field(
            description="Unused by Responses API; always 0 in this gateway (schema-compat default)."
        ),
    ] = 0.0
    top_logprobs: Annotated[
        int,
        Field(
            description=(
                "An integer specifying the number of most likely tokens to return at each token position, "
                "each with an associated log probability."
            ),
        ),
    ] = 0
    top_p: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description=(
                "An alternative to sampling with `temperature`, called nucleus sampling, "
                "where the model considers the results of the tokens with `top_p` probability mass."
            ),
        ),
    ] = 1.0
    truncation: Annotated[
        Literal["auto", "disabled"],
        Field(description=("The truncation strategy to use for the model response.")),
    ] = "disabled"
    usage: Annotated[
        OpenAIResponsesUsage | None,
        Field(
            description=(
                "Represents token usage details including input tokens, output tokens, a breakdown of output tokens, and the total tokens used."
            )
        ),
    ] = None
    user: Annotated[
        str | None,
        Field(description=("A stable identifier for your end-users.")),
    ] = None


class _OpenAIStreamEvent(BaseModel):
    type: Annotated[
        str,
        Field(description=("The type of the event.")),
    ]

    def as_responses_chunk(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json(exclude_none=True)}\n\n"

    def as_completion_chunk(self) -> str:
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"


class OpenAIResponsesError(_OpenAIStreamEvent):
    """
    An error event returned when the model fails to generate a Response.
    Also used for non-streaming error.
    """

    code: Annotated[
        str,
        Field(description=("The error code.")),
    ]
    message: Annotated[
        str,
        Field(description=("The error message.")),
    ]
    param: Annotated[
        str,
        Field(description=("The error parameter.")),
    ]
    sequence_number: Annotated[
        int | None,
        Field(description=("The sequence number for this event.")),
    ] = None
    type: Annotated[
        Literal["error"],
        Field(description=("The type of the event.")),
    ] = "error"


class OpenAIResponsesStream(_OpenAIStreamEvent):
    response: Annotated[
        OpenAIResponsesResponse,
        Field(description=("The response.")),
    ]
    sequence_number: Annotated[
        int,
        Field(description=("The sequence number for this event.")),
    ]
    type: Annotated[
        Literal[
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.queued",
        ],
        Field(description=("The type of the event.")),
    ]

    def as_responses_chunk(self) -> str:
        # OpenResponses/OpenAI operational parity:
        # - Lifecycle events embed a full ResponseResource object.
        # - Many keys are required even when their values are `null`.
        # So we must not drop `None` values from the embedded response.
        payload = dict(
            type=self.type,
            sequence_number=self.sequence_number,
            response=self.response.model_dump(mode="python", exclude_none=False),
        )
        _ensure_code_interpreter_outputs_key(payload)
        return f"event: {self.type}\ndata: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


class OpenAIResponsesStreamOutput(_OpenAIStreamEvent):
    item: Annotated[
        vLLMOutput,
        Field(description=("The output item.")),
    ]
    output_index: Annotated[
        int,
        Field(description=("The index of the output item that was added.")),
    ]
    sequence_number: Annotated[
        int,
        Field(description=("The sequence number for this event.")),
    ]
    type: Annotated[
        Literal[
            "response.output_item.added",
            "response.output_item.done",
        ],
        Field(description=("The type of the event.")),
    ]

    def as_responses_chunk(self) -> str:
        payload = self.model_dump(mode="python", exclude_none=True)
        _ensure_code_interpreter_outputs_key(payload)
        return f"event: {self.type}\ndata: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


def _ensure_code_interpreter_outputs_key(value: Any) -> None:
    """OpenAI operational parity: include `outputs: null` for code_interpreter_call items when outputs are omitted.

    Our default stream serialization uses `exclude_none=True`, which would drop `outputs` entirely when it is `None`.
    OpenAI's observed behavior includes `"outputs": null` unless the request includes `code_interpreter_call.outputs`.
    """

    if isinstance(value, dict):
        if value.get("type") == "code_interpreter_call" and "outputs" not in value:
            value["outputs"] = None
        for v in value.values():
            _ensure_code_interpreter_outputs_key(v)
        return
    if isinstance(value, list):
        for v in value:
            _ensure_code_interpreter_outputs_key(v)


class _OpenAIResponsesStream(_OpenAIStreamEvent):
    content_index: Annotated[
        int,
        Field(description=("The index of the content part.")),
    ] = 0
    item_id: Annotated[
        str,
        Field(description=("The ID of the output item that the content part.")),
    ]
    output_index: Annotated[
        int,
        Field(description=("The index of the output item.")),
    ]
    sequence_number: Annotated[
        int,
        Field(description=("The sequence number for this event.")),
    ]


class OpenAIResponsesStreamPart(_OpenAIResponsesStream):
    part: Annotated[
        OpenAIOutputTextContent
        | OpenAIOutputRefusal
        | OpenAIReasoningContent
        | OpenAIReasoningSummary,
        Field(description=("The content part.")),
    ]
    summary_index: Annotated[
        int | None,
        Field(description=("The index of the summary part within the reasoning summary.")),
    ] = None
    type: Annotated[
        Literal[
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        ],
        Field(description=("The type of the event.")),
    ]


class OpenAIResponsesStreamText(_OpenAIResponsesStream):
    # Do not add a top-level stream `error` field here for `response.mcp_call.failed`.
    # MCP failure text is item-level (`mcp_call.error`) to match OpenAI event shape and
    # avoid OpenAI SDK stream parsing this as a fatal error event.
    delta: Annotated[
        str | None,
        Field(description=("The delta for text, refusal, function-call arguments, or reasoning.")),
    ] = None
    text: Annotated[
        str | None,
        Field(description=("The text content that is finalized.")),
    ] = None
    refusal: Annotated[
        str | None,
        Field(description=("The refusal text that is finalized.")),
    ] = None
    arguments: Annotated[
        str | None,
        Field(description=("The function-call arguments.")),
    ] = None
    code: Annotated[
        str | None,
        Field(description=("The final code snippet output by the code interpreter.")),
    ] = None
    output: Annotated[
        str | None,
        Field(description=("The finalized output text for hosted MCP calls.")),
    ] = None
    logprobs: Annotated[
        # TODO: Define logprob structure
        list[dict[str, Any]] | None,
        Field(description=("The log probabilities of the tokens in the delta.")),
    ] = None
    summary_index: Annotated[
        int | None,
        Field(description=("The index of the summary part within the reasoning summary.")),
    ] = None
    type: Annotated[
        Literal[
            "response.output_text.delta",
            "response.output_text.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            # OpenResponses schema uses `response.reasoning.delta/done` (not `reasoning_text.*`).
            "response.reasoning.delta",
            "response.reasoning.done",
            "response.refusal.delta",
            "response.refusal.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call.completed",
            "response.code_interpreter_call_code.delta",
            "response.code_interpreter_call_code.done",
            "response.mcp_call.in_progress",
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.mcp_call.completed",
            "response.mcp_call.failed",
        ],
        Field(description=("The type of the event.")),
    ]


class OpenAIResponsesStreamAnnotation(_OpenAIResponsesStream):
    annotation: Annotated[
        # TODO: Define annotation structure
        dict[str, Any],
        Field(description=("The annotation object.")),
    ]
    annotation_index: Annotated[
        int | None,
        Field(description=("The index of the annotation within the content part.")),
    ] = None
    type: Annotated[
        Literal["response.output_text.annotation.added"],
        Field(description=("The type of the event.")),
    ]
