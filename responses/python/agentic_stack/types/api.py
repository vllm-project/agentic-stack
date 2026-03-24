import re
from typing import Annotated, Generic, Self, TypeVar

from pydantic import BaseModel, Field, computed_field


class UserAgent(BaseModel):
    is_browser: bool = Field(
        True,
        description="Whether the request originates from a browser or an app.",
        examples=[True, False],
    )
    agent: str = Field(
        description="The agent, such as 'SDK', 'Chrome', 'Firefox', 'Edge', or an empty string if it cannot be determined.",
        examples=["", "SDK", "Chrome", "Firefox", "Edge"],
    )
    agent_version: str = Field(
        "",
        description="The agent version, or an empty string if it cannot be determined.",
        examples=["", "5.0", "0.3.0"],
    )
    os: str = Field(
        "",
        description="The system/OS name and release, such as 'Windows NT 10.0', 'Linux 5.15.0-113-generic', or an empty string if it cannot be determined.",
        examples=["", "Windows NT 10.0", "Linux 5.15.0-113-generic"],
    )
    architecture: str = Field(
        "",
        description="The machine type, such as 'AMD64', 'x86_64', or an empty string if it cannot be determined.",
        examples=["", "AMD64", "x86_64"],
    )
    language: str = Field(
        "",
        description="The SDK language, such as 'TypeScript', 'Python', or an empty string if it is not applicable.",
        examples=["", "TypeScript", "Python"],
    )
    language_version: str = Field(
        "",
        description="The SDK language version, such as '4.9', '3.10.14', or an empty string if it is not applicable.",
        examples=["", "4.9", "3.10.14"],
    )

    @computed_field(
        description="The system/OS name, such as 'Linux', 'Darwin', 'Java', 'Windows', or an empty string if it cannot be determined.",
        examples=["", "Windows NT", "Linux"],
        return_type=str,
    )
    @property
    def system(self) -> str:
        return self._split_os_string()[0]

    @computed_field(
        description="The system's release, such as '2.2.0', 'NT', or an empty string if it cannot be determined.",
        examples=["", "10", "5.15.0-113-generic"],
        return_type=str,
    )
    @property
    def system_version(self) -> str:
        return self._split_os_string()[1]

    def _split_os_string(self) -> tuple[str, str]:
        match = re.match(r"([^\d]+) ([\d.]+).*$", self.os)
        if match:
            os_name = match.group(1).strip()
            os_version = match.group(2).strip()
            return os_name, os_version
        else:
            return "", ""

    @classmethod
    def from_user_agent_string(cls, ua_string: str) -> Self:
        if not ua_string:
            return cls(is_browser=False, agent="")

        # SDK pattern
        sdk_match = re.match(r"SDK/(\S+) \((\w+)/(\S+); ([^;]+); (\w+)\)", ua_string)
        if sdk_match:
            return cls(
                is_browser=False,
                agent="SDK",
                agent_version=sdk_match.group(1),
                os=sdk_match.group(4),
                architecture=sdk_match.group(5),
                language=sdk_match.group(2),
                language_version=sdk_match.group(3),
            )

        # Browser pattern
        browser_match = re.match(r"Mozilla/5.0 \(([^)]+)\).*", ua_string)
        if browser_match:
            os_info = browser_match.group(1).split(";")
            # Microsoft Edge
            match = re.match(r".+(Edg/.+)$", ua_string)
            if match:
                return cls(
                    agent="Edge",
                    agent_version=match.group(1).split("/")[-1].strip(),
                    os=os_info[0].strip(),
                    architecture=os_info[-1].strip() if len(os_info) == 3 else "",
                    language="",
                    language_version="",
                )
            # Firefox
            match = re.match(r".+(Firefox/.+)$", ua_string)
            if match:
                return cls(
                    agent="Firefox",
                    agent_version=match.group(1).split("/")[-1].strip(),
                    os=os_info[0].strip(),
                    architecture=os_info[-1].strip() if len(os_info) == 3 else "",
                    language="",
                    language_version="",
                )
            # Chrome
            match = re.match(r".+(Chrome/.+)$", ua_string)
            if match:
                return cls(
                    agent="Chrome",
                    agent_version=match.group(1).split("/")[-1].strip(),
                    os=os_info[0].strip(),
                    architecture=os_info[-1].strip() if len(os_info) == 3 else "",
                    language="",
                    language_version="",
                )
        return cls(is_browser="mozilla" in ua_string.lower(), agent="")


class OkResponse(BaseModel):
    ok: bool = True


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: Annotated[
        list[T], Field(description="List of items paginated items.", examples=[[]])
    ] = []
    offset: Annotated[int, Field(description="Number of skipped items.", examples=[0])] = 0
    limit: Annotated[int, Field(description="Number of items per page.", examples=[0])] = 0
    total: Annotated[int, Field(description="Total number of items.", examples=[0])] = 0
    end_cursor: Annotated[
        str | None,
        Field(
            description=(
                "Opaque cursor token for the last item in this page. "
                "Pass it as `after=<end_cursor>` to request the page that follows the current window."
            )
        ),
    ] = None
