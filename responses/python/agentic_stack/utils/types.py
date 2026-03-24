import argparse
import unicodedata
from collections import OrderedDict
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timezone
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Tuple, Type, TypeVar, Union

from pydantic import AfterValidator, BaseModel, BeforeValidator
from pydantic.types import AwareDatetime
from pydantic_extra_types.country import _index_by_alpha2 as iso_3166
from pydantic_extra_types.language_code import _index_by_alpha2 as iso_639

# fmt: off
FilePath = Union[str, Path]
# Superficial JSON input/output types
# https://github.com/python/typing/issues/182#issuecomment-186684288
JSONOutput = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONOutputBin = Union[bytes, str, int, float, bool, None, Dict[str, Any], List[Any]]
# For input, we also accept tuples, ordered dicts etc.
JSONInput = Union[str, int, float, bool, None, Dict[str, Any], List[Any], Tuple[Any, ...], OrderedDict]
JSONInputBin = Union[bytes, str, int, float, bool, None, Dict[str, Any], List[Any], Tuple[Any, ...], OrderedDict]
YAMLInput = JSONInput
YAMLOutput = JSONOutput
# fmt: on


### --- Datetime Validator --- ###


# def _datetime_to_str(value: Any) -> Any:
#     if isinstance(value, datetime):
#         return value.isoformat()
#     return value


# No longer used, but kept for reference
# ISO_8601_UTC = Annotated[
#     str, BeforeValidator(_datetime_to_str), AfterValidator(utc_iso_from_string)
# ]


def _to_utc(d: AwareDatetime) -> AwareDatetime:
    return d.astimezone(timezone.utc)


DatetimeUTC = Annotated[AwareDatetime, AfterValidator(_to_utc)]


def _to_date(value: _date | _datetime) -> _date:
    if isinstance(value, _datetime):
        return value.date()
    if isinstance(value, _date):
        return value
    raise ValueError("Value cannot be converted to date.")


DateUTC = Annotated[_date, AfterValidator(_to_date)]

### --- String Validator --- ###


def _is_bad_char(char: str) -> bool:
    """Check if a character is not allowable."""
    # Get character properties
    category = unicodedata.category(char)
    # Check for combining marks (they stack vertically)
    if category.startswith("M"):
        return True
    # Block elements
    if "\u2580" <= char <= "\u259f":
        return True
    # Braille patterns
    if "\u2800" <= char <= "\u28ff":
        return True
    # Box drawing
    if "\u2500" <= char <= "\u257f":
        return True
    return False


def _str_pre_validator(value: Any, *, disallow_empty_string: bool = False) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if disallow_empty_string and len(value) == 0:
        raise ValueError("Text is empty.")
    # Check if all characters are printable
    if not value.isprintable():
        raise ValueError("Text contains non-printable or newline characters.")
    # if not re.match(r"^\S(?:[ \S]*\S)?$", value):
    #     raise ValueError("Text contains invalid characters.")
    if any(_is_bad_char(char) for char in value):
        raise ValueError("Text contains disallowed characters.")
    return value


SanitisedStr = Annotated[
    str,
    BeforeValidator(_str_pre_validator),
    # Cannot use Field here due to conflict with SQLModel
]
SanitisedNonEmptyStr = Annotated[
    str,
    BeforeValidator(partial(_str_pre_validator, disallow_empty_string=True)),
    # Cannot use Field here due to conflict with SQLModel
]


### --- Language Code Validator --- ###


WILDCARD_LANG_CODES = {"*", "mul"}
DEFAULT_MUL_LANGUAGES = [
    # ChatGPT supported languages
    # "sq",  # Albanian
    # "am",  # Amharic
    # "ar",  # Arabic
    # "hy",  # Armenian
    # "bn",  # Bengali
    # "bs",  # Bosnian
    # "bg",  # Bulgarian
    # "my",  # Burmese
    # "ca",  # Catalan
    "zh",  # Chinese
    # "hr",  # Croatian
    # "cs",  # Czech
    # "da",  # Danish
    # "nl",  # Dutch
    "en",  # English
    # "et",  # Estonian
    # "fi",  # Finnish
    "fr",  # French
    # "ka",  # Georgian
    # "de",  # German
    # "el",  # Greek
    # "gu",  # Gujarati
    # "hi",  # Hindi
    # "hu",  # Hungarian
    # "is",  # Icelandic
    # "id",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese
    # "kn",  # Kannada
    # "kk",  # Kazakh
    "ko",  # Korean
    # "lv",  # Latvian
    # "lt",  # Lithuanian
    # "mk",  # Macedonian
    # "ms",  # Malay
    # "ml",  # Malayalam
    # "mr",  # Marathi
    # "mn",  # Mongolian
    # "no",  # Norwegian
    # "fa",  # Persian
    # "pl",  # Polish
    # "pt",  # Portuguese
    # "pa",  # Punjabi
    # "ro",  # Romanian
    # "ru",  # Russian
    # "sr",  # Serbian
    # "sk",  # Slovak
    # "sl",  # Slovenian
    # "so",  # Somali
    "es",  # Spanish
    # "sw",  # Swahili
    # "sv",  # Swedish
    # "tl",  # Tagalog
    # "ta",  # Tamil
    # "te",  # Telugu
    # "th",  # Thai
    # "tr",  # Turkish
    # "uk",  # Ukrainian
    # "ur",  # Urdu
    # "vi",  # Vietnamese
]


def _validate_lang(s: str) -> str:
    try:
        code = s.split("-")
        lang = code[0]
        lang = lang.lower().strip()
        if lang not in iso_639():
            raise ValueError
        if len(code) == 2:
            country = code[1]
            country = country.upper().strip()
            if country not in iso_3166():
                raise ValueError
            return f"{lang}-{country}"
        elif len(code) == 1:
            return lang
        else:
            raise ValueError
    except Exception as e:
        raise ValueError(
            f'Language code "{s}" is not ISO 639-1 alpha-2 or BCP-47 ([ISO 639-1 alpha-2]-[ISO 3166-1 alpha-2]).'
        ) from e


def _validate_lang_list(s: list[str]) -> list[str]:
    s = {lang.strip() for lang in s}
    if len(s & WILDCARD_LANG_CODES) > 0:
        s = list((s - WILDCARD_LANG_CODES) | set(DEFAULT_MUL_LANGUAGES))
    return [_validate_lang(lang) for lang in s]


LanguageCodeList = Annotated[list[str], AfterValidator(_validate_lang_list)]


### --- Enum Validator --- ###

E = TypeVar("E", bound=Enum)


def get_enum_validator(enum_cls: Type[E]) -> Callable[[str], E]:
    def _validator(v: str) -> E:
        try:
            return enum_cls[v]
        except KeyError:
            return enum_cls(v)

    return _validator


### --- Pydantic Model Validator --- ###

# M = TypeVar("M", bound=BaseModel)


# def get_pydantic_validator(model_cls: Type[M]) -> Callable[[dict[str, Any] | M], M]:
#     def _validator(v: dict[str, Any] | M) -> M:
#         return model_cls.model_validate(v)

#     return _validator


class CLI(BaseModel):
    @classmethod
    def parse_args(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        for field_name, field_info in cls.model_fields.items():
            field_type = field_info.annotation
            default = field_info.default
            description = field_info.description or ""
            if field_type is bool:
                parser.add_argument(
                    f"--{field_name}",
                    action="store_true",
                    help=description,
                )
            else:
                parser.add_argument(
                    f"--{field_name}",
                    type=field_type,
                    default=default,
                    required=default is ...,
                    help=description,
                )
        return cls(**vars(parser.parse_args()))
