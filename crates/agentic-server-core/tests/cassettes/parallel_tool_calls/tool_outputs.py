"""Fake client-tool implementations for parallel-tool-call cassette recording.

Loaded by record_cassette.py's --tool-outputs when given a *.py file. Each
function name here must match a tool name declared in one of this directory's
tools-*.json files (the request-side declarations, which stay JSON and
unchanged). For every pending call in a turn, the recorder parses the model's
actual `arguments` JSON string into keyword arguments and calls the matching
function; the JSON-serialized return value becomes that call's
function_call_output. Returning None omits the call's output entirely,
letting a cassette test a provider's behavior when the client leaves a
specific pending call unresolved.

Sentinel argument values below are used consistently across every cassette
scenario, so the same functions serve success, explicit-failure, and
omission cases without colliding: real city/ticker values always succeed;
"London" / "FAIL" always produce an explicit error output; "Atlantis" /
"OMIT" are always omitted entirely.
"""


def get_weather(city: str):
    if city == "London":
        return {"error": "weather service unavailable: upstream timeout"}
    if city == "Atlantis":
        return None
    return {"city": city, "temperature_c": 22, "condition": "Clear"}


def get_stock_price(ticker: str):
    if ticker == "FAIL":
        return {"error": "stock service unavailable: rate limited"}
    if ticker == "OMIT":
        return None
    return {"ticker": ticker, "price": 231.45, "currency": "USD"}


def set_temperature_unit(unit: str = "fahrenheit"):
    return {"status": "ok", "unit": unit}
