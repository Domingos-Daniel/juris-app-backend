from __future__ import annotations

import json
import urllib.request


def urlopen_without_env_proxy(request: urllib.request.Request, timeout: float):
    # Ignore ambient proxy variables from the shell/session. In this project
    # they may point to a dead localhost proxy and break all provider calls.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def read_json_response(request: urllib.request.Request, timeout: float) -> dict:
    with urlopen_without_env_proxy(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
