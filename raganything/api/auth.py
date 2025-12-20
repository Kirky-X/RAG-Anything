# Copyright (c) 2025 Kirky.X
# All rights reserved.

import fnmatch
from typing import Callable, Optional

from fastapi import Header, HTTPException, Request

from raganything.server_config import load_server_configs
from raganything.i18n import _


def _path_whitelisted(path: str, whitelist: list[str]) -> bool:
    for rule in whitelist or []:
        if fnmatch.fnmatch(path, rule):
            return True
    return False


def get_auth(
    x_api_key: Optional[str] = Header(None),
    request: Request = None,  # type: ignore[assignment]
) -> None:
    srv, api = load_server_configs()
    api_key = api.lightrag_api_key
    whitelist = api.whitelist_paths

    req_path = request.url.path if request else ""
    if _path_whitelisted(req_path, whitelist):
        return None
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail=_("Invalid API key"))
    return None
