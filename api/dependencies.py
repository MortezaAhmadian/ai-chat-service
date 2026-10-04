"""FastAPI dependency injection.

Singletons are created once in the lifespan and stored on app.state; these
functions expose them to routes. Tests swap them via app.dependency_overrides.
`Annotated[X, Depends(...)]` aliases avoid repeating Depends() everywhere and
keep signatures type-checkable.
"""

from typing import Annotated, cast

from fastapi import Depends, Request

from api.config import Settings
from api.services.chat_service import ChatService


def get_app_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_chat_service(request: Request) -> ChatService:
    return cast(ChatService, request.app.state.chat_service)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
