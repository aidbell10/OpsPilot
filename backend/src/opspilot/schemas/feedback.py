"""``POST /feedback`` response shape."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel


class FeedbackResponse(BaseModel):
    id: str
    created_at: dt.datetime
