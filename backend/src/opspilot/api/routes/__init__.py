"""API routers."""

from opspilot.api.routes.feedback import router as feedback_router
from opspilot.api.routes.health import router as health_router
from opspilot.api.routes.incidents import router as incidents_router
from opspilot.api.routes.search import router as search_router

__all__ = ["feedback_router", "health_router", "incidents_router", "search_router"]
