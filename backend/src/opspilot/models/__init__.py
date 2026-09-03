"""ORM models.

Importing this package registers every table on ``Base.metadata`` (needed for
Alembic autogenerate and for ``create_all`` in tests).
"""

from opspilot.models.deployment import Deployment
from opspilot.models.document import Document, DocumentChunk
from opspilot.models.evaluation import EvaluationCase, EvaluationResult, EvaluationRun
from opspilot.models.feedback import UserFeedback
from opspilot.models.historical_incident import HistoricalIncident
from opspilot.models.incident import Incident
from opspilot.models.service import Service

__all__ = [
    "Deployment",
    "Document",
    "DocumentChunk",
    "EvaluationCase",
    "EvaluationResult",
    "EvaluationRun",
    "HistoricalIncident",
    "Incident",
    "Service",
    "UserFeedback",
]
