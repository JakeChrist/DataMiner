"""Background services for the DataMiner application."""

from .conversation_manager import (
    AnswerLength,
    AssumptionDecision,
    ConnectionState,
    ConversationManager,
    ConversationTurn,
    PlanItem,
    ReasoningArtifacts,
    ReasoningVerbosity,
    ResponseMode,
    SelfCheckResult,
)
from .document_hierarchy import DocumentHierarchyService
from .lmstudio_client import (
    ChatMessage,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioError,
    LMStudioResponseError,
)

__all__ = [
    "AnswerLength",
    "AssumptionDecision",
    "ChatMessage",
    "ConnectionState",
    "ConversationManager",
    "ConversationTurn",
    "PlanItem",
    "ReasoningArtifacts",
    "ReasoningVerbosity",
    "ResponseMode",
    "SelfCheckResult",
    "DocumentHierarchyService",
    "LMStudioClient",
    "LMStudioConnectionError",
    "LMStudioError",
    "LMStudioResponseError",
]

# Optional UI services require PyQt6. They are imported lazily to avoid import
# errors when the runtime environment lacks Qt libraries.
try:  # pragma: no cover - optional dependency guard
    from .progress_service import ProgressService, ProgressUpdate
except ImportError:  # pragma: no cover
    ProgressService = ProgressUpdate = None  # type: ignore[assignment]
else:  # pragma: no cover - executed when Qt is available
    __all__.extend(["ProgressService", "ProgressUpdate"])

try:  # pragma: no cover - optional dependency guard
    from .settings_service import SettingsService
except ImportError:  # pragma: no cover
    SettingsService = None  # type: ignore[assignment]
else:  # pragma: no cover - executed when Qt is available
    __all__.append("SettingsService")
