"""Background services for the DataMiner application."""

from .conversation_manager import (
    AnswerLength,
    AssumptionDecision,
    ConnectionState,
    ConversationManager,
    ConversationTurn,
    ConsolidatedSection,
    ConflictNote,
    ConsolidationOutput,
    EvidenceRecord,
    DynamicPlanningError,
    JudgeReport,
    PlanItem,
    ReasoningArtifacts,
    ReasoningVerbosity,
    ResponseMode,
    SelfCheckResult,
    StateDigestEntry,
    StepContextBatch,
    StepResult,
    TaskCharter,
)
from .conversation_settings import ConversationSettings
from .document_hierarchy import DocumentHierarchyService
from .export_service import ExportService
from .lmstudio_client import (
    ChatMessage,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioError,
    LMStudioResponseError,
)
from .working_memory import WorkingMemoryService

__all__ = [
    "AnswerLength",
    "AssumptionDecision",
    "ChatMessage",
    "ConnectionState",
    "ConversationManager",
    "ConversationSettings",
    "ConversationTurn",
    "ConsolidatedSection",
    "ConflictNote",
    "ConsolidationOutput",
    "EvidenceRecord",
    "DynamicPlanningError",
    "JudgeReport",
    "ExportService",
    "PlanItem",
    "ReasoningArtifacts",
    "ReasoningVerbosity",
    "ResponseMode",
    "SelfCheckResult",
    "StateDigestEntry",
    "StepContextBatch",
    "StepResult",
    "TaskCharter",
    "WorkingMemoryService",
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

try:  # pragma: no cover - optional dependency guard
    from .project_service import ProjectRecord, ProjectService
    from .backup_service import BackupService
except ImportError:  # pragma: no cover
    ProjectRecord = ProjectService = BackupService = None  # type: ignore[assignment]
else:  # pragma: no cover - executed when Qt is available
    __all__.extend(["ProjectRecord", "ProjectService", "BackupService"])
