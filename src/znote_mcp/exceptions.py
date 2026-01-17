"""Custom exceptions for the Zettelkasten MCP server.

Provides a structured exception hierarchy with error codes and
machine-readable error information for better error handling.
"""
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCode(Enum):
    """Error codes for machine-readable error identification."""

    # Note errors (1xxx)
    NOTE_NOT_FOUND = 1001
    NOTE_VALIDATION_FAILED = 1002
    NOTE_ALREADY_EXISTS = 1003
    NOTE_TITLE_REQUIRED = 1004
    NOTE_CONTENT_REQUIRED = 1005

    # Link errors (2xxx)
    LINK_INVALID = 2001
    LINK_ALREADY_EXISTS = 2002
    LINK_NOT_FOUND = 2003
    LINK_SELF_REFERENCE = 2004

    # Tag errors (3xxx)
    TAG_NOT_FOUND = 3001
    TAG_INVALID = 3002

    # Storage errors (4xxx)
    STORAGE_READ_FAILED = 4001
    STORAGE_WRITE_FAILED = 4002
    STORAGE_DELETE_FAILED = 4003
    STORAGE_CONNECTION_FAILED = 4004
    DATABASE_CORRUPTED = 4005
    DATABASE_RECOVERY_FAILED = 4006
    FTS_CORRUPTED = 4007

    # Bulk operation errors (45xx)
    BULK_OPERATION_FAILED = 4501
    BULK_OPERATION_PARTIAL = 4502
    BULK_OPERATION_EMPTY_INPUT = 4503

    # Search errors (5xxx)
    SEARCH_FAILED = 5001
    SEARCH_INVALID_QUERY = 5002

    # Configuration errors (6xxx)
    CONFIG_INVALID = 6001
    CONFIG_MISSING = 6002

    # Validation errors (7xxx)
    VALIDATION_FAILED = 7001
    INVALID_NOTE_TYPE = 7002
    INVALID_LINK_TYPE = 7003
    INVALID_DIRECTION = 7004
    PATH_TRAVERSAL_DETECTED = 7005


class ZettelkastenError(Exception):
    """Base exception for all Zettelkasten errors.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION_FAILED,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "details": self.details
        }

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"[{self.code.name}] {self.message} ({detail_str})"
        return f"[{self.code.name}] {self.message}"


class NoteNotFoundError(ZettelkastenError):
    """Raised when a note cannot be found."""

    def __init__(self, note_id: str, message: Optional[str] = None):
        super().__init__(
            message or f"Note with ID '{note_id}' not found",
            code=ErrorCode.NOTE_NOT_FOUND,
            details={"note_id": note_id}
        )
        self.note_id = note_id


class NoteValidationError(ZettelkastenError):
    """Raised when note data fails validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        code: ErrorCode = ErrorCode.NOTE_VALIDATION_FAILED
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate for safety

        super().__init__(message, code=code, details=details)
        self.field = field
        self.value = value


class LinkError(ZettelkastenError):
    """Raised for link-related errors."""

    def __init__(
        self,
        message: str,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        link_type: Optional[str] = None,
        code: ErrorCode = ErrorCode.LINK_INVALID
    ):
        details = {}
        if source_id:
            details["source_id"] = source_id
        if target_id:
            details["target_id"] = target_id
        if link_type:
            details["link_type"] = link_type

        super().__init__(message, code=code, details=details)
        self.source_id = source_id
        self.target_id = target_id
        self.link_type = link_type


class TagError(ZettelkastenError):
    """Raised for tag-related errors."""

    def __init__(
        self,
        message: str,
        tag_name: Optional[str] = None,
        code: ErrorCode = ErrorCode.TAG_INVALID
    ):
        details = {}
        if tag_name:
            details["tag_name"] = tag_name

        super().__init__(message, code=code, details=details)
        self.tag_name = tag_name


class StorageError(ZettelkastenError):
    """Raised for storage/persistence errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        path: Optional[str] = None,
        code: ErrorCode = ErrorCode.STORAGE_READ_FAILED,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if path:
            # Don't expose full paths in error messages for security
            details["path_hint"] = path.split("/")[-1] if "/" in path else path
        if original_error:
            details["original_error"] = str(original_error)[:200]

        super().__init__(message, code=code, details=details)
        self.operation = operation
        self.path = path
        self.original_error = original_error


class DatabaseCorruptionError(StorageError):
    """Raised when database corruption is detected.

    This exception indicates that the SQLite database (or FTS5 index)
    has become corrupted and may need to be rebuilt from source files.

    Attributes:
        recovered: Whether auto-recovery was successful
        backup_path: Path to the backup of the corrupted database
    """

    def __init__(
        self,
        message: str,
        recovered: bool = False,
        backup_path: Optional[str] = None,
        code: ErrorCode = ErrorCode.DATABASE_CORRUPTED,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            operation="database_check",
            code=code,
            original_error=original_error
        )
        self.recovered = recovered
        self.backup_path = backup_path
        self.details["recovered"] = recovered
        if backup_path:
            self.details["backup_path"] = backup_path


class SearchError(ZettelkastenError):
    """Raised for search-related errors."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        code: ErrorCode = ErrorCode.SEARCH_FAILED
    ):
        details = {}
        if query:
            details["query"] = query[:100]  # Truncate for safety

        super().__init__(message, code=code, details=details)
        self.query = query


class ConfigurationError(ZettelkastenError):
    """Raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        code: ErrorCode = ErrorCode.CONFIG_INVALID
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key

        super().__init__(message, code=code, details=details)
        self.config_key = config_key


class ValidationError(ZettelkastenError):
    """Raised for general validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        code: ErrorCode = ErrorCode.VALIDATION_FAILED
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]

        super().__init__(message, code=code, details=details)
        self.field = field
        self.value = value


class BulkOperationError(ZettelkastenError):
    """Raised for bulk operation errors.

    Provides detailed information about which items succeeded and failed.

    Attributes:
        operation: Name of the bulk operation (e.g., "bulk_create", "bulk_delete")
        total_count: Total number of items attempted
        success_count: Number of items that succeeded
        failed_ids: List of IDs that failed (full list, not truncated)
        original_error: The underlying exception if applicable

    Note:
        The `details` dict contains `failed_ids` truncated to 10 items for
        safe serialization. Access `self.failed_ids` for the complete list.
    """

    def __init__(
        self,
        message: str,
        operation: str,
        total_count: int = 0,
        success_count: int = 0,
        failed_ids: Optional[List[str]] = None,
        code: ErrorCode = ErrorCode.BULK_OPERATION_FAILED,
        original_error: Optional[Exception] = None
    ):
        # Validate invariants
        if total_count < 0:
            raise ValueError("total_count must be non-negative")
        if success_count < 0:
            raise ValueError("success_count must be non-negative")
        if success_count > total_count:
            raise ValueError("success_count cannot exceed total_count")

        details = {
            "operation": operation,
            "total_count": total_count,
            "success_count": success_count,
            "failed_count": total_count - success_count
        }
        if failed_ids:
            details["failed_ids"] = failed_ids[:10]  # Truncate for safety
        if original_error:
            details["original_error"] = str(original_error)[:200]

        super().__init__(message, code=code, details=details)
        self.operation = operation
        self.total_count = total_count
        self.success_count = success_count
        # Defensive copy to prevent external mutation
        self.failed_ids: List[str] = list(failed_ids) if failed_ids else []
        self.original_error = original_error

    @property
    def failed_count(self) -> int:
        """Number of items that failed (computed from total - success)."""
        return self.total_count - self.success_count
