class ModtranError(Exception):
    """Base exception for all MODTRAN wrapper errors."""


class ModtranRunError(ModtranError):
    """Raised when the MODTRAN executable fails."""

    def __init__(self, message, returncode=None, stderr=None):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class ModtranParseError(ModtranError):
    """Raised when an output file cannot be parsed."""


class ModtranConfigError(ModtranError):
    """Raised for configuration problems (missing executable, bad paths, etc.)."""


class ModtranVersionError(ModtranError):
    """Raised when a feature is unavailable in the detected MODTRAN version."""
