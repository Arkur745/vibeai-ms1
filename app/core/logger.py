import os
import re
import json
import logging
from datetime import datetime
from logging import StreamHandler, Formatter

# Regular expressions to catch and redact sensitive strings (API keys, Authorization headers, passwords, secrets)
SENSITIVE_PATTERNS = [
    (re.compile(r'(?i)(authorization|auth|proxy-authorization)\s*:\s*[^\s,;]+'), r'\1: [REDACTED]'),
    (re.compile(r'(?i)(bearer)\s+[^\s,;]+'), r'\1 [REDACTED]'),
    (re.compile(r'(?i)\b(api_key|apikey|secret|token|password|pwd|client_secret|aws_secret_access_key|vibeai_internal_secret)\s*=\s*[^\s&]+'), r'\1=[REDACTED]'),
    (re.compile(r'(?i)"(api_key|apikey|secret|token|password|pwd|client_secret|aws_secret_access_key|vibeai_internal_secret)"\s*:\s*"[^"]+"'), r'"\1": "[REDACTED]"'),
    (re.compile(r'(?i)\b(vibe_secure_trust_secret_token|sk_[0-9a-zA-Z]{24,}|AIzaSy[0-9a-zA-Z_-]{33})\b'), r'[REDACTED_CREDENTIAL]')
]

def sanitize_message(message: str) -> str:
    """
    Scans the given message and redacts API keys, passwords, credentials,
    and HTTP headers to prevent sensitive data leaks in log streams.
    """
    if not isinstance(message, str):
        return str(message)
    
    for pattern, replacement in SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


class SanitizedConsoleFormatter(Formatter):
    """Custom standard console formatter that sanitizes output."""
    def format(self, record):
        original_msg = record.msg
        if isinstance(record.msg, str):
            record.msg = sanitize_message(record.msg)
        
        # Also sanitize args if logging as format string
        if record.args:
            record.args = tuple(sanitize_message(str(arg)) if isinstance(arg, str) else arg for arg in record.args)
            
        formatted = super().format(record)
        # Restore original message to avoid mutating the record for other handlers
        record.msg = original_msg
        return formatted


class StructuredJSONFormatter(logging.Formatter):
    """
    Highly-parsable JSON structured logging formatter designed for production containers.
    Formats logs into single-line JSON streams to be ingested by Grafana Loki/ELK.
    """
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "message": sanitize_message(record.getMessage())
        }
        
        # Inject standard trace IDs if attached to record (e.g. from middleware request tracing)
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
            
        # Incorporate Python stack traces if an exception occurred
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


def get_logger(name: str = "vibeai") -> logging.Logger:
    """
    Returns a configured logger with appropriate formatters and log thresholds.
    Reads environment variables:
      - JSON_LOGS: True/False (Production JSON stream format)
      - LOG_LEVEL: INFO/DEBUG/WARNING/ERROR (Default: INFO in production, DEBUG in dev)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Resolve environment settings
    env_mode = os.getenv("ENV", "development").lower()
    is_prod = env_mode == "production"
    
    json_logs = os.getenv("JSON_LOGS", "false").lower() in ("true", "1") or is_prod
    default_level = "INFO" if is_prod else "DEBUG"
    log_level_str = os.getenv("LOG_LEVEL", default_level).upper()
    
    # Resolve log level integer
    log_level = getattr(logging, log_level_str, logging.INFO)

    handler = StreamHandler()
    
    # Apply formatters
    if json_logs:
        handler.setFormatter(StructuredJSONFormatter())
    else:
        # Standard clean human-readable console style
        handler.setFormatter(SanitizedConsoleFormatter(
            "%(asctime)s %(name)s [%(levelname)s] (%(module)s:%(lineno)d) %(message)s"
        ))

    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Instantiate root global vibeai logger
logger = get_logger()
