"""Logging configuration for the chess framework."""

import logging
from typing import Dict, Optional


class LoggingConfig:
    """Configuration for component-specific logging."""

    def __init__(
        self,
        level: int = logging.INFO,
        component_levels: Optional[Dict[str, int]] = None,
        format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ) -> None:
        """Initialize logging configuration.

        Args:
            level: Default logging level for all components
            component_levels: Component-specific logging levels
                Keys: 'game', 'match', 'campaign', 'tournament', 'player', 'storage'
            format_string: Log message format
        """
        self.level = level
        self.component_levels = component_levels or {}
        self.format_string = format_string

    def get_logger(self, component: str, name: str = "") -> logging.Logger:
        """Get a configured logger for a component.

        Args:
            component: Component type ('game', 'match', 'campaign', 'tournament', etc.)
            name: Optional specific name for the logger

        Returns:
            Configured logger instance
        """
        logger_name = f"collective_chess.{component}"
        if name:
            logger_name += f".{name}"

        logger = logging.getLogger(logger_name)

        # Set component-specific level if configured, otherwise use default
        level = self.component_levels.get(component, self.level)
        logger.setLevel(level)

        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(self.format_string))
            logger.addHandler(handler)

        return logger

    @classmethod
    def from_dict(cls, config: Dict[str, int]) -> "LoggingConfig":
        """Create logging config from dictionary.

        Args:
            config: Dictionary with 'default' level and optional component levels

        Returns:
            LoggingConfig instance
        """
        default_level = config.get("default", logging.INFO)
        component_levels = {k: v for k, v in config.items() if k != "default"}

        return cls(level=default_level, component_levels=component_levels)
