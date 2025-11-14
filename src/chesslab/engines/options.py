"""Engine options for ChessLab.

Defines option types for UCI engine configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import structlog

logger = structlog.get_logger()


class Option(ABC):
    """Abstract base class for UCI engine options."""

    type: str
    name: str
    default: Any
    _value: Optional[Any] = None

    def __init__(self, name: str, default: Any) -> None:
        """Initialize an option.

        Args:
            name: The option name
            default: The default value

        Raises:
            ValueError: If default value is invalid
        """
        self.name = name
        self._value = None

        if self.is_valid(default):
            self.default = default
        else:
            raise ValueError(f"Option default '{default}' is invalid for {name}")

    @property
    def value(self) -> Any:
        """Get the current value, or default if not set."""
        if self._value is None:
            return self.default
        return self._value

    @abstractmethod
    def is_valid(self, value: Any) -> bool:
        """Check if a value is valid for this option.

        Args:
            value: The value to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def set_value(self, value: Any) -> None:
        """Set the option value.

        Args:
            value: The new value to set
        """
        if self.is_valid(value):
            self._value = value
            logger.debug(
                "Option set",
                option=self.name,
                value=value,
            )
        else:
            logger.error("Given value is invalid", option=self.name, value=value)

    @abstractmethod
    def __str__(self) -> str:
        """Return UCI-formatted option string."""
        raise NotImplementedError("Subclasses must implement __str__()")


class OptionSpin(Option):
    """Integer option with min/max bounds."""

    type = "spin"

    default: int
    _value: Optional[int] = None

    min: int
    max: int

    def __init__(self, name: str, default: int, min: int, max: int) -> None:
        """Initialize a spin option.

        Args:
            name: The option name
            default: The default value
            min: Minimum allowed value
            max: Maximum allowed value

        Raises:
            ValueError: If default is outside [min, max] range
        """
        if min > max:
            raise ValueError(f"Min ({min}) cannot be greater than max ({max})")

        self.min = min
        self.max = max
        super().__init__(name=name, default=default)

    def is_valid(self, value: Any) -> bool:
        """Check if value is an integer within [min, max]."""
        try:
            int_value = int(value)
            return self.min <= int_value <= self.max
        except (ValueError, TypeError):
            return False

    def set_value(self, value: Any) -> None:
        """Set the value, converting to int if needed."""
        try:
            int_value = int(value)
            super().set_value(int_value)
        except (ValueError, TypeError):
            logger.error("Cannot convert value to int", option=self.name, value=value)

    def __str__(self) -> str:
        """Return UCI-formatted spin option."""
        return f"option name {self.name} type spin default {self.default} min {self.min} max {self.max}"


class OptionCheck(Option):
    """Boolean option."""

    type = "check"

    default: bool
    _value: Optional[bool] = None

    def __init__(self, name: str, default: bool) -> None:
        """Initialize a check option.

        Args:
            name: The option name
            default: The default boolean value
        """
        super().__init__(name=name, default=default)

    def is_valid(self, value: Any) -> bool:
        """Check if value can be converted to bool."""
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.lower() in ("true", "false")
        return False

    def set_value(self, value: Any) -> None:
        """Set the value, converting to bool if needed."""
        if isinstance(value, bool):
            super().set_value(value)
        elif isinstance(value, str):
            bool_value = value.lower() == "true"
            super().set_value(bool_value)
        else:
            logger.error("Cannot convert value to bool", option=self.name, value=value)

    def __str__(self) -> str:
        """Return UCI-formatted check option."""
        return f"option name {self.name} type check default {'true' if self.default else 'false'}"


class OptionCombo(Option):
    """String option with predefined choices."""

    type = "combo"

    default: str
    _value: Optional[str] = None
    vars: List[str]

    def __init__(self, name: str, default: str, vars: List[str]) -> None:
        """Initialize a combo option.

        Args:
            name: The option name
            default: The default choice
            vars: List of allowed choices

        Raises:
            ValueError: If default not in vars or vars is empty
        """
        if not vars:
            raise ValueError("Combo option must have at least one choice")
        if default not in vars:
            raise ValueError(f"Default '{default}' not in allowed choices: {vars}")

        self.vars = vars
        super().__init__(name=name, default=default)

    def is_valid(self, value: Any) -> bool:
        """Check if value is one of the allowed choices."""
        return str(value) in self.vars

    def set_value(self, value: Any) -> None:
        """Set the value if it's a valid choice."""
        str_value = str(value)
        super().set_value(str_value)

    def __str__(self) -> str:
        """Return UCI-formatted combo option."""
        vars_str = " ".join(f"var {v}" for v in self.vars)
        return f"option name {self.name} type combo default {self.default} {vars_str}"


class OptionString(Option):
    """Freeform string option."""

    type = "string"

    default: str
    _value: Optional[str] = None

    def __init__(self, name: str, default: str = "") -> None:
        """Initialize a string option.

        Args:
            name: The option name
            default: The default string value
        """
        super().__init__(name=name, default=default)

    def is_valid(self, value: Any) -> bool:
        """Any value can be converted to string."""
        return True

    def set_value(self, value: Any) -> None:
        """Set the value, converting to string."""
        str_value = str(value)
        super().set_value(str_value)

    def __str__(self) -> str:
        """Return UCI-formatted string option."""
        return f"option name {self.name} type string default {self.default}"


class OptionButton(Option):
    """Button option (triggers an action, has no value)."""

    type = "button"

    def __init__(self, name: str) -> None:
        """Initialize a button option.

        Args:
            name: The option name
        """
        # Buttons have no default value
        self.name = name
        self._value = None

    def is_valid(self, value: Any) -> bool:
        """Buttons don't have values."""
        return True

    def set_value(self, value: Any) -> None:
        """Button pressed - subclasses should override to handle action."""
        logger.debug("Button pressed", button=self.name)

    @property
    def value(self) -> None:
        """Buttons have no value."""
        return None

    def __str__(self) -> str:
        """Return UCI-formatted button option."""
        return f"option name {self.name} type button"
