"""Utilities for dataclasses."""


class IgnoreNone:
    """Ignore None values when setting attributes.
    This allows using "None" to default to dataclass defaults."""

    def __setattr__(self, name, value):
        if value is None:
            return
        super().__setattr__(name, value)
