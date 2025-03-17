"""
Need Identifier component.

Detects when the agent requires external tools based on user queries or task requirements.
"""

from atia.need_identifier.identifier import NeedIdentifier, ToolNeed

__all__ = ["NeedIdentifier", "ToolNeed"]
