"""Specialist agents.

Each agent is a plain function with no knowledge of the graph, the HTTP
layer or each other, which keeps every one of them independently testable.
"""

from app.agents.base import AgentResponse

__all__ = ["AgentResponse"]
