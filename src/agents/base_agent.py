"""
Abstract base class for all modality agents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base class for all modality agents in the system."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the agent with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Process input data and return structured output.

        Args:
            input_data: Input data to process

        Returns:
            Processed output specific to the agent type
        """
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Return the name of this agent for logging/tracking.

        Returns:
            Agent name as string
        """
        pass

    def __repr__(self) -> str:
        return f"{self.get_agent_name()}(config={self.config})"
