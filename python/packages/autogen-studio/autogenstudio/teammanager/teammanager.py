import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Callable, List, Optional, Union

import aiofiles
import yaml
from autogen_agentchat.base import TaskResult, Team
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_core import EVENT_LOGGER_NAME, CancellationToken, Component, ComponentModel
from autogen_core.logging import LLMCallEvent
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent, ToolCallSummaryMessage, MemoryQueryEvent, UserInputRequestedEvent
from autogen_agentchat.base import TaskResult
from autogen_core import FunctionCall
from autogen_core.models import FunctionExecutionResult
from autogen_core.memory import MemoryContent
import random
from ..datamodel.types import LLMCallEventMessage, TeamResult

logger = logging.getLogger(__name__)

use_agents = False

messages = [
    (
        TextMessage(content="Checking the information about the customer in the database", source="manager_agent"),
        0.0
    ),
    (
        ToolCallRequestEvent(content=[FunctionCall(name="query_customer_info_rag", arguments=json.dumps({"customer_description": "John from the meatstore on placeholder st."}), id="12fqnpweif23")], source="manager_agent"),
        0.7 + random.random() * 0.3
    ),
    (
        ToolCallExecutionEvent(content=[FunctionExecutionResult(content="John's phone number is 1234567890. He is a regular customer.", call_id="12fqnpweif23")], source="tool_agent"),
        0.5 + random.random() * 0.3
    ),
    (
        ToolCallRequestEvent(content=[FunctionCall(name="start a phone call", arguments=json.dumps({"phone_number": "1234567890", "message": "the delivery will be delayed by 1 day. The new delivery date should be agreed upon."}), id="phone_call_1")], source="manager_agent"),
        0.5 + random.random() * 0.3
    ),
    (
        ToolCallExecutionEvent(content=[FunctionExecutionResult(content="""The phone call has been performed. The call transcript:\n
<div class="retellai-dialog"><b>Operator</b>: Hello, I would like to inform that your order will be delayed</div>\n
<div class="retellai-dialog"><b>Customer</b>: I want my money back</b></div>\n
        """, call_id="phone_call_1")], source="tool_agent"),
        1.0
    ),
    (
        UserInputRequestedEvent(request_id="user_request1", source="manager_agent"),
        0.1
    ),
    (
        TextMessage(content="The memory was updated", source="memory_agent"),
        1.1
    )
]

class RunEventLogger(logging.Handler):
    """Event logger that queues LLMCallEvents for streaming"""

    def __init__(self):
        super().__init__()
        self.events = asyncio.Queue()

    def emit(self, record: logging.LogRecord):
        if isinstance(record.msg, LLMCallEvent):
            self.events.put_nowait(LLMCallEventMessage(content=str(record.msg)))


class TeamManager:
    """Manages team operations including loading configs and running teams"""

    @staticmethod
    async def load_from_file(path: Union[str, Path]) -> dict:
        """Load team configuration from JSON/YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        async with aiofiles.open(path) as f:
            content = await f.read()
            if path.suffix == ".json":
                return json.loads(content)
            elif path.suffix in (".yml", ".yaml"):
                return yaml.safe_load(content)
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    async def load_from_directory(directory: Union[str, Path]) -> List[dict]:
        """Load all team configurations from a directory"""
        directory = Path(directory)
        configs = []
        valid_extensions = {".json", ".yaml", ".yml"}

        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in valid_extensions:
                try:
                    config = await TeamManager.load_from_file(path)
                    configs.append(config)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        return configs

    async def _create_team(
        self, team_config: Union[str, Path, dict, ComponentModel], input_func: Optional[Callable] = None
    ) -> Component:
        """Create team instance from config"""
        if isinstance(team_config, (str, Path)):
            config = await self.load_from_file(team_config)
        elif isinstance(team_config, dict):
            config = team_config
        else:
            config = team_config.model_dump()

        team = Team.load_component(config)

        for agent in team._participants:
            if hasattr(agent, "input_func"):
                agent.input_func = input_func

        return team

    async def run_stream(
        self,
        task: str,
        team_config: Union[str, Path, dict, ComponentModel],
        input_func: Optional[Callable] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[AgentEvent | ChatMessage | LLMCallEvent, ChatMessage, TeamResult], None]:
        """Stream team execution results"""
        start_time = time.time()
        team = None

        # Setup logger correctly
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        llm_event_logger = RunEventLogger()
        logger.handlers = [llm_event_logger]  # Replace all handlers

        try:
            team = await self._create_team(team_config, input_func)

            if use_agents:
                async for message in team.run_stream(task=task, cancellation_token=cancellation_token):
                    if cancellation_token and cancellation_token.is_cancelled():
                        break

                if isinstance(message, TaskResult):
                    yield TeamResult(task_result=message, usage="", duration=time.time() - start_time)
                else:
                    yield message

                ### Check for any LLM events
                while not llm_event_logger.events.empty():
                    event = await llm_event_logger.events.get()
                    yield event
            
            else:
                for (message, execution_time) in messages:
                    await asyncio.sleep(execution_time)
                    if isinstance(message, UserInputRequestedEvent):
                        print("User input requested")
                        res = await input_func(prompt="What should be done?")
                        message = TextMessage(content=res, source="user")
                    
                    yield message

        finally:
            # Cleanup - remove our handler
            logger.handlers.remove(llm_event_logger)

            # Ensure cleanup happens
            if team and hasattr(team, "_participants"):
                for agent in team._participants:
                    if hasattr(agent, "close"):
                        await agent.close()

    async def run(
        self,
        task: str,
        team_config: Union[str, Path, dict, ComponentModel],
        input_func: Optional[Callable] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> TeamResult:
        """Run team synchronously"""
        start_time = time.time()
        team = None

        try:
            team = await self._create_team(team_config, input_func)
            result = await team.run(task=task, cancellation_token=cancellation_token)

            return TeamResult(task_result=result, usage="", duration=time.time() - start_time)

        finally:
            if team and hasattr(team, "_participants"):
                for agent in team._participants:
                    if hasattr(agent, "close"):
                        await agent.close()
