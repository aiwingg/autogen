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
from ..datamodel.types import LLMCallEventMessage, TeamResult
from .system_response_loader import SystemResponseLoader
import aiohttp

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.response_loader = None
        self.server_address = "http://localhost:51005"
        # self.server_address = "http://158.160.160.132:51005"
        logger.info(f"Initialized TeamManager with server address: {self.server_address}")

    async def initialize_loader(self):
        """Initialize the SystemResponseLoader if not already initialized"""
        if self.response_loader is None:
            try:
                async with aiohttp.ClientSession() as session:
                    try:
                        logger.info(f"Attempting to connect to server at {self.server_address}")
                        async with session.get(
                            f"{self.server_address}/health", 
                            timeout=5,
                            ssl=False
                        ) as response:
                            if response.status == 200:
                                logger.info(f"Successfully connected to server at {self.server_address}")
                            else:
                                logger.warning(f"Server responded with status {response.status}")
                                logger.warning(f"Response body: {await response.text()}")
                    except Exception as e:
                        logger.error(f"Connection error: {str(e)}")
                        raise ConnectionError(f"Could not connect to server at {self.server_address}")
                        
                self.response_loader = SystemResponseLoader(
                    server_address=self.server_address,
                    max_concurrency=5,
                    poll_timeout=180.0,
                    poll_interval=5.0
                )
                await self.response_loader.__aenter__()
            except Exception as e:
                logger.error(f"Error initializing loader: {str(e)}")
                raise

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
        start_time = time.time()
        previous_task_id = None
        dialogue_history = []  # Накапливаем историю диалога

        try:
            await self.initialize_loader()
            
            while True:
                if cancellation_token and cancellation_token.is_cancelled():
                    break

                result = await self.response_loader.add_request(
                    input_text=task,
                    config={
                        "previous_task_id": previous_task_id,
                        "dialogue_history": dialogue_history
                    } if previous_task_id else None
                )
                
                current_task_id = result.get("task_id")
                if not current_task_id:
                    raise ValueError("Failed to get task_id from server")
                
                logger.info(f"Processing message with task_id: {current_task_id} (previous: {previous_task_id})")

                while True:
                    result = await self.response_loader.get_result(current_task_id)
                    if result["status"] == "completed":
                        logger.info(f"Got result {result}")
                        
                        if result.get("dialogue_history", {}).get("context"):
                            new_messages = result["dialogue_history"]["context"]
                            dialogue_history.extend(new_messages)
                        
                        yield TextMessage(
                            content=result.get("response", "No response"),
                            source="prod_system"
                        )

                        dialogue_history = result.get("dialogue_history", {}).get("context", [])
                        if dialogue_history:
                            for msg in dialogue_history:
                                msg_type = msg.get("type")
                                content = msg.get("content")

                                if msg_type == "AssistantMessage":
                                    if isinstance(content, str):
                                        yield TextMessage(
                                            content=content,
                                            source="assistant"
                                        )
                                    elif isinstance(content, list):
                                        tool_call = content[0]
                                        yield ToolCallRequestEvent(
                                            content=content,
                                            function_name=tool_call.get("name"),
                                            arguments=tool_call.get("arguments"),
                                            call_id=tool_call.get("id"),
                                            source="assistant"
                                        )

                                elif msg_type == "FunctionExecutionResultMessage":
                                    if isinstance(content, list):
                                        tool_result = content[0]
                                        yield ToolCallExecutionEvent(
                                            content=content,
                                            function_name="tool_execution",
                                            result=FunctionExecutionResult(
                                                content=tool_result.get("content"),
                                                call_id=tool_result.get("call_id")
                                            ),
                                            call_id=tool_result.get("call_id"),
                                            source="system"
                                        )

                        if input_func:
                            yield UserInputRequestedEvent(
                                request_id=current_task_id,
                                source="system"
                            )
                            task = await input_func(prompt="Continue the conversation:")
                            if task.lower() in ["exit", "quit", "bye"]:
                                return
                            
                            previous_task_id = current_task_id
                            break
                            
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in run_stream: {str(e)}")
            yield TextMessage(
                content=f"Error occurred: {str(e)}",
                source="system"
            )
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        if self.response_loader:
            await self.response_loader.cleanup()
            self.response_loader = None

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
