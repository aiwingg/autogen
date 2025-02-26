import sys
import os
import asyncio
import aiohttp
import logging
import time
from typing import Dict, Optional

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemResponseLoader:
    def __init__(self,
                 server_address: str,
                 max_concurrency: int = 5,
                 max_submit_retries: int = 4,
                 total_timeout_in_seconds: int = 200,
                 poll_timeout: float = 180.0,
                 poll_interval: float = 5.0):
        """
        Initialize the response loader.
        
        Args:
            server_address (str): Address of the server to send requests to
            max_concurrency (int): Maximum number of concurrent requests
            max_submit_retries (int): Maximum number of retries for failed submissions
            total_timeout_in_seconds (int): Timeout for requests in seconds
            poll_timeout (float): Timeout for polling in seconds
            poll_interval (float): Interval between polling checks in seconds
        """
        if not server_address.startswith(('http://', 'https://')):
            server_address = f'http://{server_address}'
            
        self.server_address = server_address
        self.max_concurrency = max_concurrency
        self.max_submit_retries = max_submit_retries
        self.timeout = aiohttp.ClientTimeout(total=total_timeout_in_seconds)
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.session = None
        self.pending_tasks = {}  # task_id -> asyncio.Task mapping
        logger.info(
            "Initialized SystemResponseLoader with server_address=%s, "
            "poll_timeout=%.1f, poll_interval=%.1f",
            server_address, poll_timeout, poll_interval
        )
    

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        logger.debug("Created new aiohttp ClientSession")
        return self


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        logger.debug("Completed cleanup in __aexit__")


    async def __submit_task(self, test_request: dict) -> Dict:
        """
        Submit a task to the server.
        
        Args:
            test_request (dict): Request to submit
            
        Returns:
            str: Task ID
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context manager.")

        for attempt in range(self.max_submit_retries):
            try:
                sent_address = f"{self.server_address}/submit_task"
                logger.info("Submitting task to %s", sent_address)
                async with self.session.post(
                    sent_address,
                    json=test_request,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        task_id = result["task_id"]
                        # Create and store the polling task
                        polling_task = asyncio.create_task(self.__poll_task_status(task_id))
                        self.pending_tasks[task_id] = polling_task
                        logger.info("Successfully submitted task with ID %s", task_id)
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        if attempt < self.max_submit_retries - 1:
                            logger.info("Retrying submission after %d seconds", 2 ** attempt)
                            await asyncio.sleep(2 ** attempt)
                        continue
            except Exception as e:
                logger.error(f"Error during request attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_submit_retries - 1:
                    logger.info("Retrying submission after %d seconds", 2 ** attempt)
                    await asyncio.sleep(2 ** attempt)
                continue
        raise Exception(f"Failed to submit task after {self.max_submit_retries} attempts")


    async def __check_task_status(self, task_id: str) -> dict:
        """Internal method to check task status"""
        logger.debug("Checking status for task %s", task_id)
        async with self.session.get(
            f"{self.server_address}/check_task_status",
            params={"task_id": task_id}
        ) as response:
            return await response.json()


    async def __poll_task_status(self, task_id: str) -> Dict:
        """Internal method to poll task status"""
        start_time = time.time()
        logger.info("Starting to poll status for task %s", task_id)
        print("STARTED POLLING")

        while True:
            try:
                status = await self.__check_task_status(task_id)
                used_tools = []
                responses = status.get("messages", [])
                if responses:
                    last_response = responses[-1]
                    dialogue_history = last_response.get("context", [])
                    call_id_to_tool_info = {}
                    for element in dialogue_history:
                        if element['type'] == 'FunctionExecutionResultMessage':
                            for tool_result in element['content']:
                                tool_call = call_id_to_tool_info[tool_result['call_id']]
                                tool_call['result'] = tool_result['content']
                        if element['type'] == 'AssistantMessage' and isinstance(element['content'], list):
                            for tool_call in element['content']:
                                tool_call_record = {
                                    'name' : tool_call['name'],
                                    'args' : tool_call['arguments'],
                                    'id' : tool_call['id']
                                }
                                call_id_to_tool_info[tool_call['id']] = tool_call_record
                            print('current call_id_to_tool_info', call_id_to_tool_info)
                            
                    used_tools = [tool_call_record['name'] for tool_call_record in call_id_to_tool_info.values()]
                    used_tools = list(set(used_tools))

                if len(used_tools) > 0:
                    response = ""
                    if dialogue_history and isinstance(dialogue_history[-1], dict):
                        response = dialogue_history[-1].get("content", "")
                    logger.info("Task %s completed with tools %s", task_id, used_tools)
                    return {
                        "status": "completed",
                        "used_tools": used_tools,
                        "response": response,
                        "dialogue_history": responses[-1]
                    }

                elapsed = time.time() - start_time
                if elapsed > self.poll_timeout:
                    logger.warning(
                        "Task %s timed out after %.1f seconds",
                        task_id, self.poll_timeout
                    )
                    return {
                        "status": "timeout",
                        "error": f"Task timed out after {self.poll_timeout:.1f} seconds"
                    }
                
                logger.debug(
                    "Task %s still processing, checking again in %.1f seconds",
                    task_id, self.poll_interval
                )
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error("Error polling task %s: %s", task_id, str(e))
                return {"status": "error", "error": str(e)}


    async def get_result(self, task_id: str) -> Optional[Dict]:
        """
        Get the result for a specific task if it's ready.
        
        Args:
            task_id (str): Task ID to check
            
        Returns:
            Optional[Dict]: Result dictionary if ready, None if still processing
        """
        if task_id not in self.pending_tasks:
            logger.warning("Attempted to get result for unknown task %s", task_id)
            return {"status": "error", "error": "Task ID not found"}

        task = self.pending_tasks[task_id]
        if task.done():
            try:
                result = await task
                del self.pending_tasks[task_id]
                logger.info("Retrieved result for task %s: %s", task_id, result["status"])
                return result
            except Exception as e:
                del self.pending_tasks[task_id]
                logger.error("Error retrieving result for task %s: %s", task_id, str(e))
                return {"status": "error", "error": str(e)}
        logger.debug("Task %s still processing", task_id)
        return {"status": "processing"}


    async def add_request(
        self, 
        input_text: str, 
        config: Optional[Dict] = None,
        trace_name: Optional[str] = "evaluation",
    ) -> Dict:
        """Add new request to processing queue"""
        try:
            test_request = {
                "description": input_text,
                "task_id": None,
                "config": config,
                "trace_name": trace_name,
            }
            return await self.__submit_task(test_request)
        except Exception as e:
            logger.error("Error adding request: %s", str(e))
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel all pending tasks
            for task_id, task in self.pending_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self.pending_tasks.clear()
            
            # Close session
            if self.session:
                await self.session.close()
                self.session = None
                
        except Exception as e:
            logger.error("Error during cleanup: %s", str(e))