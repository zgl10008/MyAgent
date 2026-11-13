from __future__ import annotations

from app.core.agent.orchestrator import AgentOrchestrator, IntentContext
from app.core.agent_service import AgentService, _ToolRegistryAdapter
from app.core.tools.registry import ToolRegistry


class _ToolNames:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def list_tool_names(self) -> list[str]:
        return list(self._names)


def test_plan_execute_tool_scope_keeps_web_search():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._tools = _ToolNames(["calculator", "web_search", "rag_search"])

    names = orchestrator._safe_react_tools(IntentContext())

    assert names == ["calculator", "web_search", "rag_search"]


class _ModeRouter:
    def __init__(self) -> None:
        self._tool_names: list[str] = []


class _FunctionCallAgent:
    def __init__(self) -> None:
        self.tools = []

    def set_tools(self, tools) -> None:
        self.tools = list(tools)


class _RAGService:
    chunks_count = 12

    async def search(self, query: str, top_k: int, mode: str):
        return []


def test_late_rag_registration_refreshes_function_calling_tools():
    service = AgentService.__new__(AgentService)
    service._tool_registry = ToolRegistry()
    service._tool_adapter = _ToolRegistryAdapter(service._tool_registry)
    service._mode_router = _ModeRouter()
    service._function_call_agent = _FunctionCallAgent()

    service.set_rag_service(_RAGService())

    assert service._tool_adapter.list_tool_names() == ["rag_search"]
    assert [tool.name for tool in service._function_call_agent.tools] == ["rag_search"]
    assert service._mode_router._tool_names == ["rag_search"]
    rag_schema = service._function_call_agent.tools[0].args_schema.model_json_schema()
    top_k_schema = rag_schema["properties"]["top_k"]
    assert any(option.get("type") == "integer" for option in top_k_schema["anyOf"])
