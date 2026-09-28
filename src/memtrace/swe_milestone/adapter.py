"""SWE-Milestone-only Codex adapter extensions."""

from __future__ import annotations

from ..harness.adapter import CodexHarnessAdapter
from ..harness.memory_tools import (
    CODE_GRAPH_SEARCH_TOOL,
    code_graph_search_dynamic_tool,
)


class SweMilestoneCodexHarnessAdapter(CodexHarnessAdapter):
    """Expose bounded Rich Graph search only to SWE-Milestone multilang runs."""

    _RICH_NAVIGATION_GUIDANCE = (
        "\n\nSWE-Milestone navigation: when the current Route Card or repository "
        "search does not identify the implementation symbol or its related tests, call "
        "search_code_graph once with the concrete requirement or symbol name. Treat the "
        "bounded result as a navigation hint, then inspect or edit the returned current-"
        "revision code surface. Do not repeat an unchanged graph query and do not treat "
        "graph relations as verification evidence."
    )

    def thread_parameters(
        self,
        *,
        developer_instructions: str,
    ) -> dict[str, object]:
        scoped_instructions = developer_instructions
        if self._RICH_NAVIGATION_GUIDANCE.strip() not in scoped_instructions:
            scoped_instructions += self._RICH_NAVIGATION_GUIDANCE
        result = dict(
            super().thread_parameters(
                developer_instructions=scoped_instructions,
            )
        )
        tools = list(result.get("dynamicTools", ()))
        if not any(
            isinstance(tool, dict) and tool.get("name") == CODE_GRAPH_SEARCH_TOOL
            for tool in tools
        ):
            tools.append(code_graph_search_dynamic_tool())
        result["dynamicTools"] = tools
        return result
