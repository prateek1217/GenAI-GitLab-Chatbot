from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool


def build_web_search_tool() -> BaseTool:
    return TavilySearchResults(
        max_results=3,
        name="tavily_web_search",
        description=(
            "Search the web for recent GitLab news, announcements, or information "
            "not found in the handbook. Use only as a fallback after searching the handbook."
        ),
    )
