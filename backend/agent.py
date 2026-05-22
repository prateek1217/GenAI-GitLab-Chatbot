import os
from typing import Annotated, Literal
from dotenv import load_dotenv
from functools import partial

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt
import psycopg
from typing_extensions import TypedDict

from backend.tools.retriever import build_retriever_tool
from backend.tools.web_search import build_web_search_tool

load_dotenv()

# Injected at the start of agent invocation to set behavior and tone
SYSTEM_PROMPT = """You are GitLab Assistant. Answer any question related to GitLab using your tools.
Always search the handbook first, and if the handbook doesn't have enough information, search the web.
Provide a complete, helpful answer every time. Cite sources at the end."""

# Returned when the user asks something unrelated to GitLab
OFF_TOPIC_RESPONSE = (
    "I'm GitLab Assistant and I'm only able to answer questions about GitLab — "
    "its handbook, culture, engineering practices, product direction, and company policies. "
    "Please ask me something GitLab-related and I'll be happy to help!"
)

# Returned when the user declines the web search (case of Human in the Loop)
WEB_SEARCH_DECLINED_RESPONSE = (
    "Understood! I searched the GitLab handbook but couldn't find enough information "
    "to fully answer your question. Try asking about a more specific GitLab topic — "
    "like its values, engineering practices, hiring process, or product direction."
)


# Shared state passed between every node in the graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  
    is_gitlab_related: bool                               


# Pydantic model for structured LLM output in the relevance gate
class RelevanceResult(BaseModel):
    is_gitlab_related: bool


#  here we are building llm chat instance 
def build_llm(streaming: bool = True) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        streaming=streaming,
        temperature=0.1,  # low temperature = consistent, factual answers
    )


#  node definitions 


# this is our node which checks the relevance of user's query which is related to Gitlab or not
def relevance_gate_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> dict:
    structured_llm = llm.with_structured_output(RelevanceResult)
    result: RelevanceResult = structured_llm.invoke([
        SystemMessage(content=(
            "You are a topic classifier for a GitLab assistant. "
            "You will receive the full conversation history. "
            "Return is_gitlab_related=true if the latest user message is about GitLab "
            "(its handbook, culture, engineering, hiring, product, or direction) "
            "OR if it is a follow-up to a GitLab topic already discussed "
            "(e.g. 'explain this', 'summarize', 'tell me more'). "
            "Return is_gitlab_related=false only if it is completely unrelated to GitLab "
            "and not a follow-up to the current conversation about gitlab."
        )),
        *state["messages"], # conversation history
    ])
    return {"is_gitlab_related": result.is_gitlab_related}


def agent_node(state: AgentState, llm) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_approval_node(state: AgentState) -> dict:
    # Human-in-the-loop: pauses the graph before web search runs and asks the user
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    needs_web_search = any(tc["name"] == "tavily_web_search" for tc in tool_calls)

    if needs_web_search:
        # interrupt() checkpoints the graph state and pauses execution here until human approves or reject the req.
        decision = interrupt("web_search_approval")
        if decision != "yes":
            return {"messages": [AIMessage(content=WEB_SEARCH_DECLINED_RESPONSE)]}

    return {}


def off_topic_node(state: AgentState) -> dict:
    # Short-circuits to a static refusal message without calling the LLM
    return {"messages": [AIMessage(content=OFF_TOPIC_RESPONSE)]}


#  Routing nodes

def route_after_relevance_gate(state: AgentState) -> Literal["agent", "off_topic"]:
    return "agent" if state.get("is_gitlab_related", True) else "off_topic"


def route_after_agent(state: AgentState) -> Literal["tool_approval", "__end__"]:
    # If the agent produced tool calls, then our flow will go to tool_approval.
    # Otherwise the agent gave a final answer — end the graph.
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tool_approval"
    return END


def route_after_tool_approval(state: AgentState) -> Literal["tools", "__end__"]:
    # If tool_approval returned {},  it means "approved" , the last message still has tool_calls -> run tools.
    # If tool_approval added a decline AIMessage, last message has no tool_calls -> end the flow.
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


#  here we are building graph , its nodes , its edges etc

def build_graph():
    tools = [build_retriever_tool(), build_web_search_tool()]

    
    llm_with_tools = build_llm(streaming=True).bind_tools(tools)
    
    relevance_llm = build_llm(streaming=False)

    # this postgresSaver persists the graph state (messages) in Supabase after each node
    db_conn = psycopg.connect(os.getenv("DATABASE_URL"), autocommit=True)
    checkpointer = PostgresSaver(db_conn)
    checkpointer.setup()  

    g = StateGraph(AgentState)

    g.add_node("relevance_gate", partial(relevance_gate_node, llm=relevance_llm))
    g.add_node("agent", partial(agent_node, llm=llm_with_tools))
    g.add_node("tool_approval", tool_approval_node)
    g.add_node("tools", ToolNode(tools))  
    g.add_node("off_topic", off_topic_node)

    # node connections 
    g.add_edge(START, "relevance_gate")
    g.add_conditional_edges("relevance_gate", route_after_relevance_gate)
    g.add_conditional_edges("agent", route_after_agent)
    g.add_conditional_edges("tool_approval", route_after_tool_approval)
    g.add_edge("tools", "agent")  
    g.add_edge("off_topic", END)

    return g.compile(checkpointer=checkpointer), db_conn


graph, db_conn = build_graph()
