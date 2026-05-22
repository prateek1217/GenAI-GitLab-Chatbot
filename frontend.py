import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.types import Command
from backend.agent import graph, db_conn, OFF_TOPIC_RESPONSE

#  Page config 
st.set_page_config(
    page_title="GitLab Handbook Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling ( for UI )
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .gitlab-header {
        background: linear-gradient(135deg, #e24329 0%, #fc6d26 60%, #fca326 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .gitlab-header h1 { margin: 0; font-size: 1.8rem; }
    .gitlab-header p  { margin: 0.4rem 0 0; opacity: 0.9; font-size: 0.95rem; }
    .status-badge {
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
    }
    .badge-searching { background: #fff3cd; color: #856404; }
    div[data-testid="stSidebarContent"] button { text-align: left !important; }
    .thinking-dots span {
        display: inline-block;
        width: 8px; height: 8px;
        margin: 0 3px;
        background: #fc6d26;
        border-radius: 50%;
        animation: thinking 1.2s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes thinking {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40%            { transform: scale(1.0); opacity: 1.0; }
    }
</style>
""", unsafe_allow_html=True)

#  Header of our chat
st.markdown("""
<div class="gitlab-header">
    <h1>🦊 GitLab Handbook Assistant</h1>
    <p>Ask me anything about GitLab's culture, engineering practices, product direction, or company policies.</p>
</div>
""", unsafe_allow_html=True)


# DB helpers 

def setup_conversations_table():
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_titles (
            thread_id TEXT PRIMARY KEY,
            title     TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

def save_conversation_title(thread_id: str, title: str):
    db_conn.execute(
        "INSERT INTO conversation_titles (thread_id, title) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (thread_id, title[:60] + "..." if len(title) > 60 else title),
    )

def load_conversations() -> list[dict]:
    rows = db_conn.execute(
        "SELECT thread_id, title, created_at FROM conversation_titles ORDER BY created_at DESC"
    ).fetchall()
    return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

def load_thread_messages(thread_id: str) -> list:
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values:
        return []
    raw = state.values.get("messages", [])
    # LangGraph state contains intermediate messages (tool-call AIMessages, ToolMessages).
    # so we will only keep human messages and final AI responses (AIMessages without tool_calls).
    return [
        m for m in raw
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    ]

def delete_conversation(thread_id: str):
    db_conn.execute("DELETE FROM conversation_titles WHERE thread_id = %s", (thread_id,))
    db_conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
    db_conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
    db_conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))


setup_conversations_table()

# Session state 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_saved" not in st.session_state:
    st.session_state.conversation_saved = False
if "awaiting_web_search_approval" not in st.session_state:
    st.session_state.awaiting_web_search_approval = False


#  Sidebar 
with st.sidebar:
    st.markdown("### 🦊 GitLab Assistant")

    if st.button("+ New Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.conversation_saved = False
        st.session_state.awaiting_web_search_approval = False
        st.rerun()

    st.divider()
    st.markdown("#### Conversations")

    conversations = load_conversations()

    if not conversations:
        st.caption("No conversations yet. Start chatting!")
    else:
        for conv in conversations:
            col1, col2 = st.columns([5, 1])
            is_active = conv["thread_id"] == st.session_state.thread_id
            label = f"{'▶ ' if is_active else ''}{conv['title']}"
            with col1:
                if st.button(label, key=f"conv_{conv['thread_id']}", use_container_width=True):
                    if conv["thread_id"] != st.session_state.thread_id:
                        st.session_state.thread_id = conv["thread_id"]
                        st.session_state.messages = load_thread_messages(conv["thread_id"])
                        st.session_state.conversation_saved = True
                        st.session_state.awaiting_web_search_approval = False
                        st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{conv['thread_id']}"):
                    delete_conversation(conv["thread_id"])
                    if conv["thread_id"] == st.session_state.thread_id:
                        st.session_state.thread_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.session_state.conversation_saved = False
                        st.session_state.awaiting_web_search_approval = False
                    st.rerun()


#  displaying chat history 
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# Streaming helpers 

def _stream_chunks(graph_input, config: dict):
    """Run the graph and yield (text_chunk, tools_used, done). Detects interrupts."""
    tool_calls_made = []
    got_content = False

    for chunk, metadata in graph.stream(graph_input, config=config, stream_mode="messages"):
        node = metadata.get("langgraph_node", "")

        if node == "tools" and hasattr(chunk, "name") and chunk.name:
            if chunk.name not in tool_calls_made:
                tool_calls_made.append(chunk.name)

        if node == "agent" and isinstance(chunk, AIMessageChunk) and chunk.content:
            got_content = True
            yield chunk.content, tool_calls_made, False

    # Check if graph paused on an interrupt (human in the loop)
    state = graph.get_state(config)
    if state.next:
        # Graph is paused — signal the UI to show the approval prompt
        yield None, tool_calls_made, "INTERRUPTED"
        return

    # off_topic / declined web search: static AIMessage not captured as chunks
    if not got_content:
        messages = state.values.get("messages", []) if state and state.values else []
        if messages and hasattr(messages[-1], "content") and messages[-1].content:
            yield messages[-1].content, tool_calls_made, False

    yield "", tool_calls_made, True


def _render_stream(graph_input, config: dict) -> tuple[str, bool]:
    """Render a streaming response in the current chat context. Returns (full_text, interrupted)."""
    tool_status = st.empty()
    response_placeholder = st.empty()
    full_text = ""
    tools_used = []

    response_placeholder.markdown(
        '<div class="thinking-dots"><span></span><span></span><span></span></div>',
        unsafe_allow_html=True,
    )

    for text_chunk, tools_so_far, status in _stream_chunks(graph_input, config):
        if status == "INTERRUPTED":
            tool_status.empty()
            response_placeholder.empty()
            return full_text, True

        if tools_so_far and tools_so_far != tools_used:
            tools_used = tools_so_far
            tool_labels = {
                "search_gitlab_handbook": "Searching GitLab Handbook...",
                "tavily_web_search": "Searching the web...",
            }
            label = tool_labels.get(tools_used[-1], f"Running {tools_used[-1]}...")
            tool_status.markdown(
                f'<span class="status-badge badge-searching">⚙ {label}</span>',
                unsafe_allow_html=True,
            )

        if text_chunk:
            full_text += text_chunk
            response_placeholder.markdown(full_text + "▌")

        if status is True:
            tool_status.empty()
            response_placeholder.markdown(full_text)

    if not full_text:
        full_text = OFF_TOPIC_RESPONSE
        response_placeholder.markdown(full_text)

    return full_text, False


# Human-in-the-loop: web search approval 
if st.session_state.awaiting_web_search_approval:
    with st.chat_message("assistant"):
        st.markdown(
            "📚 I searched the GitLab handbook but couldn't find enough information.\n\n"
            "**Should I also search the web for more context?**"
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✓ Yes, search the web", type="primary", use_container_width=True):
                st.session_state.awaiting_web_search_approval = False
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                with st.chat_message("assistant"):
                    full_text, interrupted = _render_stream(Command(resume="yes"), config)
                if not interrupted:
                    st.session_state.messages.append(AIMessage(content=full_text))
                st.rerun()
        with col2:
            if st.button("✗ No, handbook only", use_container_width=True):
                st.session_state.awaiting_web_search_approval = False
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                with st.chat_message("assistant"):
                    full_text, _ = _render_stream(Command(resume="no"), config)
                st.session_state.messages.append(AIMessage(content=full_text))
                st.rerun()


#  Chat input 
prompt = st.chat_input(
    "Ask about GitLab's handbook, culture, or direction...",
    disabled=st.session_state.awaiting_web_search_approval,  # lock input while approval is pending
)

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    graph_input = {"messages": [HumanMessage(content=prompt)], "is_gitlab_related": False}

    with st.chat_message("assistant"):
        full_text, interrupted = _render_stream(graph_input, config)

    if interrupted:
        # Graph paused at tool_approval_node —> show approval prompt on next render
        st.session_state.awaiting_web_search_approval = True
        st.rerun()
    else:
        st.session_state.messages.append(AIMessage(content=full_text))

    # Save conversation title on first message
    if not st.session_state.conversation_saved:
        save_conversation_title(st.session_state.thread_id, prompt)
        st.session_state.conversation_saved = True
        st.rerun()
