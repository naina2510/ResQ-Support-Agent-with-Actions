"""
ResQ - support agent for the ShopEase store.

Answers policy questions from the knowledge base (RAG), looks up orders from
the Excel database, and can raise a support ticket after the user confirms it.

Run with: streamlit run app.py
"""

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import tools
from rag import get_retriever
from tools import ALL_TOOLS, commit_pending_ticket, discard_pending_ticket

st.set_page_config(
    page_title="ResQ Support Agent",
    layout="centered",
    initial_sidebar_state="expanded",
)


def inject_theme():
    """Load the Archive theme: fonts, tokens, and component styling.

    Streamlit renames internal classes between versions, so the selectors here
    use data-testid attributes, which are the stable ones.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
          --ink:#17263F; --paper:#F5F1E8; --surface:#FFFDF8;
          --accent:#B07A2C; --accent-strong:#8A5E1F; --accent-soft:#C79A4F;
          --muted:#5B6472; --line:rgba(23,38,63,0.12);
          --ok:#5FB865; --warn:#E0B64A; --danger:#E05C4A; --shipped:#5B9BD5;
        }

        /* Base */
        html, body, [data-testid="stAppViewContainer"] {
          background: var(--paper);
          color: var(--ink);
          font-family: 'IBM Plex Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }

        /* Headings */
        h1, h2, h3 {
          font-family: 'Source Serif 4', serif !important;
          font-weight: 600 !important;
          color: var(--ink) !important;
          letter-spacing: -0.01em;
        }
        h1 { font-size: 2.2rem !important; }

        /* Code / tool names */
        code, kbd, pre {
          font-family: 'IBM Plex Mono', monospace !important;
          color: var(--accent) !important;
          background: rgba(176,122,44,0.08) !important;
          border-radius: 4px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
          background: var(--surface);
          border-right: 1px solid var(--line);
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 6px;
          padding: 4px 6px;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
          border-left: 3px solid var(--accent);
        }
        [data-testid="stChatMessage"] p { line-height: 1.5; }

        /* Chat input */
        [data-testid="stChatInput"] {
          border: 1px solid var(--line);
          border-radius: 8px;
          background: var(--surface);
        }
        [data-testid="stChatInput"] textarea { color: var(--ink); }

        /* Buttons: primary is ochre, secondary is an outline */
        .stButton > button {
          font-family: 'IBM Plex Sans', sans-serif;
          font-weight: 600;
          border-radius: 5px;
          border: 1px solid var(--accent);
          background: var(--accent);
          color: var(--paper);
          transition: background .15s, border-color .15s;
        }
        .stButton > button:hover {
          background: var(--accent-strong);
          border-color: var(--accent-strong);
          color: var(--paper);
        }
        .stButton > button[kind="secondary"] {
          background: var(--surface);
          color: var(--ink);
          border: 1px solid var(--line);
        }
        .stButton > button[kind="secondary"]:hover {
          background: var(--surface);
          color: var(--accent-strong);
          border-color: var(--accent);
        }

        /* Alerts */
        [data-testid="stAlert"] {
          border-radius: 6px;
          border-left: 3px solid var(--accent);
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }

        /* Masthead */
        .wordmark {
          font-family: 'Source Serif 4', serif; font-weight: 700;
          font-size: 2.6rem; color: var(--ink); letter-spacing: -0.02em;
          line-height: 1.1; margin: 0;
        }
        .tagline {
          font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
          font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase;
          color: var(--accent); margin: 0.3rem 0 0;
        }
        .rule { border-bottom: 1px solid var(--line); margin: 1rem 0 1.4rem; }

        /* Labels */
        .label {
          font-family: 'IBM Plex Sans', sans-serif; font-size: 0.72rem;
          font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase;
          color: var(--muted); margin: 0 0 0.5rem;
        }

        /* Status badges */
        .legend__row {
          display: flex; align-items: center; justify-content: space-between;
          gap: 0.5rem; padding: 0.3rem 0;
        }
        .legend__id {
          font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
          font-weight: 600; color: var(--ink);
        }
        .badge {
          font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
          font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
          border-radius: 5px; padding: 0.16rem 0.5rem; color: #FFFDF8;
        }
        .badge--processing { background: var(--warn); color: #17263F; }
        .badge--shipped    { background: var(--shipped); }
        .badge--completed  { background: var(--ok); }
        .badge--cancelled  { background: var(--danger); }
        .badge--returned   { background: var(--muted); }

        /* Tool-call chip shown above an answer */
        .chip {
          display: inline-block; font-family: 'IBM Plex Mono', monospace;
          font-size: 0.7rem; font-weight: 600; color: var(--accent);
          background: rgba(176,122,44,0.08); border-radius: 4px;
          padding: 0.14rem 0.45rem; margin-bottom: 0.35rem;
        }
        .grounding {
          font-size: 0.75rem; color: var(--muted); margin: 0.4rem 0 0;
          font-style: italic;
        }

        /* Guardrail card */
        .draft {
          background: var(--surface); border: 1px solid var(--line);
          border-radius: 6px; padding: 1.1rem 1.25rem 1rem; margin: 0.4rem 0 0.7rem;
        }
        .draft__fn {
          font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem;
          font-weight: 600; color: var(--accent); margin: 0 0 0.85rem;
        }
        .draft__row { display: flex; gap: 0.75rem; padding: 0.28rem 0; }
        .draft__key {
          font-family: 'IBM Plex Sans', sans-serif; font-size: 0.72rem;
          font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
          color: var(--muted); min-width: 5.5rem; padding-top: 0.12rem;
        }
        .draft__val { color: var(--ink); font-size: 0.95rem; }
        .draft__val--mono { font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
        .draft__note {
          font-size: 0.8rem; color: var(--muted); margin: 0.9rem 0 0;
          padding-top: 0.7rem; border-top: 1px solid var(--line);
        }

        /* Routing log rows */
        .route {
          display: flex; align-items: center; gap: 0.6rem;
          padding: 0.45rem 0; border-bottom: 1px solid var(--line);
        }
        .route__q { font-size: 0.88rem; color: var(--ink); flex: 1; }
        .route__tool {
          font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 600;
          color: var(--accent); background: rgba(176,122,44,0.08);
          border-radius: 4px; padding: 0.14rem 0.45rem; white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

st.markdown(
    """
    <p class="wordmark">ResQ</p>
    <p class="tagline">Support agent with actions</p>
    <div class="rule"></div>
    """,
    unsafe_allow_html=True,
)


def read_secret(name):
    """Return a secret value, or an empty string if secrets aren't set up.

    When deployed, the key lives in the Streamlit Cloud secrets panel. Running
    locally there is usually no secrets file, hence the try/except.
    """
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


google_secret = read_secret("GOOGLE_API_KEY")
groq_secret = read_secret("GROQ_API_KEY")

with st.sidebar:
    st.header("Model")
    if google_secret and groq_secret:
        # Deployed: Groq runs the chat model, Gemini does the embeddings.
        provider = "Groq (free)"
        api_key = groq_secret
        embed_key = google_secret
        st.success("Connected. Start chatting.")
        st.caption("Llama 3.3 70B via Groq, Gemini embeddings.")
    elif google_secret:
        # Deployed with only a Google key: Gemini for chat and embeddings.
        provider = "Google Gemini"
        api_key = google_secret
        embed_key = google_secret
        st.success("Connected. Start chatting.")
        st.caption("Running on Gemini.")
    else:
        provider = st.selectbox("Provider", ["Groq (free)", "Google Gemini", "OpenAI"])
        api_key = st.text_input("API key", type="password")
        if provider == "Google Gemini":
            st.markdown(
                "Get a free Gemini key at "
                "[aistudio.google.com](https://aistudio.google.com/app/apikey)"
            )
            embed_key = api_key
        else:
            if provider.startswith("Groq"):
                st.markdown(
                    "Get a free Groq key at "
                    "[console.groq.com](https://console.groq.com)"
                )
            # Embeddings run on Gemini whatever the chat model is.
            embed_key = st.text_input("Google API key (for embeddings)", type="password")

    st.divider()
    st.markdown('<p class="label">Try these orders</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="legend__row"><span class="legend__id">SE10009</span>
          <span class="badge badge--shipped">Shipped</span></div>
        <div class="legend__row"><span class="legend__id">SE10001</span>
          <span class="badge badge--processing">Processing</span></div>
        <div class="legend__row"><span class="legend__id">SE10002</span>
          <span class="badge badge--returned">Returned</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.routing_log = []
        discard_pending_ticket()
        st.rerun()


@st.cache_resource(show_spinner="Building knowledge base index (first run only)...")
def load_retriever(key):
    return get_retriever(api_key=key, k=3)


@st.cache_resource(show_spinner="Starting agent...")
def build_agent(provider_name, key):
    """Build the tool-calling agent. Cached so it isn't rebuilt on every rerun."""
    if provider_name == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", temperature=0, google_api_key=key
        )
    elif provider_name.startswith("Groq"):
        from langchain_groq import ChatGroq

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=key)
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)

    # The routing rules live here rather than in if/else code, so the model
    # picks the tool itself.
    system_prompt = (
        "You are ResQ, the customer support agent for ShopEase, an Indian "
        "e-commerce marketplace.\n\n"
        "ROUTING RULES - follow strictly:\n"
        "1. For general policy/FAQ questions (shipping, returns, refunds, "
        "payments, warranty, account), ALWAYS call search_knowledge_base and "
        "answer ONLY from what it returns. Never answer from memory.\n"
        "2. If the customer asks about a specific order, call "
        "check_order_status with their order ID. If they haven't given an "
        "ID, ask for it.\n"
        "3. If the issue needs human investigation (shipment stuck over 48 "
        "hours, refund not received after the promised window, unauthorized "
        "activity), call raise_support_ticket. Tickets need user "
        "confirmation, so after drafting one, tell the customer to review "
        "and confirm it on screen.\n"
        "4. If the knowledge base has no answer, say you don't know and "
        "offer to raise a ticket. NEVER invent policies.\n\n"
        "Be concise, warm, and professional. Never ask for passwords or OTPs."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []      # {role, text, tools: [tool names]}
if "routing_log" not in st.session_state:
    st.session_state.routing_log = []   # which tool handled each query

if not api_key or not embed_key:
    st.info("Enter your API key in the sidebar to start chatting.")
    st.stop()

retriever = load_retriever(embed_key)
tools.set_retriever(retriever)
agent_executor = build_agent(provider, api_key)


def render_answer(text, used):
    """Show which tool produced an answer, then the answer itself.

    Seeing the route is the whole point of the agent, so it belongs in the
    interface rather than only in the log.
    """
    for name in used:
        st.markdown(f'<span class="chip">-> {name}</span>', unsafe_allow_html=True)
    st.markdown(text)
    if "search_knowledge_base" in used:
        st.markdown(
            '<p class="grounding">Answered from knowledge_base.md.</p>',
            unsafe_allow_html=True,
        )


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_answer(m["text"], m.get("tools", []))
        else:
            st.markdown(m["text"])

# Empty screen is an invitation to act: one starter per capability.
STARTERS = [
    "What is your return policy for electronics?",
    "Where is my order SE10009?",
    "Order SE10001 has been stuck in processing for five days.",
]
if not st.session_state.messages and not tools.pending_ticket:
    st.markdown('<p class="label">Try one of these</p>', unsafe_allow_html=True)
    for i, s in enumerate(STARTERS):
        if st.button(s, key=f"starter_{i}", use_container_width=True):
            st.session_state.queued_prompt = s
            st.rerun()

# Nothing is written to disk until the user clicks Confirm here.
if tools.pending_ticket:
    t = tools.pending_ticket
    st.markdown(
        f"""
        <div class="draft">
          <p class="draft__fn">raise_support_ticket()</p>
          <div class="draft__row">
            <span class="draft__key">Ticket</span>
            <span class="draft__val draft__val--mono">{t['ticket_id']}</span>
          </div>
          <div class="draft__row">
            <span class="draft__key">Order</span>
            <span class="draft__val draft__val--mono">{t['order_id']}</span>
          </div>
          <div class="draft__row">
            <span class="draft__key">Issue</span>
            <span class="draft__val">{t['issue_summary']}</span>
          </div>
          <p class="draft__note">Nothing is written until you confirm.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([2, 1])
    if c1.button("Confirm & save", type="primary", use_container_width=True):
        ticket = commit_pending_ticket()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": f"Ticket **{ticket['ticket_id']}** has been submitted. "
                        "Our team will reach out within 24 hours.",
                "tools": [],
            }
        )
        st.rerun()
    if c2.button("Discard", use_container_width=True):
        discard_pending_ticket()
        st.session_state.messages.append(
            {"role": "assistant", "text": "No problem, the draft ticket was discarded.",
             "tools": []}
        )
        st.rerun()

user_input = st.chat_input("Ask about an order, a policy, or report a problem")

# A clicked starter runs through exactly the same path as typed input.
if st.session_state.get("queued_prompt"):
    user_input = st.session_state.pop("queued_prompt")

if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input, "tools": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # The agent has no memory of its own, so pass the history back each turn.
    history = []
    for m in st.session_state.messages[:-1]:
        history.append(
            HumanMessage(m["text"]) if m["role"] == "user" else AIMessage(m["text"])
        )

    used = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_executor.invoke(
                    {"input": user_input, "chat_history": history}
                )
                answer = result["output"]

                # Record which tool was picked so the routing is visible.
                steps = result.get("intermediate_steps", [])
                used = [a.tool for a, _ in steps]
                st.session_state.routing_log.append(
                    {"query": user_input, "route": " -> ".join(used) or "answered directly"}
                )
            except Exception as e:
                answer = f"Something went wrong: {e}"

        render_answer(answer, used)
    st.session_state.messages.append(
        {"role": "assistant", "text": answer, "tools": used}
    )
    st.rerun()

if st.session_state.routing_log:
    with st.expander(f"Routing log ({len(st.session_state.routing_log)} handled)"):
        st.caption("Which capability answered each message.")
        for entry in st.session_state.routing_log:
            st.markdown(
                f'<div class="route"><span class="route__q">{entry["query"]}</span>'
                f'<span class="route__tool">{entry["route"]}</span></div>',
                unsafe_allow_html=True,
            )