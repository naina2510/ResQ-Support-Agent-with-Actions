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

st.set_page_config(page_title="ResQ Support Agent", layout="wide")
st.title("ResQ Support Agent")
st.caption("Knowledge base answers, order lookups, and tickets with confirmation")


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
    st.header("Model settings")
    if google_secret and groq_secret:
        # Deployed: Groq runs the chat model, Gemini does the embeddings.
        provider = "Groq (free)"
        api_key = groq_secret
        embed_key = google_secret
        st.success("Model connected. Start chatting below.")
        st.caption("Running on Llama 3.3 70B via Groq.")
    else:
        provider = st.selectbox("LLM provider", ["Google Gemini", "Groq (free)", "OpenAI"])
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
    st.header("Demo order IDs")
    st.code("SE10009  Shipped\nSE10001  Processing\nSE10002  Returned")

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
    st.session_state.messages = []      # list of (role, text)
if "routing_log" not in st.session_state:
    st.session_state.routing_log = []   # which tool handled each query

if not api_key or not embed_key:
    st.info("Enter your API key in the sidebar to start chatting.")
    st.stop()

retriever = load_retriever(embed_key)
tools.set_retriever(retriever)
agent_executor = build_agent(provider, api_key)

for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# Nothing is written to disk until the user clicks Confirm here.
if tools.pending_ticket:
    t = tools.pending_ticket
    with st.container(border=True):
        st.subheader("Confirm before this action is executed")
        st.write(f"**Draft ticket {t['ticket_id']}** | Order: `{t['order_id']}`")
        st.write(f"Issue: {t['issue_summary']}")
        c1, c2 = st.columns(2)
        if c1.button("Confirm and submit ticket", use_container_width=True):
            ticket = commit_pending_ticket()
            st.session_state.messages.append(
                (
                    "assistant",
                    f"Ticket **{ticket['ticket_id']}** has been submitted. "
                    "Our team will reach out within 24 hours.",
                )
            )
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            discard_pending_ticket()
            st.session_state.messages.append(
                ("assistant", "No problem, the draft ticket was discarded.")
            )
            st.rerun()

user_input = st.chat_input("Ask about an order, a policy, or report a problem")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # The agent has no memory of its own, so pass the history back each turn.
    history = []
    for role, text in st.session_state.messages[:-1]:
        history.append(HumanMessage(text) if role == "user" else AIMessage(text))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_executor.invoke(
                    {"input": user_input, "chat_history": history}
                )
                answer = result["output"]

                # Record which tool was picked so the routing is visible.
                steps = result.get("intermediate_steps", [])
                used = [a.tool for a, _ in steps] or ["(answered directly)"]
                st.session_state.routing_log.append(
                    {"query": user_input, "route": " -> ".join(used)}
                )
            except Exception as e:
                answer = f"Something went wrong: {e}"

        st.markdown(answer)
    st.session_state.messages.append(("assistant", answer))
    st.rerun()

if st.session_state.routing_log:
    with st.expander("Agent routing log (how each query was handled)"):
        for entry in st.session_state.routing_log:
            st.markdown(f"- **{entry['query']}** -> `{entry['route']}`")
