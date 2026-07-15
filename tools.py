"""
The three tools the agent can call.

1. search_knowledge_base - retrieval from the policy docs (read only).
2. check_order_status    - looks up an order in data/orders.xlsx (read only).
3. raise_support_ticket  - creates a ticket, so it changes state. This one
   never writes to disk itself. It only stages a draft, and the UI asks the
   user to confirm before it is saved to data/tickets.xlsx.
"""

import os
import uuid
from datetime import datetime

import pandas as pd
from langchain_core.tools import tool

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ORDERS_PATH = os.path.join(DATA_DIR, "orders.xlsx")
TICKETS_PATH = os.path.join(DATA_DIR, "tickets.xlsx")

# app.py sets this once the vector store is ready.
_retriever = None

# Holds the unconfirmed ticket until the user clicks Confirm or Cancel. A
# module-level dict works here because Streamlit reruns share the process.
pending_ticket: dict = {}


def set_retriever(retriever):
    """Give the knowledge base tool access to the vector store."""
    global _retriever
    _retriever = retriever


_orders_df = None


def _load_orders():
    """Read the order spreadsheet once and keep it in memory.

    One row per order, keyed by order_id. Reading it on every lookup would be
    slow, since Streamlit reruns the script on each interaction.
    """
    global _orders_df
    if _orders_df is None:
        df = pd.read_excel(ORDERS_PATH, sheet_name="Orders", dtype=str)
        df["order_id"] = df["order_id"].str.strip().str.upper()
        _orders_df = df.set_index("order_id")
    return _orders_df


@tool
def search_knowledge_base(query: str) -> str:
    """Search the official ShopEase support knowledge base for policies on
    shipping, returns, refunds, payments, cancellations, warranty, or account
    issues. Use this for ANY general question. Answer ONLY from what this
    tool returns; if it returns nothing relevant, say you don't know."""
    if _retriever is None:
        return "Knowledge base is not loaded."
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant policy found in the knowledge base."
    return "\n\n---\n\n".join(d.page_content for d in docs)


@tool
def check_order_status(order_id: str) -> str:
    """Look up the live status of a customer's order in the order system.
    Requires an Order ID in the format SE followed by 5 digits, e.g. SE10023.
    Use this whenever the customer asks about a specific order."""
    order_id = order_id.strip().upper()
    try:
        orders = _load_orders()
    except FileNotFoundError:
        return "Order database unavailable."

    if order_id not in orders.index:
        return (
            f"No order found with ID {order_id}. Ask the customer to "
            "double-check the ID (format: SE followed by 5 digits)."
        )
    record = orders.loc[order_id].to_dict()
    lines = [f"order_id: {order_id}"]
    lines += [f"{k}: {v}" for k, v in record.items()]
    return "\n".join(lines)


@tool
def raise_support_ticket(order_id: str, issue_summary: str) -> str:
    """Draft a support ticket for a human agent to investigate an issue,
    e.g. a stuck shipment, a failed refund, or suspicious account activity.
    Provide the order ID (or 'N/A' if not order-related) and a one-line
    summary of the issue. The ticket is NOT created immediately: the
    customer must confirm it first."""
    global pending_ticket
    pending_ticket = {
        "ticket_id": "TKT-" + uuid.uuid4().hex[:6].upper(),
        "order_id": order_id.strip().upper(),
        "issue_summary": issue_summary.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PENDING_CONFIRMATION",
    }
    return (
        f"Ticket {pending_ticket['ticket_id']} has been DRAFTED but not yet "
        "submitted. Tell the customer to review the draft shown on screen "
        "and press Confirm to submit it, or Cancel to discard it."
    )


def commit_pending_ticket() -> dict:
    """Called by the UI when the user clicks Confirm. This is the only place
    a ticket actually gets written to disk.

    Tickets are appended to data/tickets.xlsx. The file is read back and
    rewritten each time, which is fine at this volume and keeps the sheet
    readable in Excel.
    """
    global pending_ticket
    if not pending_ticket:
        return {}
    ticket = {**pending_ticket, "status": "OPEN"}

    if os.path.exists(TICKETS_PATH):
        existing = pd.read_excel(TICKETS_PATH, sheet_name="Tickets", dtype=str)
        df = pd.concat([existing, pd.DataFrame([ticket])], ignore_index=True)
    else:
        df = pd.DataFrame([ticket])
    df.to_excel(TICKETS_PATH, sheet_name="Tickets", index=False)

    pending_ticket = {}
    return ticket


def discard_pending_ticket() -> None:
    """Called by the UI when the user clicks Cancel."""
    global pending_ticket
    pending_ticket = {}


ALL_TOOLS = [search_knowledge_base, check_order_status, raise_support_ticket]
