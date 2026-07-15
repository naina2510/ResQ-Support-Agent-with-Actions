# ResQ - Support Agent with Actions

A customer support agent for a fictional e-commerce store called ShopEase. It
answers policy questions from an approved knowledge base using RAG, looks up
order status from the order database, and can raise a support ticket. The agent
decides for itself whether to answer or call a tool, and any action that changes
data has to be confirmed by the user first.

Built with LangChain (tool-calling agent), Google Gemini, FAISS and Streamlit.
Gemini handles both chat and embeddings, so nothing large is downloaded locally
and the install stays small. Groq and OpenAI can also be selected in the sidebar.

## How it works

```
User (Streamlit chat)
        |
        v
LangChain tool-calling agent (Gemini 2.5 Flash-Lite, or Groq / OpenAI)
        |  picks a route per query
        |--> search_knowledge_base --> FAISS (Gemini embeddings) --> KB chunks
        |--> check_order_status ------> data/orders.xlsx
        |--> raise_support_ticket ---> draft ticket, not saved yet
                                          |
                              user clicks Confirm
                                          |
                                          v
                                  data/tickets.xlsx
```

Routing is not hard-coded. The tool descriptions and the routing rules in the
system prompt are what the model uses to choose, and the routing log in the UI
shows which tool handled each message.

The ticket tool is deliberately unable to write to disk. It only stages a draft,
and `commit_pending_ticket()` (called by the Confirm button) is the only code
path that saves anything. So the confirmation step can't be skipped even if the
model misbehaves.

## Requirements

Python 3.12 (3.14 is not supported yet by LangChain's dependencies) and a free
API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Running it locally

```bash
cd resq

# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Pick a provider in the sidebar and paste
your key. The first run embeds the knowledge base and builds the FAISS index,
which takes a few seconds; after that the saved index loads instantly.

If you don't want to paste the key every time, copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put your key
in there. That file is gitignored.

## Demo

1. "What is your return policy for electronics?"
   Answers from the knowledge base only.
2. "Where is my order SE10009?"
   Calls the order lookup tool. Open the routing log to see the two different
   routes taken so far.
3. "What about order SE10001? It's been stuck in processing for days."
   The tool shows it is still Processing, and the agent offers a ticket.
4. "Yes, please raise a ticket for it."
   A draft ticket appears with Confirm and Cancel buttons. Confirm writes it to
   `data/tickets.xlsx`.
5. "Do you sell mobile phones on EMI?"
   Not covered in the knowledge base, so the agent says it doesn't know rather
   than inventing a policy.

Sample order IDs are shown in the sidebar: SE10009 (Shipped), SE10001
(Processing), SE10002 (Returned).

## Data

`data/orders.xlsx` (sheet "Orders") holds 1,000 orders, 200 each of Processing,
Shipped, Completed, Cancelled and Returned. They were sampled from a public
synthetic e-commerce dataset and joined across its orders, order items, users and
products files. Order IDs were remapped to the ShopEase format SE10001 to
SE11000, and the `source_order_id` column keeps the original ID from the dataset.

`data/knowledge_base.md` is written by hand and covers shipping, returns,
refunds, payments, cancellations, warranty and account policies. It is the only
source the agent is allowed to answer policy questions from.

`data/tickets.xlsx` is created at runtime once the first ticket is confirmed,
with one row per ticket.

## Deploying

The app reads the API key from server-side secrets when they exist, so the
deployed link works for visitors without a key of their own, and the key stays
out of the repo and out of the browser.

1. Push to a public GitHub repo (Streamlit's free tier needs it public):

   ```bash
   git init
   git add .
   git commit -m "ResQ support agent"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```

   Run `git status` first and check that `.streamlit/secrets.toml` is not listed.
   `.gitignore` already excludes it, along with `.venv/` and `faiss_index/`.

2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, click
   Create app, and pick the repo, branch `main`, and `app.py`.

3. Before deploying, open Advanced settings > Secrets and paste:

   ```toml
   GOOGLE_API_KEY = "your-gemini-key"
   ```

4. Deploy. The first build takes a few minutes, and you get a link like
   `https://<app-name>.streamlit.app`.

To change the key later: app page > Settings > Secrets.

Note that anyone with the link uses your Gemini quota. The free tier is fine for
a demo, but delete the app when you're done with it.

## Design notes

**Why RAG rather than just asking the model.** The agent has to answer from
approved policy, not from whatever the model remembers. Retrieval grounds the
answer in the knowledge base and makes it possible to say "I don't know" for
anything outside it.

**How the routing works.** The model gets the tool names and descriptions plus
the routing rules in the system prompt, and the tool-calling API lets it return a
structured tool call. LangChain's AgentExecutor runs the tool and feeds the
result back until the model produces an answer. Temperature is 0 to keep this
consistent.

**Why the ticket tool doesn't write.** Staging the draft and committing it from
the UI makes the guardrail structural rather than something the model is merely
asked to respect.

**Why Gemini embeddings and FAISS.** Embedding through the API means no local
model download, and FAISS is a small in-process index, which suits a knowledge
base this size and keeps the deployment light.

**Limitations.** The order data is a spreadsheet, not a live API. There's one
knowledge base file and no authentication. Escalating to a human when the model
is unsure would be the obvious next thing to add.

## Files

```
resq/
├── app.py                # Streamlit UI, agent setup, confirmation panel
├── rag.py                # chunking, embeddings, FAISS index
├── tools.py              # the three tools and the ticket staging logic
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
└── data/
    ├── knowledge_base.md   # policy text used for RAG
    └── orders.xlsx         # 1,000 orders
                           # tickets.xlsx is created here at runtime
```
