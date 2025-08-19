import os, json

from pathlib import Path
import re, datetime

from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# --- Use Step 3 CSV helpers (so app.py stays small) ---
# Make sure step3_csv_tools.py is in the project root.
from step3_csv_tools import record_user_details, log_unknown

# ---------- Paths & Setup ----------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
INDEX_PATH = DATA_DIR / "retrieval_index.json"        # created by Step 2
PROMPTS_DIR = BASE / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

# Load OpenAI key from .env
load_dotenv(BASE / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- (Option 1) Rebuild index if missing or when REBUILD_INDEX=1 ----------
DOC_PATH = BASE / "activities.md"  # used only for (re)building the index

def build_index():
    """(Re)create data/retrieval_index.json from activities.md."""
    text = DOC_PATH.read_text(encoding="utf-8")
    # heading-aware chunking (same approach as Step 2)
    parts = re.split(r'(^## .*?$)', text, flags=re.M)
    chunks_local, heading = [], ""
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if s.startswith("## "):
            heading = s
        else:
            chunks_local.append(f"{heading}\n{s}" if heading else s)

    # embed all chunks using the small, cheap model
    emb = client.embeddings.create(model="text-embedding-3-small", input=chunks_local)
    chunk_vecs_local = [d.embedding for d in emb.data]

    artifact = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "model": "text-embedding-3-small",
        "doc": str(DOC_PATH),
        "chunks": chunks_local,
        "embeddings": chunk_vecs_local,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(artifact), encoding="utf-8")

def load_index():
    """Load chunks + embeddings from the saved JSON into globals."""
    global chunks, chunk_embeds, embedding_model
    artifact_local = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    chunks = artifact_local["chunks"]                      # list[str]
    chunk_embeds = artifact_local["embeddings"]            # list[list[float]]
    embedding_model = artifact_local.get("model", "text-embedding-3-small")

# ---------- Load system prompt ----------
if SYSTEM_PROMPT_PATH.exists():
    SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
else:
    # fallback so the app still runs if the file is missing
    SYSTEM_PROMPT = (
        "You are Sanuwar’s assistant. Be friendly and concise. "
        "If the message is a greeting/thanks/farewell, reply naturally in one short sentence. "
        "For factual questions, use ONLY the provided Context. "
        "If the answer is not in the Context, reply exactly: \"I do not know.\""
    )

# ---------- Load (or rebuild) retrieval index ----------
if (not INDEX_PATH.exists()) or os.getenv("REBUILD_INDEX") == "1":
    # On Hugging Face, set REBUILD_INDEX=1 under “Settings → Variables and secrets”
    # to regenerate after updating activities.md, then remove it again.
    build_index()

load_index()  # populates: chunks, chunk_embeds, embedding_model

# ---------- Tiny search (embed query + cosine) ----------
def embed_query(text: str):
    """Create an embedding for the user's query using the same model as Step 2."""
    resp = client.embeddings.create(model=embedding_model, input=[text])
    return resp.data[0].embedding

def cosine(a, b):
    """Simple cosine similarity; higher = more similar."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    na = (na ** 0.5) or 1.0
    nb = (nb ** 0.5) or 1.0
    return dot / (na * nb)

def search(question: str, k: int = 5):
    """Return top-k most relevant chunk texts for this question."""
    qv = embed_query(question)
    scored = []
    for c_text, c_vec in zip(chunks, chunk_embeds):
        scored.append((c_text, cosine(qv, c_vec)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _ in scored[:k]]

# ---------- Ask the assistant ----------
def ask_bot(user_message: str) -> str:
    if not user_message:
        return "Please type a question."

    # Build a small context pack from the saved chunks
    top = search(user_message, k=5)
    context_text = "\n\n---\n\n".join(top)

    # The system prompt teaches the bot to handle hi/bye/thanks naturally
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": (
             "Context:\n" + context_text + "\n\n"
             "User message: " + user_message + "\n\n"
             "If the message is just a greeting/thanks/farewell, reply naturally without using the Context. "
             "Otherwise, use ONLY the Context. Keep it concise."
         )},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()

    # If the model couldn't find an answer in the context, we log the question
    if "i do not know" in answer.lower():
        log_unknown(user_message)

    return answer

def on_send(message, name, email, history):
    # Save contact if provided (optional)
    if name or email:
        record_user_details(name=name, email=email)

    reply = ask_bot(message)
    return history + [(message, reply)], ""  # clear textbox after send

# NEW: tiny handler to save contact without sending a chat
def on_save_contact(name, email):
    if not (name or email):
        return "⚠️ Please enter a name or email."
    record_user_details(name=name, email=email)
    return "✅ Contact saved."

with gr.Blocks(title="Meet Sanuwar") as demo:
    gr.Markdown("### Meet Sanuwar\nFriendly, professional Q&A about Sanuwar’s background.")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=420)
            msg = gr.Textbox(label="Your question", placeholder="Type and press Enter to send")
            send = gr.Button("Send")
        with gr.Column(scale=2):
            name = gr.Textbox(label="Name (optional)")
            email = gr.Textbox(label="Email (optional)")
            save = gr.Button("Save contact")      # NEW: optional button
            status = gr.Markdown("")              # NEW: feedback text

    # Pressing Enter in the message box sends
    msg.submit(on_send, inputs=[msg, name, email, chat], outputs=[chat, msg])
    # Clicking Send sends
    send.click(on_send, inputs=[msg, name, email, chat], outputs=[chat, msg])

    # Save contact by button OR pressing Enter in name/email
    save.click(on_save_contact, inputs=[name, email], outputs=[status])
    name.submit(on_save_contact, inputs=[name, email], outputs=[status])
    email.submit(on_save_contact, inputs=[name, email], outputs=[status])

if __name__ == "__main__":
    demo.launch()
