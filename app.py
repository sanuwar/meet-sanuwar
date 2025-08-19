# app.py — Meet Sanuwar (tiny, beginner-friendly)
# What this does:
# 1) Load the saved retrieval index from Step 2 (data/retrieval_index.json)
# 2) Load a shared system prompt (prompts/system_prompt.txt)
# 3) For each message: find relevant chunks, ask gpt-4o-mini, show the answer
# 4) Log unknown questions + optional name/email via your Step 3 helpers

import os, json
from pathlib import Path

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

# ---------- Load retrieval index (built in Step 2) ----------

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Missing {INDEX_PATH}. Run Step 2 to create data/retrieval_index.json."
    )

artifact = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
chunks = artifact["chunks"]                      # list[str]
chunk_embeds = artifact["embeddings"]            # list[list[float]]
embedding_model = artifact.get("model", "text-embedding-3-small")

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

    # NEW: Save contact by button OR pressing Enter in name/email
    save.click(on_save_contact, inputs=[name, email], outputs=[status])
    name.submit(on_save_contact, inputs=[name, email], outputs=[status])
    email.submit(on_save_contact, inputs=[name, email], outputs=[status])

if __name__ == "__main__":
    demo.launch()