import json, datetime, math, re, os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---- paths (root-level) ----
PROJECT_ROOT= Path.cwd().resolve()
DATA_DIR = PROJECT_ROOT /"data"
DATA_DIR.mkdir(parents=True, exist_ok = True)

DOC_PATH = PROJECT_ROOT / "activities.md"           # profile document
INDEX_PATH = DATA_DIR / "retrieval_index.json"      # where we save the index

# ---- env + client ----
load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- load & chunk (keep headings with content) ----
text = DOC_PATH.read_text(encoding="utf-8")
parts = re.split(r'(^## .*?$)', text, flags=re.M)

chunks, heading = [], ""
for p in parts:
    s = p.strip()
    if not s:
        continue
    if s.startswith("##"):
        heading =s

    else:
        chunks.append(f"{heading}\n{s}" if heading else s)

print(f"chunks build: {len(chunks)}")

# ---- embed chunks ----

emb = client.embeddings.create(model="text-embedding-3-small", input=chunks)
chunk_vecs = [d.embedding for d in emb.data]

# ---- save one compact artifact ----

artifact = {
    "created_at":datetime.datetime.utcnow().isoformat() + "Z",
    "model": "text-embedding-3-small",
    "doc": str(DOC_PATH),
    "chunks": chunks,
    "embeddings": chunk_vecs
}

INDEX_PATH.write_text(json.dumps(artifact), encoding="utf-8")
print("Saved index", INDEX_PATH)