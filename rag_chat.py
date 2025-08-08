import os, json, requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

# --- env & paths ---
BASE_URL = os.getenv("FRIENDLI_BASE_URL", "https://api.friendli.ai/dedicated").rstrip("/")
TOKEN = os.getenv("FRIENDLI_TOKEN")
ENDPOINT_ID = os.getenv("FRIENDLI_ENDPOINT_ID")
INDEX_DIR = os.getenv("PATH_TO_INDEX")
if not (TOKEN and ENDPOINT_ID and INDEX_DIR):
    raise ValueError("Set FRIENDLI_TOKEN, FRIENDLI_ENDPOINT_ID, and PATH_TO_INDEX in your environment.")

INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
DOCS_JSON = os.path.join(INDEX_DIR, "docs.json")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- retrieval ---
def retrieve(query, k=7, debug=False):
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_JSON) as f:
        docs = json.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k)
    ctx = [docs[i] for i in idxs[0]]
    if debug:
        print("DEBUG: Retrieved documents:")
        for i, doc_idx in enumerate(idxs[0]):
            print(f"  [{i}] Index: {doc_idx}, Text: {docs[doc_idx]}")
    return ctx

# --- generation (Friendli chat completions) ---
def ask_llm(question, context_snippets):
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

    system_msg = (
        "You are a concise analyst. Use ONLY the provided context. "
        "If the context is insufficient, say so explicitly."
    )
    ctx_blob = "\n".join(f"- {c}" for c in context_snippets)
    user_msg = f"Question: {question}\n\nContext:\n{ctx_blob}\n\nAnswer with dates and numbers when possible."

    payload = {
        "model": ENDPOINT_ID,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
        "stream": False,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    DEBUG = True
    question = "What has my sleep been like for the past week?"
    ctx = retrieve(question, k=7, debug=DEBUG)
    print(ask_llm(question, ctx))