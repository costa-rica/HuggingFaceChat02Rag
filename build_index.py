import os, json, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

CONTEXT_DIR = os.getenv("PATH_TO_CONTEXT_DATA")
INDEX_DIR = os.getenv("PATH_TO_INDEX")

if not CONTEXT_DIR or not INDEX_DIR:
    raise ValueError("Please set PATH_TO_CONTEXT_DATA and PATH_TO_INDEX in your environment.")

import os
CSV_PATH = os.path.join(CONTEXT_DIR, "daily_sleep.csv")
DOCS_JSON = os.path.join(INDEX_DIR, "docs.json")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def row_to_doc(r):
    day = str(r.get('day', '')).strip()
    score = r.get('score')
    hours = r.get('total_sleep_hours')

    score_str = "" if pd.isna(score) else (str(int(score)) if float(score).is_integer() else f"{score}")
    hours_str = "" if pd.isna(hours) else f"{float(hours):.2f}"

    return f"[{day}] hours:{hours_str} score:{score_str}"

# def main():
def step_1_ingest():
    df = pd.read_csv(CSV_PATH)
    df['score'] = pd.to_numeric(df.get('score'), errors='coerce')
    df['total_sleep_hours'] = pd.to_numeric(df.get('total_sleep_hours'), errors='coerce')
    df = df.dropna(subset=['day'])
    docs = [row_to_doc(r) for _, r in df.iterrows()]
    return docs

def step_2_encode(docs):
    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized dot product
    index.add(embs)
    return index


def step_3_write_index(index, docs):
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_JSON, "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} docs → {INDEX_PATH}")

if __name__ == "__main__":
    docs = step_1_ingest()
    index = step_2_encode(docs)
    step_3_write_index(index, docs)