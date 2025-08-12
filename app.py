import os
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from ollama import Client

# ---------------- CONFIG ----------------
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"  # must be installed in Ollama
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = Path("./memory_index")
META_FILE = INDEX_DIR / "meta.json"
INDEX_FILE = INDEX_DIR / "vectors.faiss"
UPLOAD_FOLDER = Path("./uploads")
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

INDEX_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = Client(host=OLLAMA_HOST)
# -----------------------------------------


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    text = []
    reader = PdfReader(str(path))
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except:
            pass
    return "\n".join(text)

def chunk_text(text: str, max_tokens=500, overlap=50) -> List[str]:
    chunk_size = max_tokens * 4
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - (overlap*4)
    return chunks

def safe_load_meta():
    if not META_FILE.exists() or META_FILE.stat().st_size == 0:
        meta = {"ids": [], "metas": []}
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict) or "metas" not in meta or "ids" not in meta:
            raise ValueError("meta.json invalid format")
        return meta
    except Exception:
        meta = {"ids": [], "metas": []}
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

def load_or_create_index(dim: int):
    if INDEX_FILE.exists():
        try:
            idx = faiss.read_index(str(INDEX_FILE))
            meta = safe_load_meta()
            return idx, meta
        except Exception as e:
            print("Warning: failed to read existing FAISS index:", e)
            try:
                INDEX_FILE.unlink()
            except Exception:
                pass
    idx = faiss.IndexFlatIP(dim)
    meta = safe_load_meta()
    return idx, meta

def add_documents(paths: List[Path]):
    texts = []
    metas = []
    for p in paths:
        if p.suffix.lower() in [".txt", ".md"]:
            raw = read_txt(p)
        elif p.suffix.lower() == ".pdf":
            raw = read_pdf(p)
        else:
            continue
        chunks = chunk_text(raw)
        for ci, c in enumerate(chunks):
            texts.append(c)
            metas.append({"source": str(p), "chunk": ci, "length": len(c)})
    if not texts:
        return 0

    emb = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(emb)

    idx, meta = load_or_create_index(emb.shape[1])
    try:
        idx.add(emb)
    except Exception as e:
        print("Index add failed (possibly dim mismatch). Recreating index. Error:", e)
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)

    start_id = len(meta["ids"])
    for i, m in enumerate(metas):
        meta["ids"].append(start_id + i)
        meta["metas"].append(m)

    try:
        faiss.write_index(idx, str(INDEX_FILE))
    except Exception as e:
        print("Warning: failed to write FAISS index:", e)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return len(texts)

def retrieve(query: str, top_k=5) -> List[Tuple[dict, float, str]]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    idx, meta = load_or_create_index(q_emb.shape[1])
    if idx.ntotal == 0:
        return []
    D, I = idx.search(q_emb, top_k)
    results = []
    for score, idx_id in zip(D[0], I[0]):
        try:
            m = meta["metas"][int(idx_id)]
        except Exception:
            continue
        src = Path(m["source"])
        raw = read_pdf(src) if src.suffix.lower() == '.pdf' else read_txt(src)
        chunks = chunk_text(raw)
        chunk_idx = m.get("chunk", 0)
        chunk_text_selected = chunks[chunk_idx] if chunk_idx < len(chunks) else chunks[0]
        results.append((m, float(score), chunk_text_selected))
    return results

def extract_ollama_content(resp) -> str:
    if isinstance(resp, dict):
        if "message" in resp and isinstance(resp["message"], dict) and "content" in resp["message"]:
            return str(resp["message"]["content"]).strip()
        if "response" in resp and isinstance(resp["response"], str):
            return resp["response"].strip()
        if "content" in resp and isinstance(resp["content"], str):
            return resp["content"].strip()
    try:
        if hasattr(resp, "message") and getattr(resp.message, "content", None) is not None:
            return str(resp.message.content).strip()
    except Exception:
        pass
    return str(resp).strip()

# Flask app setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_secret_key_here")  # Change to env var in prod

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/index_info")
def index_info():
    meta = safe_load_meta()
    count = len(meta.get("metas", []))
    return jsonify({"chunks": count})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        safe_name = file.filename.replace("..", "_")
        filepath = UPLOAD_FOLDER / safe_name
        file.save(filepath)
        added = add_documents([filepath])
        return jsonify({"success": True, "filename": safe_name, "added_chunks": added})
    return jsonify({"error": "Invalid file type"}), 400

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    history = session.get("history", [])
    history.append({"role": "user", "content": question})

    try:
        meta = safe_load_meta()
        chunks_count = len(meta.get("metas", []))

        if chunks_count > 0:
            # Try to retrieve relevant context chunks
            docs = retrieve(question, top_k=4)
            if docs:
                context_chunks = [d[2] for d in docs]
                context_str = "\n\n---\n\n".join(context_chunks)
                system_prompt = "You are a helpful assistant. Answer using only the provided context. If unknown, say you don't know."
                messages = [{"role": "system", "content": system_prompt}]
                messages += history[:-1]
                messages.append({"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"})
            else:
                # No relevant chunks found, fallback to zero-shot
                system_prompt = "You are a smart assistant answering questions accurately."
                messages = [{"role": "system", "content": system_prompt}] + history
        else:
            # No files uploaded, zero-shot answering
            system_prompt = "You are a smart assistant answering questions accurately."
            messages = [{"role": "system", "content": system_prompt}] + history

        resp = client.chat(model=MODEL_NAME, messages=messages, stream=False)
        answer = extract_ollama_content(resp)

        history.append({"role": "assistant", "content": answer})
        session["history"] = history

        return jsonify({"answer": answer})
    except Exception as e:
        print("Error in /ask:", e)
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
