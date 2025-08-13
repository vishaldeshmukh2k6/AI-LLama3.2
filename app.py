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
from duckduckgo_search import DDGS

# --- CONFIG ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"
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

# --- HELPERS ---
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
        with open(META_FILE, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
        return meta
    try:
        with open(META_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except:
        meta = {"ids": [], "metas": []}
        with open(META_FILE, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
        return meta

def load_or_create_index(dim: int):
    if INDEX_FILE.exists():
        try:
            idx = faiss.read_index(str(INDEX_FILE))
            return idx, safe_load_meta()
        except:
            INDEX_FILE.unlink(missing_ok=True)
    return faiss.IndexFlatIP(dim), safe_load_meta()

def extract_ollama_content(resp) -> str:
    try:
        if isinstance(resp, dict):
            if "message" in resp and isinstance(resp["message"], dict) and "content" in resp["message"]:
                return resp["message"]["content"].strip()
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return resp.message.content.strip()
        return str(resp).strip()
    except:
        return str(resp)

def add_documents(paths: List[Path]):
    texts, metas = [], []
    for p in paths:
        raw = read_pdf(p) if p.suffix.lower() == ".pdf" else read_txt(p)
        chunks = chunk_text(raw)
        for ci, c in enumerate(chunks):
            metas.append({"source": str(p), "chunk": ci, "length": len(c), "query_count": 0})
            texts.append(c)
    if not texts:
        return 0, None

    emb = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(emb)
    idx, meta = load_or_create_index(emb.shape[1])
    idx.add(emb)

    start_id = len(meta["ids"])
    for i, m in enumerate(metas):
        meta["ids"].append(start_id+i)
        meta["metas"].append(m)

    faiss.write_index(idx, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)

    # Auto-summary
    summary_prompt = f"Summarize the following document in 5 concise bullet points:\n\n{raw[:2000]}"
    resp = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": summary_prompt}], stream=False)
    summary = extract_ollama_content(resp)
    return len(texts), summary

def retrieve(query: str, top_k=5) -> List[Tuple[dict, float, str]]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    idx, meta = load_or_create_index(q_emb.shape[1])
    if idx.ntotal == 0: return []
    D, I = idx.search(q_emb, top_k)
    results = []
    updated = False
    for score, idx_id in zip(D[0], I[0]):
        m = meta["metas"][int(idx_id)]
        m["query_count"] = m.get("query_count", 0) + 1
        updated = True
        src = Path(m["source"])
        raw = read_pdf(src) if src.suffix.lower() == ".pdf" else read_txt(src)
        chunk_txt = chunk_text(raw)
        chunk_text_selected = chunk_txt[m.get("chunk", 0)] if m.get("chunk", 0) < len(chunk_txt) else chunk_txt[0]
        results.append((m, float(score), chunk_text_selected))
    if updated:
        with open(META_FILE, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
    return results

# --- FLASK APP ---
app = Flask(__name__)
app.secret_key = "super_secret"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    safe_name = file.filename.replace("..", "_")
    save_path = UPLOAD_FOLDER / safe_name
    file.save(save_path)
    added, summary = add_documents([save_path])
    return jsonify({"success": True, "filename": safe_name, "added_chunks": added, "summary": summary})

def web_search(query, num_results=3):
    """DuckDuckGo web search fallback"""
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query, safesearch="Moderate", max_results=num_results):
            results.append(f"{r['title']} - {r['href']}\n{r['body']}")
        return "\n\n".join(results)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    q = data.get("question", "")
    if not isinstance(q, str):
        q = str(q)
    q = q.strip()
    if not q:
        return jsonify({"error": "Question cannot be empty"}), 400

    hist = session.get("hist", [])
    hist.append({"role": "user", "content": q})

    try:
        meta = safe_load_meta()
        if meta["metas"]:
            docs = retrieve(q, top_k=4)
            if docs:
                context = "\n\n---\n\n".join([
                    f"Source: {d[0]['source']} (chunk {d[0]['chunk']})\n{d[2]}"
                    for d in docs
                ])
                prompt_messages = [
                    {"role": "system", "content": (
                        "You are a helpful AI assistant. Answer the user's question using ONLY the provided context. "
                        "If the answer is not contained within the context, reply exactly with: NEED_WEB_SEARCH."
                    )}
                ] + hist[:-1] + [{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {q}"
                }]
            else:
                prompt_messages = [
                    {"role": "system", "content": (
                        "You are a helpful AI assistant. If you don't know from your own knowledge, reply exactly with: NEED_WEB_SEARCH."
                    )}
                ] + hist
        else:
            prompt_messages = [
                {"role": "system", "content": (
                    "You are a helpful AI assistant. If you don't know from your own knowledge, reply exactly with: NEED_WEB_SEARCH."
                )}
            ] + hist

        # First try local/context answer
        resp = client.chat(model=MODEL_NAME, messages=prompt_messages, stream=False)
        answer = extract_ollama_content(resp)

        # If AI signals web search needed
        if "NEED_WEB_SEARCH" in answer or not answer.strip():
            web_results = web_search(q)
            web_prompt = [
                {"role": "system", "content": (
                    "You are a helpful AI assistant. Use the provided live web search results to answer the user's question. "
                    "Be concise, and say that the information is from the internet."
                )},
                {"role": "user", "content": f"Search Results:\n{web_results}\n\nQuestion: {q}"}
            ]
            resp2 = client.chat(model=MODEL_NAME, messages=web_prompt, stream=False)
            answer = extract_ollama_content(resp2)

        # Append to history
        hist.append({"role": "assistant", "content": answer})
        session["hist"] = hist

        # Suggestions
        sug_prompt = f"Generate 3 concise and relevant follow-up questions based on this answer:\n{answer}"
        sug_resp = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": sug_prompt}], stream=False)
        suggestions_text = extract_ollama_content(sug_resp)
        suggestions = [s.strip("-• ") for s in suggestions_text.split("\n") if s.strip()]

        return jsonify({"answer": answer, "suggestions": suggestions})

    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.route("/index_info")
def index_info():
    meta = safe_load_meta()
    return jsonify({"chunks": len(meta.get("metas", []))})

@app.route("/query_stats")
def query_stats():
    st = {}
    for m in safe_load_meta()["metas"]:
        st[m["source"]] = st.get(m["source"], 0) + m.get("query_count", 0)
    return jsonify(st)

@app.route("/files")
def files():
    meta = safe_load_meta()
    files_data = {}
    for m in meta["metas"]:
        src = m["source"]
        if src not in files_data:
            files_data[src] = {"chunks": 0, "queries": 0}
        files_data[src]["chunks"] += 1
        files_data[src]["queries"] += m.get("query_count", 0)
    return jsonify(files_data)

@app.route("/delete_file", methods=["POST"])
def delete_file():
    fname = request.json.get("filename")
    if not fname:
        return jsonify({"error": "No filename"}), 400
    path = UPLOAD_FOLDER / fname
    if path.exists():
        path.unlink()
    meta = safe_load_meta()
    new_meta = {"ids": [], "metas": []}
    for i, m in enumerate(meta["metas"]):
        if Path(m["source"]).name != fname:
            new_meta["ids"].append(len(new_meta["ids"]))
            new_meta["metas"].append(m)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)
