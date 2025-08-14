import os
import re
import json
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from flask import Flask, render_template, request, jsonify, Response
from ollama import Client
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# -------------------- CONFIG --------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "memory_index"
INDEX_DIR.mkdir(exist_ok=True)
META_FILE = INDEX_DIR / "meta.json"
INDEX_FILE = INDEX_DIR / "vectors.faiss"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

# Force CPU by default to avoid CUDA errors on unsupported GPUs (override with FORCE_CPU=0)
FORCE_CPU = os.getenv("FORCE_CPU", "1") == "1"
if FORCE_CPU:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# -------------------- CLIENTS --------------------
# Select device safely
_device = "cpu"
if not FORCE_CPU:
    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
    except Exception:
        _device = "cpu"

embedder = SentenceTransformer(EMBED_MODEL_NAME, device=_device)
client = Client(host=OLLAMA_HOST)

# -------------------- UTILS --------------------
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    out = []
    reader = PdfReader(str(path))
    for p in reader.pages:
        try:
            out.append(p.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []
    # rough token->char conversion
    size = max_tokens * 4
    step = size - (overlap * 4)
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += max(step, 1)
    return chunks


def safe_load_meta():
    if not META_FILE.exists() or META_FILE.stat().st_size == 0:
        meta = {"ids": [], "metas": []}
        META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        meta = {"ids": [], "metas": []}
        META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta


def load_or_create_index(dim: int):
    if INDEX_FILE.exists():
        try:
            idx = faiss.read_index(str(INDEX_FILE))
            return idx, safe_load_meta()
        except Exception:
            INDEX_FILE.unlink(missing_ok=True)
    return faiss.IndexFlatIP(dim), safe_load_meta()


def add_documents(paths: List[Path]) -> Tuple[int, List[dict]]:
    texts = []
    metas = []
    for p in paths:
        raw = read_pdf(p) if p.suffix.lower() == ".pdf" else read_txt(p)
        for ci, ch in enumerate(chunk_text(raw)):
            texts.append(ch)
            metas.append({"source": str(p), "chunk": ci, "length": len(ch), "query_count": 0})
    if not texts:
        return 0, []
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(emb)
    idx, meta = load_or_create_index(emb.shape[1])
    idx.add(emb)
    start = len(meta["ids"])
    for i, m in enumerate(metas):
        meta["ids"].append(start + i)
        meta["metas"].append(m)
    faiss.write_index(idx, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return len(texts), metas


def retrieve(query: str, top_k: int = 5) -> List[Tuple[dict, float, str]]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    idx, meta = load_or_create_index(q_emb.shape[1])
    if idx.ntotal == 0:
        return []
    D, I = idx.search(q_emb, top_k)
    results = []
    for score, idx_id in zip(D[0], I[0]):
        m = meta["metas"][int(idx_id)]
        src = Path(m["source"]) 
        raw = read_pdf(src) if src.suffix.lower() == ".pdf" else read_txt(src)
        chunks = chunk_text(raw)
        ch_txt = chunks[m.get("chunk", 0)] if chunks else ""
        results.append((m, float(score), ch_txt))
    return results


def web_search(query: str, num_results: int = 5) -> str:
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query, safesearch="Moderate", max_results=num_results):
            results.append(f"{r['title']} - {r['href']}\n{r['body']}")
        return "\n\n".join(results)


def scrape_url_details(url: str) -> str:
    try:
        res = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if not res.ok:
            return f"Failed to fetch URL (status {res.status_code})"
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc = ""
        mt = soup.find("meta", attrs={"name": "description"})
        if mt and mt.get("content"):
            desc = mt["content"]
        heads = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])][:40]
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")][:20]
        return (
            f"Page Title: {title}\n"
            f"Meta Description: {desc}\n"
            f"Headings: {heads}\n"
            f"Sample Text: {' '.join(paras)}\n"
        )
    except Exception as e:
        return f"Failed to fetch URL: {e}"


def extract_ollama_content(resp) -> str:
    try:
        if isinstance(resp, dict):
            if "message" in resp and isinstance(resp["message"], dict):
                return (resp["message"].get("content") or "").strip()
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return (resp.message.content or "").strip()
        return str(resp).strip()
    except Exception:
        return str(resp)


def chat_llm(messages, stream=False):
    return client.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=stream,
        options={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_ctx": NUM_CTX,
        },
    )


def enforce_english_output(text: str) -> str:
    try:
        # Translate if any non-ASCII chars are present or language is not English
        if re.search(r"[^\x00-\x7F]", text) or (text.strip() and extract_ollama_content(chat_llm(messages=[{"role":"system","content":"Detect language code only: return 'en' for English, or two-letter ISO code for others."},{"role":"user","content":text}], stream=False)).lower() != 'en'):
            tr = chat_llm(
                messages=[
                    {"role": "system", "content": "Translate to English only. Keep formatting and code blocks. Do not add any non-English text or extra commentary."},
                    {"role": "user", "content": text},
                ],
                stream=False,
            )
            return extract_ollama_content(tr)
        return text
    except Exception:
        return text

def extract_plain_text_from_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        # Prefer article/main
        main_nodes = soup.find_all(["article", "main"]) or soup.find_all("section") or [soup.body or soup]
        parts = []
        for node in main_nodes:
            text = node.get_text(" ", strip=True)
            if text and len(text) > 200:
                parts.append(text)
        if not parts:
            parts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n\n".join([p for p in parts if p])
    except Exception:
        return ""


def retrieve_from_texts(query: str, texts: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
    if not texts:
        return []
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    sims = np.dot(embeddings, q_emb[0])
    order = np.argsort(-sims)[: top_k]
    return [(int(i), float(sims[int(i)])) for i in order]


def build_realtime_context(query: str, num_results: int = 6, max_pages: int = 4, top_k: int = 12) -> str:
    """Search the web, fetch top pages, rank relevant excerpts, and return a context block with inline citations and a Sources list."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, safesearch="Moderate", max_results=num_results):
            results.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
    if not results:
        return ""
    texts = []
    owners = []  # index -> source id
    for i, res in enumerate(results[:max_pages]):
        url = res.get("url")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if not resp.ok or not resp.text:
                continue
            page_text = extract_plain_text_from_html(resp.text)
            for chunk in chunk_text(page_text, max_tokens=400, overlap=60):
                if chunk.strip():
                    texts.append(chunk)
                    owners.append(i)
        except Exception:
            continue
    if not texts:
        # Fallback to just result snippets
        snippets = [f"[{i+1}] {r['title']} - {r['url']}\n{r['snippet']}" for i, r in enumerate(results)]
        return "Search results:\n" + "\n\n".join(snippets)

    ranked = retrieve_from_texts(query, texts, top_k=top_k)
    # Build context with inline citations
    blocks = []
    used = set()
    for idx, _ in ranked:
        src_id = owners[idx]
        used.add(src_id)
        excerpt = texts[idx]
        blocks.append(f"[{src_id+1}] {excerpt}")

    sources = [f"[{i+1}] {r['title']} - {r['url']}" for i, r in enumerate(results) if i in used]
    context = "Relevant web excerpts with citations:\n" + "\n\n---\n\n".join(blocks)
    if sources:
        context += "\n\nSources:\n" + "\n".join(sources)
    return context

# -------------------- FLASK APP --------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secret_key")


@app.route("/")
def home():
    return render_template("index.html")


# Persistent knowledge ingestion
@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        if not request.content_type or not request.content_type.startswith("multipart/form-data"):
            return jsonify({"error": "Use multipart/form-data with files"}), 400
        files = request.files.getlist("files") or request.files.getlist("file")
        if not files:
            return jsonify({"error": "No files provided"}), 400
        saved = []
        for f in files:
            name = getattr(f, "filename", "")
            if not name or not allowed_file(name):
                continue
            safe = name.replace("..", "_")
            dest = UPLOAD_FOLDER / safe
            f.save(dest)
            saved.append(dest)
        if not saved:
            return jsonify({"error": "No valid files"}), 400
        added_total = 0
        for p in saved:
            added, _ = add_documents([p])
            added_total += added
        return jsonify({"success": True, "files": [p.name for p in saved], "added_chunks": added_total})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}
        q = data.get("question", "")
        realtime = bool(data.get("realtime"))
        if not isinstance(q, str):
            q = str(q)
        q = q.strip()
        if not q:
            return jsonify({"error": "Question cannot be empty"}), 400

        contexts = []
        # Real-time context if requested or if it looks like a fresh query
        trigger_terms = ("latest", "today", "now", "current", "breaking", "live")
        if realtime or any(t in q.lower() for t in trigger_terms):
            rtc = build_realtime_context(q)
            if rtc:
                contexts.append(rtc)

        # URL context
        for url in re.findall(r"(https?://\S+)", q):
            contexts.append(f"Extracted content from URL {url}:\n{scrape_url_details(url)}")
        # Local RAG context
        meta = safe_load_meta()
        if meta.get("metas"):
            docs = retrieve(q, top_k=4)
            if docs:
                contexts.append("\n\n---\n\n".join([f"Source: {d[0]['source']} (chunk {d[0]['chunk']})\n{d[2]}" for d in docs]))

        full_context = "\n\n===\n\n".join(contexts) if contexts else ""
        system_message = (
            "You are a helpful AI assistant. Always respond in English.\n"
            "Translate the user's question to English implicitly if needed. Be precise and factual.\n"
            "Do not include any non-English text; if sources or quotes are not in English, translate them.\n"
            "When using URLs or web results, cite sources as [1], [2] and add a Sources list at the end.\n"
        )
        if full_context:
            system_message += f"Context:\n{full_context}\n"
        else:
            system_message += "No additional context available.\n"

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": q}]
        resp = chat_llm(messages=messages, stream=False)
        answer = extract_ollama_content(resp)
        if "NEED_WEB_SEARCH" in answer or not answer.strip():
            # Use richer realtime context instead of raw DDG text
            rtc = build_realtime_context(q)
            web_prompt = [
                {"role": "system", "content": (
                    "Answer in English using the provided real-time web excerpts below. "
                    "Cite sources as [1], [2] and add a Sources list at the end. If unsure, say 'I don't know'."
                )},
                {"role": "user", "content": f"Real-time Web Context:\n{rtc}\n\nOriginal Question: {q}"}
            ]
            resp2 = chat_llm(messages=web_prompt, stream=False)
            answer = extract_ollama_content(resp2)
        answer = enforce_english_output(answer)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    try:
        q = ""
        files = []
        realtime = False
        if request.content_type and request.content_type.startswith("multipart/form-data"):
            q = request.form.get("question", "")
            realtime = request.form.get("realtime") in ("1", "true", "True")
            files = request.files.getlist("files") or request.files.getlist("file")
        else:
            data = request.get_json() or {}
            q = data.get("question", "")
            realtime = bool(data.get("realtime"))
        if not isinstance(q, str):
            q = str(q)
        q = q.strip()
        if not q:
            def gen_empty():
                yield f"data: {json.dumps({'error': 'Question cannot be empty'})}\n\n"
            return Response(gen_empty(), mimetype='text/event-stream')

        contexts = []
        # Real-time context
        trigger_terms = ("latest", "today", "now", "current", "breaking", "live")
        if realtime or any(t in q.lower() for t in trigger_terms):
            rtc = build_realtime_context(q)
            if rtc:
                contexts.append(rtc)

        # Ephemeral attachments context
        if files:
            for f in files:
                name = getattr(f, 'filename', '') or 'file'
                ext = Path(name).suffix.lower()
                raw = f.read() or b""
                text = raw.decode('utf-8', errors='ignore') if ext in {'.txt', '.md'} else ""
                if ext == '.pdf':
                    try:
                        reader = PdfReader(f)
                        text = "\n".join([(p.extract_text() or "") for p in reader.pages])
                    except Exception:
                        text = ""
                if text:
                    chunks = chunk_text(text)
                    contexts.append(f"Attached: {name}\n" + "\n\n---\n\n".join(chunks[:3]))
        # URL context
        for url in re.findall(r"(https?://\S+)", q):
            contexts.append(f"Extracted content from URL {url}:\n{scrape_url_details(url)}")
        # Local RAG
        meta = safe_load_meta()
        if meta.get("metas"):
            docs = retrieve(q, top_k=4)
            if docs:
                contexts.append("\n\n---\n\n".join([f"Source: {d[0]['source']} (chunk {d[0]['chunk']})\n{d[2]}" for d in docs]))

        full_context = "\n\n===\n\n".join(contexts) if contexts else ""
        system_message = (
            "You are a helpful AI assistant. Always respond in English.\n"
            "Translate the user's question to English implicitly if needed. Be precise and factual.\n"
            "Do not include any non-English text; if sources or quotes are not in English, translate them.\n"
            "When using URLs or web results, cite sources as [1], [2] and add a Sources list at the end.\n"
        )
        if full_context:
            system_message += f"Context:\n{full_context}\n"
        else:
            system_message += "No additional context available.\n"

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": q}]

        # Compute full answer first, then stream English-only chunks
        resp = chat_llm(messages=messages, stream=False)
        answer = extract_ollama_content(resp)
        if "NEED_WEB_SEARCH" in answer or not answer.strip():
            rtc = build_realtime_context(q)
            web_prompt = [
                {"role": "system", "content": (
                    "Answer in English using the provided real-time web excerpts below. "
                    "Cite sources as [1], [2] and add a Sources list at the end. If unsure, say 'I don't know'."
                )},
                {"role": "user", "content": f"Real-time Web Context:\n{rtc}\n\nOriginal Question: {q}"}
            ]
            resp2 = chat_llm(messages=web_prompt, stream=False)
            answer = extract_ollama_content(resp2)
        answer = enforce_english_output(answer)

        def gen_stream(txt: str, size: int = 800):
            for i in range(0, len(txt), size):
                yield f"data: {json.dumps({'answer': txt[i:i+size]})}\n\n"
        return Response(gen_stream(answer), mimetype='text/event-stream')
    except Exception as e:
        def gen_err():
            yield f"data: {json.dumps({'error': 'Error: ' + str(e)})}\n\n"
        return Response(gen_err(), mimetype='text/event-stream')


@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        data = request.get_json() or {}
        answer = (data.get("answer") or "").strip()
        if not answer:
            return jsonify({"error": "No answer provided"}), 400
        sug_prompt = f"Generate 3 concise and relevant follow-up questions based on this answer:\n{answer}"
        sug_resp = chat_llm(messages=[{"role": "user", "content": sug_prompt}], stream=False)
        suggestions_text = extract_ollama_content(sug_resp)
        suggestions_text = enforce_english_output(suggestions_text)
        suggestions = [s.strip("-• ") for s in suggestions_text.split("\n") if s.strip()]
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.before_request
def log_request_info():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {request.method} {request.path} from {request.remote_addr}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, threaded=True, use_reloader=False, port=port)
