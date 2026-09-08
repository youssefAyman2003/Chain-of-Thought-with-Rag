from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from src.schemas import AskRequest, AskResponse
from src.state import RAGCoTState

app = FastAPI(
    title="Chain-of-Thought RAG",
    description="RAG-based question answering with step-by-step reasoning",
    version="0.1.0",
)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from src.workflow import graph
        _graph = graph
    return _graph


# ── API routes (must come before static mount) ──────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        state = RAGCoTState(question=request.question)
        final = get_graph().invoke(state)

        if isinstance(final, dict):
            sub_steps = final.get("sub_steps", [])
            answer = final.get("answer", "")
            retrieved_docs = final.get("retrieved_docs", [])
        else:
            sub_steps = final.sub_steps
            answer = final.answer
            retrieved_docs = final.retrieved_docs

        sources = [doc.page_content for doc in retrieved_docs]

        return AskResponse(
            question=request.question,
            sub_steps=sub_steps,
            answer=answer,
            sources=sources,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Frontend static files ───────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

if os.path.isdir(FRONTEND_DIR):
    # Serve index.html at root
    @app.get("/")
    def root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # Serve all other static assets (css, js, images, etc.)
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
else:
    # Fallback if frontend folder is missing
    @app.get("/")
    def root():
        return {
            "message": "Chain-of-Thought RAG API",
            "docs": "/docs",
            "endpoints": {"health": "GET /health", "ask": "POST /ask"},
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)