from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from .model import get_reranker_model, RerankerModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BGE Reranker API",
    description="REST API for BGE Reranker v2-m3 model",
    version="1.0.0"
)

# Pydantic model for request body
class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int

# Pydantic model for response body
class RerankResultItem(BaseModel):
    index: int
    relevance_score: float

class RerankResponse(BaseModel):
    results: list[RerankResultItem]

@app.on_event("startup")
async def startup_event():
    """
    Load the model when the FastAPI application starts.
    """
    logger.info("Starting up and loading reranker model...")
    get_reranker_model()
    logger.info("Reranker model loaded.")

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(
    request: RerankRequest,
    reranker: RerankerModel = Depends(get_reranker_model),
):
    """
    Reranks a list of documents based on a given query.
    """
    try:
        scores = reranker.rerank(request.query, request.documents)
        # Build results as list of {index, relevance_score}
        results = [RerankResultItem(index=i, relevance_score=score) for i, score in enumerate(scores)]
        # Sort by relevance_score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if request.top_n is not None:
            results = results[:request.top_n]
        logger.info(f"Scores: {scores}")            
        return RerankResponse(results=results)
    except Exception as e:
        logger.error(f"Error during reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    try:
        reranker = get_reranker_model()
        model_loaded = reranker is not None
    except Exception:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}