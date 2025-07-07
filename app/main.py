from fastapi import FastAPI, HTTPException, Depends
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

# Pydantic model for response body
class RerankResponse(BaseModel):
    scores: list[float]

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
    reranker: RerankerModel = Depends(get_reranker_model)
):
    """
    Reranks a list of documents based on a given query.
    """
    try:
        scores = reranker.rerank(request.query, request.documents)
        return RerankResponse(scores=scores)
    except Exception as e:
        logger.error(f"Error during reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "model_loaded": reranker_instance is not None}