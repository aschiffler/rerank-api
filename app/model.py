import torch
from FlagEmbedding import FlagReranker

class RerankerModel:
    def __init__(self):
        # Load the reranker model
        # You can specify use_fp16=True for faster inference if you have a GPU
        # If running on CPU, keep use_fp16=False or omit it
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        # Ensure the model is moved to GPU if available
        if torch.cuda.is_available():
            self.reranker.model.cuda()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Reranker model loaded on {self.device} successfully!")

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Reranks a list of documents based on a query.

        Args:
            query (str): The search query.
            documents (list[str]): A list of documents to rerank.

        Returns:
            list[float]: A list of relevance scores for each document.
        """
        if not documents:
            return []

        # Prepare pairs for the reranker
        pairs = [[query, doc] for doc in documents]

        # Compute scores
        # The compute_score method returns raw scores, which can be mapped to [0,1]
        # using a sigmoid function if needed, but often raw scores are sufficient for ranking.
        scores = self.reranker.compute_score(pairs)
        return scores # Convert numpy array to list for JSON serialization

# Global model instance
reranker_instance: RerankerModel = None

def get_reranker_model():
    global reranker_instance
    if reranker_instance is None:
        reranker_instance = RerankerModel()
    return reranker_instance