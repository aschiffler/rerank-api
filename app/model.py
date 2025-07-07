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
            list[float]: A list of relevance scores for each document, scaled between 0 and 1 using a sigmoid function.
        """
        if not documents:
            return []

        # Prepare pairs for the reranker
        pairs = [[query, doc] for doc in documents]

        # Compute scores
        # The compute_score method returns raw scores.
        raw_scores = self.reranker.compute_score(pairs)

        # Apply sigmoid function to scale scores to [0, 1]
        # Convert numpy array to torch tensor, apply sigmoid, then convert back to list
        scores = torch.sigmoid(torch.tensor(raw_scores)).tolist() #
        return scores

# Global model instance
reranker_instance: RerankerModel = None

def get_reranker_model():
    global reranker_instance
    if reranker_instance is None:
        reranker_instance = RerankerModel()
    return reranker_instance