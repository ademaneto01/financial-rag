from qdrant_client import QdrantClient, models
from models.search import SearchResult, SearchResponse
from services.embeddings import EmbeddingService


class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()

    def search(self, query: str, limit: int = 3):
        query_dense, query__sparse, query_colbert = self.embedding_service.embed_query(
            query
        )

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                {
                    "prefetch": [
                        {"query": query_dense, "using": "dense", "limit": 10},
                        {"query": query__sparse, "using": "sparse", "limit": 10},
                    ],
                    "query": models.FusionQuery(
                        fusion=models.Fusion.RRF,  # Reciprocal Rank Fusion, que é uma técnica de fusão de resultados de busca que combina os resultados de várias fontes de busca, atribuindo a cada resultado uma pontuação com base em sua posição nas listas de resultados individuais. A pontuação é calculada usando a fórmula: score = 1 / (k + rank), onde k é um parâmetro de ajuste e rank é a posição do resultado na lista de resultados individuais. O RRF é eficaz para combinar resultados de diferentes modelos de busca, como dense e sparse, e pode melhorar a relevância dos resultados finais.
                    ),
                }
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
        )
        max_score = max(result.score for result in results.points)

        search_results = [
            SearchResult(
                score=result.score
                / max_score,  ## normaliza a pontuação para ficar entre 0 e 1, dividindo pela pontuação máxima encontrada nos resultados
                text=result.payload["text"],
                metadata=result.payload["metadata"],
            )
            for result in results.points
        ]
        return SearchResponse(results=search_results)
