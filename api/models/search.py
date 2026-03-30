from typing import List
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int = 3  ## define quantos resultados a API deve retornar, o default é 3, mas pode ser alterado pelo usuário


class SearchResult(BaseModel):
    score: float
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
