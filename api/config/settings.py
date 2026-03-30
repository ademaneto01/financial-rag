from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  ## importante porque no arquivo .env tem mais variáveis do que as que estão aqui, e se não tiver isso, ele vai dar erro de validação
    )

    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "financial"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"
    colbert_model: str = "colbert-ir/colbertv2.0"
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"


settings = Settings()
