from fastapi import FastAPI
from routers import search, rag

app = FastAPI(title="Financial Search API", version="1.0")

app.include_router(search.router)
app.include_router(rag.router)


@app.get("/")
def root():
    return {"status": "online"}
