from fastapi import FastAPI
from ollama import AsyncClient
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/similarity-score")
async def get_cos_similarity(text1: str, text2: str):
    responce = await AsyncClient().embed(
        model="nomic-embed-text-v2-moe",
        input=[text1, text2]
    )

    similarity = cosine_similarity(responce["embeddings"])[0][1]
    return {"result": float(similarity)}



