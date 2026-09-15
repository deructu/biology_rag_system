from fastapi import FastAPI
from ollama import AsyncClient
from sklearn.metrics.pairwise import cosine_similarity
from src.core.schemas import AskRequest, AskResponse
from langchain_ollama import ChatOllama
from src.core.vectorstore import get_store

SYSTEM_PROMPT = (
    "Відповідай на запитання українською, спираючись виключно на наданий контекст. "
    "Якщо в контексті немає відповіді — так і скажи, не вигадуй."
    "Перефразуй своїми словами, не копіюй речення дослівно"
)

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

@app.get("/get-text-by-query")
async def search_in_store(query: str, count_of_texts: int):
    store = get_store()
    responce = await store.asimilarity_search(
        query=query,
        k=count_of_texts
    )

    return responce


@app.post("/ask")
async def ask(request: AskRequest) -> AskResponse:
    store = get_store()
    chunks = await store.asimilarity_search(request.query, k=1)

    context = "\n\n".join(c.page_content for c in chunks)
    sources = [c.metadata["source"] for c in chunks]

    model = ChatOllama(model="qwen2.5", validate_model_on_init=True)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Контекст:\n{context}\n\nЗапитання: {request.query}"),
    ]
    response = await model.ainvoke(messages)

    return AskResponse(answer=response.content, sources=sources)