from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

COLLECTION_NAME = "biology_rag_collection"
PERSIST_DIRECTORY = "store/chromaDB"

client = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OllamaEmbeddings(
            model="nomic-embed-text-v2-moe"
        )
    )

def add_to_store(chunks):
    client.add_documents(chunks)


def get_store():
    return client
