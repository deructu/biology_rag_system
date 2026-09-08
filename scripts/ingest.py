from src.core.loader import documents_load
from src.core.chunker import chunking_documents
from src.core.vectorstore import add_to_store


documents = documents_load("data/docs")

chunks = chunking_documents(documents)

db = add_to_store(chunks)

print(f"Documetns {len(documents)} its {len(chunks)} loaded to store")