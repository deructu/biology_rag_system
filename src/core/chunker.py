from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chuks = splitter.split_documents(documents=docs)
    return chuks
