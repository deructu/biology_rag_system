from langchain_community.document_loaders import DirectoryLoader, TextLoader

def documents_load(path: str):
    loader = DirectoryLoader(
        path=path,
        loader_cls=TextLoader,
        glob="**/*.txt"
    )

    documents = loader.load()
    return documents