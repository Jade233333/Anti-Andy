import chromadb

client = chromadb.PersistentClient(path="question_bank")

collection = client.get_or_create_collection(name="question_bank"
                                             metadata={"hnsw:space": "l2"})

collection.add(
    documents=["lorem ipsum...", "doc2", "doc3", ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)       