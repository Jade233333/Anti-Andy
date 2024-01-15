import chromadb

client = chromadb.PersistentClient(path="question_bank")

question_bank = client.get_or_create_collection(name="question_bank",
                                                metadata={"hnsw:space": "l2"})

question_bank.upsert(
    documents=["testing, testing"],
    metadatas=[{"paper": 1, "page": 1, "question": 1}],
    ids=["id1"]
) 


result = question_bank.query(
    query_texts="testing",
    n_results=1,
    include=["documents","distances","metadatas"]
)


print(result)
