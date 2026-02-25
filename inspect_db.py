import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("faq")

results = collection.get()

for i, doc in enumerate(results["documents"]):
    print(f"--- Chunk {i+1} ---")
    print(doc)
    print()