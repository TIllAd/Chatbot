import chromadb
import ollama

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("faq")

# Test query to see scores
question = "Wie kann ich mit dem Basisticket fahren?"
response = ollama.embeddings(model="nomic-embed-text", prompt=question)
embedding = response["embedding"]

results = collection.query(
    query_embeddings=[embedding],
    n_results=5,
    include=["documents", "distances"]
)

chunks = results["documents"][0]
distances = results["distances"][0]
ids = results["ids"][0]

print(f"Query: {question}\n")
for i in range(len(chunks)):
    print(f"--- Rank {i+1} | ID: {ids[i]} | Distance: {distances[i]:.4f} ---")
    print(chunks[i][:200])
    print()