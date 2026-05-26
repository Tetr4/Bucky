import time
import uuid

import chromadb


class VectorDatabase:
    def __init__(self, db_path="./robot_memory_db"):
        print("Initializing Local Long-Term Memory...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="long_term_memory")

    def add_memory(self, facts, embedding):
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[facts],
            metadatas=[{"timestamp": time.time()}]
        )

    def query_memory(self, embedding, n_results=2):
        if self.collection.count() == 0:
            return ""
        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        if results['documents'] and results['documents'][0]:
            return " | ".join(results['documents'][0])
        return ""
