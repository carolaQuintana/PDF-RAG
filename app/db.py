from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

db = QdrantClient(url="http://localhost:6333")

collection_name = "rag_project_db"

try:
    collections = db.get_collections()
    existing_collections = [col.name for col in collections.collections]
    if collection_name not in existing_collections:
        db.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE, on_disk=True
            ),
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")
except UnexpectedResponse as e:
    print(f"Error checking collections: {e}")
