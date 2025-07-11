import chromadb
from chromadb.utils import embedding_functions
import logging
import os
import uuid
import time
from .llama_runner import get_embedding, get_embeddings_batch

# Import the embedding functions from llama_runner (where the embedding model is loaded)
# IMPORTANT: Import get_embeddings_batch for performance optimization
from core.llama_runner import get_embedding, get_embeddings_batch

logger = logging.getLogger(__name__)

# Directory where ChromaDB will store its data. Make sure it's persistent.
CHROMA_DB_PATH = "chroma_db_data"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Initialize ChromaDB client
try:
    start_time = time.perf_counter()
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    end_time = time.perf_counter()
    logger.info("ChromaDB client initialized with path: %s (Duration: %.4f s)", CHROMA_DB_PATH, end_time - start_time)
except Exception as e:
    logger.error("Error initializing ChromaDB client: %s", e, exc_info=True)
    raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

# Custom Embedding Function for ChromaDB to use your loaded model
class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, texts: embedding_functions.Documents) -> embedding_functions.Embeddings:
        # Instead of a list comprehension making individual calls,
        # we now pass the entire list of texts for batch processing.
        # This will significantly speed up embedding generation for multiple chunks.
        if not texts: # Added a check for empty texts
            return []
        embeddings = get_embeddings_batch(texts) # Use batch embedding
        return embeddings

# Instantiate the custom embedding function once
_custom_ef = CustomEmbeddingFunction()
logger.info("CustomEmbeddingFunction for ChromaDB created.")

def get_or_create_collection(collection_name: str):
    """Gets an existing collection or creates a new one with a custom embedding function."""
    start_time = time.perf_counter()
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=_custom_ef
        )
        end_time = time.perf_counter()
        logger.info("ChromaDB collection '%s' accessed/created successfully (Duration: %.4f s).", collection_name, end_time - start_time)
        return collection
    except Exception as e:
        logger.error("Error accessing/creating ChromaDB collection '%s': %s", collection_name, e, exc_info=True)
        raise RuntimeError(f"Failed to access/create ChromaDB collection: {e}")

def add_documents_to_chroma(
    collection_name: str,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str] = None
):
    """Adds documents (text chunks) and their metadata to a specified ChromaDB collection."""
    if not documents:
        logger.warning("No documents provided to add to ChromaDB collection '%s'.", collection_name)
        return

    collection = get_or_create_collection(collection_name)
    
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]

    start_time = time.perf_counter()
    try:
        # Using upsert is safer as it handles adding or updating documents if an ID were to ever conflict.
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        end_time = time.perf_counter()
        logger.info("Added/updated %d documents in ChromaDB collection '%s' (Duration: %.4f s).", len(documents), collection_name, end_time - start_time)
        logger.debug("First 5 documents added to ChromaDB: %s", documents[:5])
    except Exception as e:
        logger.error("Error adding documents to ChromaDB collection '%s': %s", collection_name, e, exc_info=True)
        raise RuntimeError(f"Failed to add documents to ChromaDB: {e}")

def query_chroma(
    collection_name: str,
    query_texts: list[str],
    n_results: int = 5,
    where_metadata: dict = None
) -> dict:
    """Queries a ChromaDB collection for relevant documents."""
    collection = get_or_create_collection(collection_name)
    
    start_time = time.perf_counter()
    try:
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where_metadata
        )
        end_time = time.perf_counter()
        logger.info("Queried ChromaDB collection '%s' for %d results (Duration: %.4f s).", collection_name, n_results, end_time - start_time)
        logger.debug("ChromaDB query raw results: %s", results)
        return results
    except Exception as e:
        logger.error("Error querying ChromaDB collection '%s': %s", collection_name, e, exc_info=True)
        raise RuntimeError(f"Failed to query ChromaDB: {e}")

def clear_collection(collection_name: str):
    """Deletes a specified ChromaDB collection."""
    start_time = time.perf_counter()
    try:
        client.delete_collection(name=collection_name)
        end_time = time.perf_counter()
        logger.info("ChromaDB collection '%s' deleted successfully (Duration: %.4f s).", collection_name, end_time - start_time)
    except Exception as e:
        logger.warning("Could not delete/clear ChromaDB collection '%s' (might not exist or error): %s", collection_name, e)