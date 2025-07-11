#core/llama_runner.py
import logging
import os
import torch
from sentence_transformers import SentenceTransformer
import ollama # For Ollama client
import time

logger = logging.getLogger(__name__)

# --- Ollama LLM Configuration ---
OLLAMA_MODEL_NAME = "mistral:7b-instruct-v0.3-q4_K_M" # Configured for quantized Mistral
OLLAMA_HOST = "http://localhost:11434" # Default Ollama host

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = None # This will hold the loaded embedding model instance

# Initialize Ollama client globally and test connectivity
try:
    start_time = time.perf_counter()
    ollama.Client(host=OLLAMA_HOST).list()
    end_time = time.perf_counter()
    logger.info(f"Ollama client initialized and connected to {OLLAMA_HOST}. Found models. (Duration: %.4f s)", end_time - start_time)
except Exception as e:
    logger.error(f"Error initializing Ollama client or connecting to {OLLAMA_HOST}: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize Ollama client: {e}. Make sure Ollama server is running and model '{OLLAMA_MODEL_NAME}' is pulled.")

try:
    start_time = time.perf_counter()
    logger.info(f"Attempting to load embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    end_time = time.perf_counter()
    logger.info("Embedding model loaded successfully (Duration: %.4f s).", end_time - start_time)

except Exception as e:
    logger.error(f"Failed to load embedding model: {e}", exc_info=True)
    raise RuntimeError(f"Failed to load required embedding model: {e}. Check dependencies and memory resources.")

# --- Embedding Function ---
def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using the loaded embedding model."""
    if embedding_model is None:
        logger.error("Embedding model not loaded. Cannot generate embedding.")
        raise RuntimeError("Embedding model not available.")
    
    start_time = time.perf_counter()
    embeddings = embedding_model.encode(text).tolist()
    end_time = time.perf_counter()
    logger.debug("Embedding generated for text (Length: %d) (Duration: %.4f s).", len(text), end_time - start_time)
    return embeddings

# --- NEW: Batch Embedding Function ---
def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts in a batch using the loaded embedding model.
    This is significantly faster for multiple texts than individual calls.
    """
    if embedding_model is None:
        logger.error("Embedding model not loaded. Cannot generate embeddings in batch.")
        raise RuntimeError("Embedding model not available.")
    
    if not texts:
        return []

    start_time = time.perf_counter()
    # The SentenceTransformer.encode() method can directly take a list of strings
    embeddings = embedding_model.encode(texts).tolist()
    end_time = time.perf_counter()
    logger.debug("Batch embeddings generated for %d texts (Duration: %.4f s).", len(texts), end_time - start_time)
    return embeddings

# --- Ollama LLM Generation Function ---
def run_llama_prompt(prompt: str, context: str = "", max_new_tokens: int = 2048, temperature: float = 0.1, top_p: float = 0.95) -> str:
    """
    Generates a response using the Ollama-served LLM with a given prompt and context.
    Args:
        prompt (str): The specific question or instruction for the LLM.
        context (str): The contextual text (e.g., retrieved chunks) the LLM should use.
        max_new_tokens (int): Maximum number of tokens to generate. (Mapped to options in Ollama)
        temperature (float): Controls randomness of generation.
        top_p (float): Controls nucleus sampling.
    Returns:
        str: The generated response from the Ollama model, or an error message.
    """
    logger.info("Running Ollama prompt. Max tokens: %d", max_new_tokens)
    logger.debug("Prompt: %s", prompt[:200] + "...")
    logger.debug("Context length: %d", len(context))
    logger.debug("Context preview: %s...", context[:200])

    messages = [
        {"role": "system", "content": "You are an expert at extracting structured information from documents and responding ONLY with valid JSON. Your entire output MUST be a single, complete JSON object. Do NOT include any conversational text, preamble, explanations, acknowledgments, or markdown outside the JSON object. If information is not available, use empty strings or lists as per the schema. Do not say 'Here is the JSON:' or similar. Strictly adhere to the requested JSON format and nothing more."},
        # {"role": "system", "content": "You are an accurate, concise, and helpful AI assistant. Answer questions directly based on the provided context. If the information is not in the context, explicitly state that you cannot answer based on the given information. Do not use external knowledge."},
    ]
    if context:
        messages.append({"role": "user", "content": f"Context for your answer:\n{context}\n\nUser Query: {prompt}"})
    else:
        messages.append({"role": "user", "content": f"User Query: {prompt}"})


    try:
        client = ollama.Client(host=OLLAMA_HOST)
        start_time = time.perf_counter()
        response = client.chat(
            model=OLLAMA_MODEL_NAME, # Uses the updated model name
            messages=messages,
            options={
                'temperature': temperature,
                'top_p': top_p,
                'num_predict': max_new_tokens,
                'num_ctx': 16384 #8192
            },
            stream=False
        )
        end_time = time.perf_counter()

        generated_content = response['message']['content'].strip()
        logger.info(f"Ollama model generation complete (Duration: %.4f s). Result length: {len(generated_content)}", end_time - start_time)
        logger.debug(f"Extracted result preview: {generated_content[:200]}...")
        return generated_content

    except ollama.ResponseError as e:
        logger.error(f"Ollama API error during LLM generation for prompt '{prompt}': {e}", exc_info=True)
        return f"Error from Ollama API: {e}. Please check Ollama server status, model availability, or context window limits."
    except Exception as e:
        logger.error(f"Failed during LLM generation for prompt '{prompt}': {e}", exc_info=True)
        return "Error generating response from LLM."
