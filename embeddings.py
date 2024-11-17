# embeddings.py
import os
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
load_dotenv()


def generate_embedding_vector_store(text):
    """Generate embeddings for the given text."""
    embeddings_model = CohereEmbeddings(model="embed-english-v3.0")
    embedding_vector = embeddings_model.embed_query(text)  # Generate embedding for the text
    return embedding_vector