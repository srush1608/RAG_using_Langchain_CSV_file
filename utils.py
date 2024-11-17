import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector  
from langchain_cohere import CohereEmbeddings
from database import connect_to_database, store_embedding

load_dotenv()

file_path = "world_happiness.csv"

# Load the dataset
def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

# Set up PGVector
def setup_vector_store():
    """Set up PGVector for embedding storage and retrieval."""
    connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    embedding_model = CohereEmbeddings(model="embed-english-v3.0")
    return PGVector(connection_string=connection_string, embedding_function=embedding_model.embed_query)

# Store embeddings
def store_embedding_in_db(original_text, embeddings):
    """Store embeddings in the database."""
    connection = connect_to_database()
    try:
        store_embedding(connection, original_text, embeddings)
    finally:
        connection.close()
