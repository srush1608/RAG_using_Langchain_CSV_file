import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

def connect_to_database():
    """Establish a connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def create_tables(connection):
    """Create necessary tables in the database."""
    cursor = connection.cursor()
    
    create_chat_table_query = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        user_input TEXT NOT NULL,
        bot_response TEXT NOT NULL
    );
    """
    cursor.execute(create_chat_table_query)

    create_embeddings_table_query = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        original_text TEXT NOT NULL,
        embedding FLOAT8[] NOT NULL
    );
    """
    cursor.execute(create_embeddings_table_query)
    connection.commit()
    cursor.close()

def store_chat(connection, user_input, bot_response):
    """Store chat history in the database."""
    cursor = connection.cursor()
    insert_query = sql.SQL("INSERT INTO chat_history (user_input, bot_response) VALUES (%s, %s)")
    cursor.execute(insert_query, (user_input, bot_response))
    connection.commit()
    cursor.close()

def store_embedding(connection, original_text, embeddings):
    """Store embedding data in the database."""
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO embeddings (original_text, embedding) VALUES (%s, %s)
    """
    # Convert the embedding to a Python list if needed
    cursor.execute(insert_query, (original_text, embeddings))
    connection.commit()
    cursor.close()

def check_existing_embedding(connection, original_text):
    """
    Check if the embedding for the given text already exists in the database.
    """
    cursor = connection.cursor()
    check_query = """
    SELECT 1 FROM embeddings WHERE original_text = %s
    """
    cursor.execute(check_query, (original_text,))
    result = cursor.fetchone()
    cursor.close()

    # If result is None, no matching embedding was found
    return result is not None

