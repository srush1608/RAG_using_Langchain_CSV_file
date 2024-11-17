import os
import requests
from dotenv import load_dotenv
from utils import load_data, setup_vector_store
from embeddings import generate_embedding_vector_store
from database import connect_to_database, store_embedding, check_existing_embedding, store_chat
from langchain_cohere import CohereEmbeddings
from groq import Groq

# Load environment variables from .env file
load_dotenv()

def chatbot_response(user_input):
    """
    Generate chatbot responses by querying the vector store.
    """
    # Generate an embedding for the user input using LangChain
    embeddings_model = CohereEmbeddings(model="embed-english-v3.0")
    user_input_embedding = embeddings_model.embed_query(user_input)

    # Retrieve all embeddings from the database
    connection = connect_to_database()
    cursor = connection.cursor()
    cursor.execute("SELECT original_text, embedding FROM embeddings")
    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    # Find the most similar embedding
    best_match = None
    best_similarity = float('-inf')

    for original_text, embedding in rows:
        # Calculate similarity
        similarity = calculate_similarity(user_input_embedding, embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = original_text

    # If a best match is found, send it to the LLM for a response
    if best_match:
        response = generate_response_with_groq(best_match)
        return response
    else:
        return "I'm sorry, I couldn't find relevant information."

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    from numpy import dot
    from numpy.linalg import norm
    return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

def generate_response_with_groq(relevant_info):
    """Send the relevant information to the Groq API and get a response."""
    client = Groq()
    
    # Prepare the messages for the Groq API
    messages = [{"role": "user", "content": relevant_info}]
    
    # Create a completion request
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    # Collect and return the response
    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""
    
    return response_text.strip()

def main():
    # Load data
    df = load_data('world_happiness.csv')
    
    # Set up vector store
    vector_store = setup_vector_store()

    # Establish a database connection outside the loop
    connection = connect_to_database()
    
    # Process each row in the DataFrame
    for index, row in df.iterrows():
        # Convert the row to a string
        original_text = row.to_string()

        # Check if embeddings for this text already exist in the database
        if check_existing_embedding(connection, original_text):
            print(f"Embedding already exists for: {original_text}")
            continue  

        # Generate embedding for the text
        embeddings = generate_embedding_vector_store(text=original_text)
        print("Embeddings generated:", embeddings)

        # Store the embedding in the database
        store_embedding(connection, original_text, embeddings)

    # Close the database connection after processing
    # connection.close()  # Do not close the connection here

    # Uncomment below for chatbot interaction
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = chatbot_response(user_input)
        print(f"Bot: {response}")
        
        # Store chat history
        store_chat(connection, user_input, response)  # Store user input and bot response

    # Close the database connection after chat
    connection.close()  # Close the connection here

if __name__ == "__main__":
    main()
