import streamlit as st
import time
import chromadb
from chromadb.config import Settings
import os 
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def config_llm(temperature:int):
  '''LLM api calling using Langchain  '''
  # Steps for students:
  # - Go to https://aistudio.google.com/app/apikey and generate your Gemini API key.
  # - Add the necessary packages to your requirements.txt:
  #    langchain
  #    langchain-google-genai
  # - Run the following command to install them:
  #     !pip install -r requirements.txt
  # - Follow the official integration guide for LangChain + Google Generative AI:
  #     https://python.langchain.com/docs/integrations/chat/google_generative_ai/
  return "llm" #Should return the LLM

def add_prompt_to_message(query:str, prompt:str, context:str):
  """Formats the system and user messages with prompt and query for LLM input."""
  return "message" # Should return the system and user messages

#Extracting information from pdfs
def extract_text_from_pdfs_in_folder(documents_path='Documents'):
  """Extracts and returns text from all PDF files in the specified folder."""
    # Important:
    # - Create a folder named "Documents" on Google Colab (files) where you'll store the PDF files.
  return "pdf_texts" # Should return the texts from files

# Split the text into smaller overlapping chunks
def split_text_into_chunks(texts, chunk_size=500):
    """Splits a list of texts into smaller overlapping chunks for processing."""
    # Create a text splitter that breaks text into chunks with specified size
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    # Initialize an empty list to hold all the chunks
    chunks = []
    
    # Loop through each text input and split it into chunks
    for text in texts:
        chunks.extend(splitter.split_text(text)) # Add the resulting chunks to the list
    
    # Return the full list of chunks
    return chunks

# Generate Embeddings
def generate_embeddings(text_chunks):
    """Generates embeddings for each text chunk using an open-source embedding model."""
    # Initialize the HuggingFaceEmbeddings model
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Convert text chunks into vector embeddings
    embeddings = model.embed_documents(text_chunks)
    # Print the embeddings to verify they are being generated correctly
    return embeddings

def store_chunks_in_chromadb(documents_path="Documents", collection_name="collection"):
    """Extracts text from PDFs, splits it into chunks, creates embeddings, and stores them in ChromaDB."""
    # Extract text from PDFs
    pdf_texts = extract_text_from_pdfs_in_folder(documents_path)
    
    # Remove any empty texts
    pdf_texts = [text for text in pdf_texts if text]
    
    # Split the text into chunks
    chunks = split_text_into_chunks(pdf_texts)
   
    embeddings = generate_embeddings(chunks)  # Create embeddings for each chunk
    
    ids = [f"chunk_{i}" for i in range(len(chunks))] # Generate unique IDs for each chunk
    
    chroma_client = chromadb.Client(Settings(persist_directory="./chroma_store",
    anonymized_telemetry=False)) # Initialize Chroma Client (in-memory by default)
    
    # Get or create the collection in ChromaDB
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Add the chunks, embeddings, and metadata to the collection
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": "Documents"} for _ in chunks]
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB.")

# Retrieve Relevant Context from AI Search (from pre-existing ChromaDB)
def retrieve_context(query):
    """Fetches the most relevant text chunks based on the user query."""
    
    # Generate embeddings for the query using the generate_embeddings function
    query_embedding = generate_embeddings([query])  # Pass the query as a list to generate embeddings
    
    chroma_client = chromadb.Client(Settings(persist_directory="./chroma_store",
    anonymized_telemetry=False))

    # Get or create the collection in ChromaDB
    collection = chroma_client.get_or_create_collection(name="collection")

    #Check if the collection is empty or not
    if collection.count() == 0:
      store_chunks_in_chromadb()
    else:
      print(f"Collection count: {collection.count()}")
    
    # Query ChromaDB collection for the most similar chunks to the query embedding
    results = collection.query(
        query_embeddings=query_embedding,  # Query with the generated embedding
        n_results=2  # Number of similar results to retrieve
    )
    
    # Extract the most relevant documents from the results
    retrieved_documents = results['documents']

    print(f"Retrieved documents: {retrieved_documents}")
    
    # Check if relevant documents were found
    if retrieved_documents:
        # Flatten the list of documents and return them concatenated as context
        flat_documents = [doc for sublist in retrieved_documents for doc in sublist]  # Flatten list
        print("Retrieved documents:", flat_documents)  # Debugging line
        return "\n".join(flat_documents)  # Join all documents as a single string
    
    # If no relevant documents found, return a fallback message
    return "Sorry, no relevant context found. Please try again with a different query."

# Generate AI Response
def generate_response(query):
    """Generates a response using an open-source LLM with the retrieved context."""
    # Steps for students:
    # - Format input by combining user query with retrieved context
    # - Generate response using the LLM
    return "Sorry, I couldn't generate an answer at this time."
    
# Streamlit User Interface
def main():
  # Create the streamlit interface for the chatbot with this documentation 
  # https://docs.streamlit.io/develop/api-reference
  st.title("EY AI Challenge")

if __name__ == "__main__":
    main()
