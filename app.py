import streamlit as st
import time
import chromadb
from chromadb.config import Settings
import os 
from pypdf import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

def config_llm(temperature:int=0):
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
  llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # ou "gemini-pro" se quiseres mais contexto
        temperature=temperature,
        max_tokens=None,
        timeout=30,
        max_retries=2
  )
  return llm

def _extract_text_from_pdf(filepath: str) -> str:
    """Extracts text from a single PDF file."""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF {filepath}: {e}")
    return text

def _extract_text_from_excel(filepath: str) -> str:
    """Extracts data from an Excel file and converts it to a string."""
    text = ""
    try:
        # Read all sheets from the Excel file
        xls = pd.ExcelFile(filepath)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Convert DataFrame to a string representation, suitable for text extraction
            # You might want to customize this further based on your Excel data structure
            text += f"\n--- Sheet: {sheet_name} ---\n"
            text += df.to_string(index=False, header=True, na_rep="") # Convert to string, no index, with header
            text += "\n"
    except Exception as e:
        print(f"Error extracting text from Excel {filepath}: {e}")
    return text

def _extract_text_from_txt(filepath: str) -> str:
    """Extracts text from a plain text file."""
    text = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error extracting text from TXT {filepath}: {e}")
    return text

# --- Main function for document categorization and extraction ---
def extract_and_categorize_documents(documents_path='Documents') -> dict:
    """
    Extracts text from various file types in the specified folder and categorizes them.

    Args:
        documents_path (str): The path to the folder containing documents.

    Returns:
        dict: A dictionary with categories as keys and lists of extracted texts as values.
              Categories: "excels", "text_files", "pdfs_cvs", "pdfs_ey_articles".
    """
    categorized_texts = {
        "excels": [],
        "text_files": [],
        "pdfs_cvs": [],
        "pdfs_ey_articles": []
    }

    if not os.path.exists(documents_path):
        print(f"Error: Directory '{documents_path}' not found.")
        return categorized_texts

    print(f"Scanning directory: {documents_path}")
    for filename in os.listdir(documents_path):
        filepath = os.path.join(documents_path, filename)
        lower_filename = filename.lower()

        if os.path.isfile(filepath):
            extracted_text = ""
            category_assigned = False

            # Handle Excel Files
            if lower_filename.endswith(('.xlsx', '.xls')):
                extracted_text = _extract_text_from_excel(filepath)
                if extracted_text:
                    categorized_texts["excels"].append(extracted_text)
                    print(f"Categorized '{filename}' as 'excels'.")
                    category_assigned = True

            # Handle Text Files
            elif lower_filename.endswith('.txt'):
                extracted_text = _extract_text_from_txt(filepath)
                if extracted_text:
                    categorized_texts["text_files"].append(extracted_text)
                    print(f"Categorized '{filename}' as 'text_files'.")
                    category_assigned = True

            # Handle PDF Files (and categorize them)
            elif lower_filename.endswith('.pdf'):
                extracted_text = _extract_text_from_pdf(filepath)
                if extracted_text:
                    # Categorization logic for PDFs based on filename hints
                    # "articles have spaces in their title or special characters or EY"
                    # We'll use 'EY' as a strong indicator for articles.
                    # Otherwise, assume it's a CV if it's a PDF and not clearly an article.
                    if 'ey' in lower_filename: # Primary indicator for articles
                        categorized_texts["pdfs_ey_articles"].append(extracted_text)
                        print(f"Categorized '{filename}' as 'pdfs_ey_articles'.")
                    elif 'cv' in lower_filename or 'curriculum' in lower_filename or 'partner' in lower_filename: # Common for CVs
                         categorized_texts["pdfs_cvs"].append(extracted_text)
                         print(f"Categorized '{filename}' as 'pdfs_cvs'.")
                    else: # Fallback for PDFs if no clear indicator, default to articles (or consider adding a 'pdfs_other' category)
                         # For this challenge, assuming if not clearly CV, it's an article.
                         categorized_texts["pdfs_ey_articles"].append(extracted_text)
                         print(f"Categorized '{filename}' as 'pdfs_ey_articles' (default/ambiguous PDF).")
                    category_assigned = True
            
            if not category_assigned:
                print(f"Skipped unknown file type or empty content for: {filename}")
    return categorized_texts

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

def split_text_into_chunks(texts, chunk_size=500, chunk_overlap=50):
    """Split a list of texts into smaller chunks using LangChain's TextSplitter."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    all_chunks = []
    for text in texts:
        if text.strip():  # evita texto vazio
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
    
    return all_chunks

def store_chunks_in_chromadb(documents_path="Documents", collection_name="collection"):
    """Extracts text from PDFs, splits it into chunks, creates embeddings, and stores them in ChromaDB."""
    
    pdf_texts = extract_and_categorize_documents(documents_path)  # dict
    all_texts = sum(pdf_texts.values(), [])  # Flatten para lista de strings (cada doc)
    
    all_texts = [text for text in all_texts if text.strip()]  # Remove vazios
    
    chunks = split_text_into_chunks(all_texts)  # <-- CORRETO AGORA
    
    embeddings = generate_embeddings(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    chroma_client = chromadb.Client(Settings(
        persist_directory="./chroma_store",
        anonymized_telemetry=False
    ))
    
    collection = chroma_client.get_or_create_collection(name=collection_name)

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
    print(query)
    
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
        n_results=15  # Number of similar results to retrieve
    )
    print(results)

    # Extract the most relevant documents from the results
    retrieved_documents = results['documents']

    print(f"Retrieved documents: {retrieved_documents}")
    
    # Check if relevant documents were found
    if retrieved_documents:
        # Flatten the list of documents and return them concatenated as context
        flat_documents = [doc for sublist in retrieved_documents for doc in sublist]  # Flatten list
        print("Retrieved documents:", flat_documents)  # Debugging line
        print(flat_documents)
        return "\n".join(flat_documents)  # Join all documents as a single string
    
    # If no relevant documents found, return a fallback message
    return "Desculpe, nenhum contexto relevante encontrado. Tente novamente com uma consulta diferente."

def add_prompt_to_message(query: str, prompt: str, context: str):
    """Combina contexto, prompt base e a query do utilizador numa mensagem final para o LLM."""
    full_prompt = f"""{prompt}

Contexto:
\"\"\"
{context}
\"\"\"

Pergunta: {query}
"""
    return full_prompt

def build_prompt():
    """Prompt base que orienta o comportamento do assistente."""
    return (
        "Atua como assistente especializado da EY. "
        "Usa apenas o contexto fornecido para responder. "
        "Responde de forma clara, concisa e, se possível, com bullets.\n"
        "Se a resposta não estiver no contexto, diz isso diretamente sem inventar."
    )

def generate_response(query):
    """Gera uma resposta do LLM com base em contexto relevante."""
    context = retrieve_context(query)
    
    if "no relevant context found" in context.lower():
        return context

    base_prompt = build_prompt()
    formatted_input = add_prompt_to_message(query, base_prompt, context)

    llm = config_llm()
    response = llm.invoke(formatted_input)

    return response.content if hasattr(response, "content") else str(response)

# Callback que envia mensagem e recebe resposta
def enviar():
    entrada = st.session_state.input_usuario.strip()
    if entrada:
        st.session_state.mensagens.append({"autor": "Você", "texto": entrada})

        # Gera a resposta com sua função
        resposta = generate_response(entrada)
        st.session_state.mensagens.append({"autor": "Bot", "texto": resposta})

        # Limpa o campo
        st.session_state.input_usuario = ""

# Função principal do app
def main():
    st.set_page_config(page_title="EY Virtual Agent", page_icon="🤖")
    st.title("🤖 EY AI Challenge – Virtual Assistant")

    # Carregar os dados apenas uma vez
    if "dados_carregados" not in st.session_state:
      with st.spinner("A preparar os dados..."):
        store_chunks_in_chromadb()
        st.session_state.dados_carregados = True

    # Histórico de mensagens
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    # Input do utilizador
    st.text_input("Você:", key="input_usuario", on_change=enviar)

    # Mostrar histórico
    st.subheader("💬 Histórico da Conversa")
    for msg in st.session_state.mensagens:
        autor = "🧑‍💼 *Você:*" if msg["autor"] == "Você" else "🤖 *Bot:*"
        st.markdown(f"{autor} {msg['texto']}")

if __name__ == "__main__":
    main()