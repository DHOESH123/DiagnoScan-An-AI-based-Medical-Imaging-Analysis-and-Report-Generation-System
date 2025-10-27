from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings,OllamaLLM

def create_chatbot(pdf_paths, chunk_size=1000, chunk_overlap=100):
    """
    Creates a chatbot using Llama 2 with FAISS retrieval from PDF documents.
    
    Args:
        pdf_paths (list): List of PDF file paths.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap characters between chunks.
        
    Returns:
        RetrievalQA chatbot instance.
    """
    
    # 1️⃣ Read PDFs
    full_text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
    
    # 2️⃣ Split into manageable chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(full_text)
    
    print(f"[INFO] Total chunks created: {len(chunks)}")
    
    # 3️⃣ Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or your preferred embedding model
    
    # 4️⃣ Create FAISS vectorstore from chunks
    db = FAISS.from_texts(chunks, embedding=embeddings)
    
    # 5️⃣ Setup retriever and QA chain
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama2:latest", temperature=0)
    
    chatbot = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # can also use 'map_reduce' for large docs
    )
    
    return chatbot
