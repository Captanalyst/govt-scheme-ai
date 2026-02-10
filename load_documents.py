from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings




import os

DATA_PATH = "data"

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs = loader.load()
        documents.extend(docs)

print(f"Loaded {len(documents)} pages from PDFs.")

print("\nSample content from first page:\n")
print(documents[0].page_content[:500])

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"\nSplit into {len(chunks)} chunks.")
print("\nSample chunk:\n")
print(chunks[0].page_content[:500])

# Initialize embeddings model
print("\nCreating embeddings... This may take a minute.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_index")

print("Embeddings created and vector database saved.")

