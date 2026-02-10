import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

st.set_page_config(page_title="Govt Scheme AI Assistant")

st.title("🇮🇳 Government Scheme AI Assistant")
st.write("Ask questions about PM Fasal Bima Yojana and other schemes.")

@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    return retriever, generator

retriever, generator = load_system()

query = st.text_input("Enter your question:")

if query:
    q_lower = query.lower()

    if "eligible" in q_lower or "eligibility" in q_lower:
        search_query = query + " eligibility criteria coverage of farmers who can apply"
    elif "document" in q_lower or "required" in q_lower:
        search_query = query + " required documents application form proof"
    elif "benefit" in q_lower or "compensation" in q_lower:
        search_query = query + " benefits amount financial assistance coverage"
    else:
        search_query = query

    docs = retriever.invoke(search_query)

    filtered_chunks = []
    for doc in docs:
        text = doc.page_content.lower()
        if any(word in text for word in ["eligible", "coverage", "farmer", "applicable", "insured"]):
            filtered_chunks.append(doc.page_content[:300])

    if not filtered_chunks:
        filtered_chunks = [doc.page_content[:300] for doc in docs[:2]]

    context = "\n\n".join(filtered_chunks)

    prompt = f"""
You are an AI assistant helping with Indian government schemes.

Use ONLY the information from the context below.

Context:
{context}

Question:
{query}

Answer clearly:
"""

    with st.spinner("Thinking..."):
        response = generator(prompt)[0]["generated_text"]

    st.subheader("AI Answer:")
    st.write(response)
