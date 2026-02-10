from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector database
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Loading local language model... (first time takes a few minutes)")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

print("Government Scheme AI Assistant Ready!")

while True:
    query = input("\nAsk a question about government schemes (or type exit): ")

    if query.lower() == "exit":
        break

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

    # fallback if nothing matched
    if not filtered_chunks:
        filtered_chunks = [doc.page_content[:300] for doc in docs[:2]]


    context = "\n\n".join(filtered_chunks)

    prompt = f"""
You are an AI assistant helping with Indian government schemes.

From the context below, extract WHO IS ELIGIBLE for the scheme.

Only include eligibility conditions such as:
- Type of farmers
- Land ownership or cultivation requirements
- Crop or area conditions

If eligibility details are not clearly mentioned, say:
"Eligibility details not found in the provided context."

Context:
{context}

Question:
{query}

Answer in bullet points:
"""


    response = generator(prompt)[0]["generated_text"]

    print("\nAI Answer:\n")
    print(response)
