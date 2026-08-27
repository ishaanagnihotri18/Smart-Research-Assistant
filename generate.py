from langchain_groq import ChatGroq

def create_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

def generate_answer(llm, query, relevant_chunks):
    context = "\n\n".join(
        document.page_content
        for document in relevant_chunks
    )

    prompt = f"""
You are a helpful research assistant.

Answer the question using only the provided document context.

If the answer cannot be found in the context, say:
"I could not find the answer in the uploaded document."

Document Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content 