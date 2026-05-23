from langchain_ollama import OllamaLLM


def get_rag_response(vectorstore, question):

    # Retrieve documents (more chunks = better coverage)
    docs_with_scores = vectorstore.similarity_search_with_score(
        question,
        k=10
    )
    
    filtered_docs = [doc for doc, score in docs_with_scores if score < 1.5]

    # fallback if filtering removes everything
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in docs_with_scores]

    
    context = "\n\n".join(doc.page_content for doc in filtered_docs)


    all_docs = vectorstore.similarity_search("", k=1)

    if all_docs:
        filtered_docs.append(all_docs[0])

    prompt = f"""
You are an intelligent document assistant.

Rules:
1. Use ONLY the provided context
2. If exact answer is not present:
   - Try to COMBINE relevant parts from context
3. Do NOT hallucinate
4. If still unclear, say not available

Context:
{context}

Question:
{question}

Answer:
"""

    llm = OllamaLLM(model="llama3", temperature=0)
    answer = llm.invoke(prompt)

    sources = []
    seen_pages = set()

    for doc in filtered_docs:
        page = doc.metadata.get("page", "N/A")

        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({
                "page": page,
                "text": doc.page_content[:250]
            })

    return answer, sources