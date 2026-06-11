from transformers import pipeline

def get_rag_response(vectorstore, question):
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=10)
    filtered_docs = [doc for doc, score in docs_with_scores if score < 1.5]
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in docs_with_scores]
    context = " ".join(doc.page_content for doc in filtered_docs)
    # Use a real QA model
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/minilm-uncased-squad2",
        device=-1
    )
    result = qa_pipeline(question=question, context=context[:3000])
    answer = result["answer"]
    sources = []
    seen_pages = set()
    for doc in filtered_docs:
        page = doc.metadata.get("page", "N/A")
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({"page": page, "text": doc.page_content[:250]})
    return answer, sources
