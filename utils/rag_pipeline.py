from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def get_rag_response(vectorstore, question):
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=10)
    filtered_docs = [doc for doc, score in docs_with_scores if score < 1.5]
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in docs_with_scores]
    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    prompt = (
        "You are an intelligent document assistant.\n"
        "Rules:\n"
        "1. Use ONLY the provided context\n"
        "2. If exact answer is not present, combine relevant parts\n"
        "3. Do NOT hallucinate\n"
        "4. If still unclear, say not available\n\n"
        "Context:\n" + context +
        "\n\nQuestion:\n" + question +
        "\n\nAnswer:"
    )
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512, device=-1)
    llm = HuggingFacePipeline(pipeline=pipe)
    answer = llm.invoke(prompt)
    sources = []
    seen_pages = set()
    for doc in filtered_docs:
        page = doc.metadata.get("page", "N/A")
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({"page": page, "text": doc.page_content[:250]})
    return answer, sources
