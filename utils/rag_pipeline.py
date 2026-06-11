from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def get_rag_response(vectorstore, question):
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=10)
    filtered_docs = [doc for doc, score in docs_with_scores if score < 1.5]
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in docs_with_scores]
    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    prompt = (
        "You are an intelligent document assistant. "
        "Answer using ONLY the context below. "
        "If unclear, say not available.\n\n"
        "Context:\n" + context +
        "\n\nQuestion: " + question +
        "\nAnswer:"
    )
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=200,
        device=-1,
        truncation=True,
        pad_token_id=50256
    )
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
