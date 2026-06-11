from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
        _model = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")

def get_rag_response(vectorstore, question):
    load_model()
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=10)
    filtered_docs = [doc for doc, score in docs_with_scores if score < 1.5]
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in docs_with_scores]
    context = " ".join(doc.page_content for doc in filtered_docs)[:3000]
    inputs = _tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = _tokenizer.convert_tokens_to_string(
        _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )
    if not answer.strip() or answer.strip() == "[CLS]":
        answer = "Not available in the document."
    sources = []
    seen_pages = set()
    for doc in filtered_docs:
        page = doc.metadata.get("page", "N/A")
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({"page": page, "text": doc.page_content[:250]})
    return answer, sources
