from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import re

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
        _model = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")

def clean_answer(text):
    text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]|\[UNK\]', '', text)
    text = text.replace(' ##', '').strip()
    return text

def extract_section(text, keyword):
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        start = match.start()
        next_section = re.search(r'[A-Z]{4,}', text[start + len(keyword) + 10:])
        if next_section:
            end = start + len(keyword) + 10 + next_section.start()
        else:
            end = start + 600
        return text[start:end].strip()
    return None

def find_best_chunk(docs, question):
    question_words = set(question.lower().split())
    best_doc = docs[0]
    best_score = 0
    for doc in docs:
        text_lower = doc.page_content.lower()
        score = sum(1 for word in question_words if word in text_lower)
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc

def get_rag_response(vectorstore, question):
    load_model()
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=10)
    filtered_docs = [doc for doc, score in docs_with_scores if score < 1.5]
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in docs_with_scores]

    best_chunk = find_best_chunk(filtered_docs, question)
    context = best_chunk.page_content[:3000]

    inputs = _tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=False
    )
    with torch.no_grad():
        outputs = _model(**inputs)

    answer_start = torch.argmax(outputs.start_logits).item()
    answer_end = torch.argmax(outputs.end_logits).item() + 1

    input_ids = inputs["input_ids"][0]
    answer_ids = input_ids[answer_start:answer_end]
    answer = _tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    answer = clean_answer(answer)

    if not answer or len(answer) < 4:
        meaningful_words = [w for w in question.upper().split()
                           if len(w) > 3 and w not in ['WHAT', 'WHICH', 'DOES', 'HAVE', 'THIS', 'THAT', 'WITH', 'FROM', 'PROJECT']]
        section_text = None
        for keyword in meaningful_words:
            section_text = extract_section(best_chunk.page_content, keyword)
            if section_text:
                break
        answer = section_text if section_text else "This information is not available in the uploaded document."

    sources = []
    seen_pages = set()
    for doc in filtered_docs:
        page = doc.metadata.get("page", "N/A")
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({"page": page, "text": doc.page_content[:250]})
    return answer, sources
