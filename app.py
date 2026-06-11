import streamlit as st
import tempfile
import uuid
from utils.pdf_loader import extract_text_from_pdf
from utils.text_splitter import split_text
from utils.embeddings import create_vector_store
from utils.rag_pipeline import get_rag_response
from utils.pdf_highlighter import highlight_pdf

st.set_page_config(page_title="Intelligent Document Analysis", layout="wide")
st.title("📄 Intelligent Document Analysis System")
st.subheader("Using LLM + RAG Architecture")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded successfully ✅")

    # Extract text with metadata
    documents = extract_text_from_pdf(pdf_path)
    st.info(f"Extracted text from {len(documents)} pages")

    # Split into chunks
    chunks = split_text(documents)
    st.success(f"Text split into {len(chunks)} chunks")

    # Preview chunk
    st.text_area("Sample Chunk", chunks[0].page_content[:1500], height=250)

    # Create vector database
    vectorstore = create_vector_store(chunks)
    st.success("Vector database created successfully ✅")

    st.subheader("Ask Questions from Document")
    question = st.text_input("Ask a question from the document")

    if question:
        with st.spinner("Thinking..."):
            answer, sources = get_rag_response(vectorstore, question)

        # Show Answer
        st.subheader("Answer")
        st.success(answer)

        # Show Source Info
        if sources:
            st.subheader("📄 Source Information")
            for src in sources:
                st.markdown(f"**Page {src['page']}**")
                st.write(src["text"])

        # Highlighting
        if sources:
            pages_to_highlight = []
            best_page = sources[0]["page"]
            if best_page != "N/A":
                pages_to_highlight.append(int(best_page))

            output_pdf = f"highlighted_{uuid.uuid4().hex}.pdf"
            try:
                highlight_pdf(pdf_path, output_pdf, answer, pages_to_highlight)
                st.success("🟨 Highlighted PDF generated successfully")
                with open(output_pdf, "rb") as f:
                    st.download_button(
                        label="📥 Download Highlighted PDF",
                        data=f,
                        file_name="highlighted.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.warning("⚠️ Highlighting failed, but answer is correct.")