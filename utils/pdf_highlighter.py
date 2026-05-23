import fitz  # PyMuPDF


def highlight_pdf(input_pdf, output_pdf, pages_to_highlight):
    doc = fitz.open(input_pdf)

    for page_num in pages_to_highlight:
        page = doc[page_num - 1]  # page index

        # Highlight central content area
        rect = fitz.Rect(50, 100, 550, 500)

        highlight = page.add_highlight_annot(rect)
        highlight.update()

    doc.save(output_pdf)
    doc.close()

    return output_pdf