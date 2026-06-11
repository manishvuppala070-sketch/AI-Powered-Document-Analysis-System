import fitz

def highlight_pdf(input_pdf, output_pdf, answer_text, pages_to_highlight):
    doc = fitz.open(input_pdf)
    highlighted = False

    # First try to find answer text across ALL pages
    if answer_text and len(answer_text) > 4:
        # Clean answer text for searching
        search_text = answer_text.replace(' - ', '-').replace('  ', ' ').strip()

        # Try full answer and progressively shorter versions
        search_attempts = [
            search_text,
            search_text[:80],
            search_text[:40],
        ]

        # Also try individual sentences/phrases
        phrases = [p.strip() for p in search_text.replace('•', '.').split('.') if len(p.strip()) > 6]
        search_attempts.extend(phrases[:5])

        for page_num in range(len(doc)):
            page = doc[page_num]
            for attempt in search_attempts:
                instances = page.search_for(attempt)
                if instances:
                    for inst in instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()
                    highlighted = True

    # Fallback: highlight specified pages with small rect
    if not highlighted and pages_to_highlight:
        for page_num in pages_to_highlight:
            if 1 <= page_num <= len(doc):
                page = doc[page_num - 1]
                rect = fitz.Rect(50, 100, 550, 200)
                highlight = page.add_highlight_annot(rect)
                highlight.update()

    doc.save(output_pdf)
    doc.close()
    return output_pdf
