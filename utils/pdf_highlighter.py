import fitz

def highlight_pdf(input_pdf, output_pdf, answer_text, pages_to_highlight):
    doc = fitz.open(input_pdf)

    for page_num in pages_to_highlight:
        if page_num < 1 or page_num > len(doc):
            continue
        page = doc[page_num - 1]

        # Try to find and highlight exact answer text
        if answer_text and len(answer_text) > 4:
            # Search for full answer first
            instances = page.search_for(answer_text)

            # If not found, search sentence by sentence
            if not instances:
                sentences = [s.strip() for s in answer_text.replace('\n', ' ').split('.') if len(s.strip()) > 10]
                for sentence in sentences[:3]:
                    found = page.search_for(sentence)
                    if found:
                        instances.extend(found)

            # If still not found, search for first 60 characters
            if not instances:
                short = answer_text[:60].strip()
                instances = page.search_for(short)

            # Highlight all found instances
            for inst in instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()

        else:
            # Fallback: highlight small central area only
            rect = fitz.Rect(50, 100, 550, 200)
            highlight = page.add_highlight_annot(rect)
            highlight.update()

    doc.save(output_pdf)
    doc.close()
    return output_pdf