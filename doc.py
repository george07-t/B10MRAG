# -*- coding: utf-8 -*-
from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document
import logging
logging.basicConfig(level=logging.INFO)

def main():
    ## create document for extraction with configurations
    pdf_document = Document(
        document_path='HSC26-Bangla1st-Paper.pdf',
        language='ben'
        )
    pdf2text = PDF2Text(document=pdf_document)
    content = pdf2text.extract()
    print(content)
    
    # Save the extracted content to a text file
    with open('extracted_content.txt', 'w', encoding='utf-8') as file:
        # Convert everything to string first
        text_content = [str(item) for item in content]
        file.write('\n'.join(text_content))

if __name__ == "__main__":
    main()