# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import unicodedata
import re

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document

import logging
logging.basicConfig(level=logging.INFO)


class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class SimplePDFProcessor:
    def __init__(self, model_name="all-minilm:latest", persist_directory="./chroma_db"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "।", " ", ""]
        )

    def clean_bangla_text(self, text):
        """Clean and normalize Bangla + English mixed text"""
        if not text:
            return ""

        try:
            text = unicodedata.normalize('NFC', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s*।\s*', '। ', text)
            return text.strip()
        except Exception as e:
            print(f"⚠️ Cleaning error: {e}")
            return str(text)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text with better processing"""
        print("📄 Extracting text from PDF...")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

        try:
            doc = Document(document_path=pdf_path, language="ben+eng")
            pdf2text = PDF2Text(document=doc)
            raw_content = pdf2text.extract()
            
            if not raw_content:
                return ""

            # Process content without page separators
            full_text = ""
            
            for item in raw_content:
                if isinstance(item, dict) and 'text' in item:
                    text_content = item['text']
                    if text_content and text_content.strip():
                        cleaned_text = self.clean_bangla_text(text_content)
                        if cleaned_text:
                            full_text += cleaned_text + "\n\n"  # Double newline between pages

            print(f"✅ Successfully processed {len(full_text)} characters")
            return full_text

        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""

    def chunk_text(self, text):
        """Improved chunking strategy"""
        if not text.strip():
            raise ValueError("Text is empty, cannot create chunks.")

        print("🔪 Chunking text...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "।", ".", "!", "?", "\n", " ", ""]
        )
        
        doc = LangchainDocument(page_content=text)
        chunks = text_splitter.split_documents([doc])
        
        # Filter chunks between 50-1200 characters
        valid_chunks = [
            chunk for chunk in chunks 
            if 50 <= len(chunk.page_content.strip()) <= 1200
        ]
        
        print(f"✅ Created {len(valid_chunks)} balanced chunks")
        return valid_chunks
    def create_vectorstore(self, chunks):
        """Create vector store from chunks"""
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
            
        print("💾 Creating vector store...")
        
        # Remove existing database
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("🗑️ Removed existing vector store")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("✅ Vector store created successfully")
            return self.vectorstore
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def get_all_documents_from_vectorstore(self):
        """Get all documents from the vector store"""
        if not self.vectorstore:
            print("❌ No vector store available")
            return []

        print("📂 Retrieving all documents from vector store...")
        
        try:
            # Get all documents using the vectorstore's get method
            all_data = self.vectorstore.get()
            
            if 'documents' in all_data and all_data['documents']:
                documents = all_data['documents']
                metadatas = all_data.get('metadatas', [])
                ids = all_data.get('ids', [])
                
                print(f"✅ Retrieved {len(documents)} documents from vector store")
                
                # Combine documents with their metadata
                combined_docs = []
                for i, doc_content in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    doc_id = ids[i] if i < len(ids) else f"doc_{i}"
                    
                    combined_docs.append({
                        'id': doc_id,
                        'content': doc_content,
                        'metadata': metadata
                    })
                
                return combined_docs
            else:
                print("❌ No documents found in vector store")
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []

    def save_all_content_to_file(self, documents, filename):
        """Save all vector store content to a text file"""
        print(f"💾 Saving all content to {filename}...")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("VECTOR STORE CONTENT DUMP\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Documents: {len(documents)}\n")
                f.write("="*80 + "\n\n")
                
                total_chars = 0
                for i, doc in enumerate(documents, 1):
                    doc_content = doc['content']
                    doc_id = doc['id']
                    metadata = doc['metadata']
                    
                    f.write(f"DOCUMENT {i}\n")
                    f.write("-"*40 + "\n")
                    f.write(f"ID: {doc_id}\n")
                    f.write(f"Metadata: {metadata}\n")
                    f.write(f"Length: {len(doc_content)} characters\n")
                    f.write("-"*40 + "\n")
                    f.write(doc_content + "\n")
                    f.write("\n" + "="*80 + "\n\n")
                    
                    total_chars += len(doc_content)
                
                f.write(f"\nSUMMARY:\n")
                f.write(f"Total Documents: {len(documents)}\n")
                f.write(f"Total Characters: {total_chars}\n")
                f.write(f"Average Document Length: {total_chars // len(documents) if documents else 0} characters\n")
                
            print(f"✅ Content saved to {filename}")
            print(f"📊 Summary: {len(documents)} documents, {total_chars} total characters")
            
        except Exception as e:
            print(f"❌ Error saving content: {e}")


def main():
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"processing_log_{timestamp}.txt"
    logger = OutputLogger(log_file)
    sys.stdout = logger

    print("="*80)
    print("SIMPLE PDF TO VECTOR STORE PROCESSOR")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    processor = SimplePDFProcessor()
    pdf_path = "HSC26-Bangla1st-Paper.pdf"

    try:
        # Step 1: Extract text from PDF
        print("\n🔹 STEP 1: EXTRACTING TEXT")
        print("-"*50)
        text = processor.extract_text_from_pdf(pdf_path)
        
        if not text or not text.strip():
            raise ValueError("No text extracted from PDF")

        # Save raw extracted text
        raw_text_file = f"raw_extracted_text_{timestamp}.txt"
        with open(raw_text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"📝 Raw text saved to: {raw_text_file}")

        # Step 2: Chunk the text
        print("\n🔹 STEP 2: CHUNKING TEXT")
        print("-"*50)
        chunks = processor.chunk_text(text)

        # Step 3: Create vector store
        print("\n🔹 STEP 3: CREATING VECTOR STORE")
        print("-"*50)
        processor.create_vectorstore(chunks)

        # Step 4: Retrieve all documents from vector store
        print("\n🔹 STEP 4: RETRIEVING ALL DOCUMENTS")
        print("-"*50)
        all_documents = processor.get_all_documents_from_vectorstore()

        # Step 5: Save all content to file
        print("\n🔹 STEP 5: SAVING ALL CONTENT")
        print("-"*50)
        output_file = f"vectorstore_content_{timestamp}.txt"
        processor.save_all_content_to_file(all_documents, output_file)

        print("\n" + "="*80)
        print("✅ PROCESS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"📄 Raw text file: {raw_text_file}")
        print(f"📊 Vector store content: {output_file}")
        print(f"📝 Process log: {log_file}")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n✅ All output saved to: {log_file}")


if __name__ == "__main__":
    main()