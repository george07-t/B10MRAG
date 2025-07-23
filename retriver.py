# -*- coding: utf-8 -*-
import unicodedata
import re
import os
import sys
from datetime import datetime

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document

import logging
import torch
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


class OptimizedBanglaRAGRetriever:
    def __init__(self, embedding_type="bangla_sbert", persist_directory="./chroma_db"):
        """Initialize optimized Bangla RAG Retriever with better embeddings"""
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.embedding_type = embedding_type

        # Initialize the best embedding model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_embeddings()

        # OPTIMIZED chunking configuration based on analysis
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n\n",          # Multiple paragraph breaks (highest priority)
                "\n\n",            # Double newlines (paragraph breaks)
                "।\n",             # Bangla sentence end + newline
                "।",               # Bangla sentence end
                "?\n",             # Question + newline
                "!\n",             # Exclamation + newline
                ". ",              # English sentence + space
                "? ",              # Question + space
                "! ",              # Exclamation + space
                "\n",              # Single newline
                " ",               # Spaces (last resort)
                ""                 # Character level (final fallback)
            ],
            chunk_size=800,        # LARGER chunks for better context
            chunk_overlap=200,     # INCREASED overlap
            length_function=len,
            is_separator_regex=False,
        )

        # Storage for content and documents
        self._raw_text_content = ""
        self._documents = []

    def _init_embeddings(self):
        """Initialize the best embedding model for Bangla"""
        try:
            if self.embedding_type == "bangla_sbert":
                print("🔧 Loading Bangla-specific SBERT model...")
                model_kwargs = {"device": self._device}
                encode_kwargs = {"normalize_embeddings": True}
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                print("✅ Bangla SBERT model loaded successfully")
                
            else:
                print("🔧 Loading Ollama embeddings as fallback...")
                self.embeddings = OllamaEmbeddings(model="all-minilm:latest")
                print("✅ Ollama embeddings loaded")
                
        except Exception as e:
            print(f"⚠️ Error loading Bangla SBERT, falling back to Ollama: {e}")
            self.embeddings = OllamaEmbeddings(model="all-minilm:latest")

    def clean_bangla_text(self, text):
        """Enhanced text cleaning for better chunking"""
        if not text:
            return ""

        try:
            # Unicode normalization
            text = unicodedata.normalize('NFC', text)
            
            # Remove excessive whitespace but preserve structure
            text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 consecutive newlines
            text = re.sub(r'[ \t]+', ' ', text)       # Multiple spaces to single
            
            # Fix Bangla punctuation spacing
            text = re.sub(r'।(\S)', r'। \1', text)    # Space after দাঁড়ি
            text = re.sub(r'([!?])(\S)', r'\1 \2', text)  # Space after !?
            
            # Remove OCR artifacts
            text = re.sub(r'্\s+', '্', text)         # Remove space after halant
            text = re.sub(r'\s*।\s*', '। ', text)     # Normalize দাঁড়ি spacing
            
            return text.strip()
        except Exception as e:
            print(f"⚠️ Cleaning error: {e}")
            return str(text)

    def extract_text_from_pdf(self, pdf_path):
        """Extract and clean text from PDF"""
        print("📄 Extracting text from PDF...")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

        try:
            # Create document for extraction
            doc = Document(document_path=pdf_path, language="ben+eng")
            pdf2text = PDF2Text(document=doc)
            raw_content = pdf2text.extract()
            
            if not raw_content:
                return ""

            # Process content with better structure preservation
            full_text = ""
            
            for item in raw_content:
                if isinstance(item, dict) and 'text' in item:
                    text_content = item['text']
                    if text_content and text_content.strip():
                        cleaned_text = self.clean_bangla_text(text_content)
                        if cleaned_text and len(cleaned_text) > 20:  # Skip very short segments
                            full_text += cleaned_text + "\n\n"

            if full_text.strip():
                # Final cleaning pass
                full_text = self.post_process_extracted_text(full_text)
                print(f"✅ Successfully extracted and cleaned {len(full_text)} characters")
                self._raw_text_content = full_text
                return full_text
            else:
                print("❌ No meaningful text content found")
                return ""

        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""

    def post_process_extracted_text(self, text):
        """Post-process extracted text to improve quality"""
        print("🔧 Post-processing extracted text...")
        
        # Remove repeated whitespace patterns
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Fix common OCR issues
        text = re.sub(r'([অ-হ])\s+([অ-হ])', r'\1\2', text)  # Remove spaces within words
        text = re.sub(r'(\d+)\s*\.\s*([অ-হ])', r'\1. \2', text)  # Fix numbering
        
        # Remove standalone punctuation lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^[।!?\.,\s]+$', line):  # Skip punctuation-only lines
                cleaned_lines.append(line)
        
        return '\n\n'.join(cleaned_lines)

    def save_raw_text(self, filename):
        """Save raw extracted text to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RAW EXTRACTED TEXT (OPTIMIZED)\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Characters: {len(self._raw_text_content)}\n")
                f.write(f"Embedding Model: {self.embedding_type}\n")
                f.write("="*80 + "\n\n")
                f.write(self._raw_text_content)
            print(f"📝 Raw text saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving raw text: {e}")

    def create_documents(self, text):
        """Create high-quality document chunks"""
        if not text.strip():
            raise ValueError("Text is empty, cannot create documents.")

        print("🔪 Creating optimized document chunks...")

        try:
            # Split text into chunks
            raw_chunks = self.text_splitter.split_text(text)
            
            # Apply quality filters
            quality_chunks = self.filter_chunks_by_quality(raw_chunks)
            
            self._documents = quality_chunks
            print(f"✅ Created {len(self._documents)} high-quality document chunks")
            
            # Debug: Show chunk statistics
            self.analyze_chunk_quality()
            
            return self._documents

        except Exception as e:
            print(f"❌ Error creating documents: {e}")
            raise

    def filter_chunks_by_quality(self, raw_chunks):
        """Filter chunks by content quality"""
        print("🔧 Filtering chunks by quality...")
        
        quality_chunks = []
        stats = {"too_short": 0, "low_bangla": 0, "repetitive": 0, "accepted": 0}
        
        seen_chunks = set()
        
        for chunk in raw_chunks:
            chunk = chunk.strip()
            
            # Skip very short chunks
            if len(chunk) < 100:
                stats["too_short"] += 1
                continue
            
            # Check for sufficient Bangla content
            bangla_chars = len(re.findall(r'[অ-হ]', chunk))
            if bangla_chars < 30:  # Need at least 30 Bangla characters
                stats["low_bangla"] += 1
                continue
            
            # Check for repetitive content (first 50 chars as signature)
            chunk_signature = chunk[:50]
            if chunk_signature in seen_chunks:
                stats["repetitive"] += 1
                continue
            
            # Skip chunks that are mostly punctuation or numbers
            content_chars = re.sub(r'[।!?.,\s\d\n]+', '', chunk)
            if len(content_chars) < 50:
                stats["low_bangla"] += 1
                continue
            
            # Accept this chunk
            quality_chunks.append(chunk)
            seen_chunks.add(chunk_signature)
            stats["accepted"] += 1
        
        print(f"📊 Chunk filtering stats: {stats}")
        return quality_chunks

    def analyze_chunk_quality(self):
        """Analyze and display chunk quality metrics"""
        if not self._documents:
            return
            
        print(f"\n📊 CHUNK QUALITY ANALYSIS:")
        print("="*60)
        
        lengths = [len(chunk) for chunk in self._documents]
        bangla_ratios = []
        
        for chunk in self._documents:
            bangla_chars = len(re.findall(r'[অ-হ]', chunk))
            total_chars = len(re.sub(r'\s', '', chunk))
            ratio = bangla_chars / max(total_chars, 1)
            bangla_ratios.append(ratio)
        
        print(f"Total chunks: {len(self._documents)}")
        print(f"Average length: {sum(lengths) / len(lengths):.0f} chars")
        print(f"Min length: {min(lengths)} chars")
        print(f"Max length: {max(lengths)} chars")
        print(f"Average Bangla ratio: {sum(bangla_ratios) / len(bangla_ratios):.2f}")
        
        # Show sample chunks
        print(f"\n🔍 SAMPLE HIGH-QUALITY CHUNKS:")
        for i, chunk in enumerate(self._documents[:3]):
            print(f"\nChunk {i+1} ({len(chunk)} chars):")
            print("-" * 40)
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(preview)

    def update_chroma_db(self):
        """Create optimized vector store"""
        if not self._documents:
            raise ValueError("No documents available for vector store creation")
            
        print("💾 Creating optimized vector store...")
        
        # Remove existing database
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("🗑️ Removed existing vector store")
        
        try:
            # Create vector store with better embeddings
            self.vectorstore = Chroma.from_texts(
                texts=self._documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("✅ Optimized vector store created successfully")
            return self.vectorstore
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise

    def get_retriever(self, k=4):
        """Initialize optimized retriever"""
        if not self.vectorstore:
            raise ValueError("Vector store not available")
            
        try:
            # Use similarity search with score threshold
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": 0.1  # Filter out very poor matches
                }
            )
            print(f"✅ Optimized retriever initialized (k={k}, threshold=0.1)")
            return self.retriever
            
        except Exception as e:
            print(f"⚠️ Falling back to basic similarity search: {e}")
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            return self.retriever

    def load_existing_vectorstore(self):
        """Load existing vector store if available"""
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                print("📂 Loading existing vector store...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                # Initialize retriever for existing store
                self.get_retriever()
                print("✅ Existing vector store loaded")
                return True
        except Exception as e:
            print(f"⚠️ Warning: Could not load existing vector store: {e}")
        
        return False

    def format_docs(self, docs):
        """Format retrieved documents"""
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_context(self, query, k=4):
        """Optimized retrieval with deduplication"""
        if not self.retriever:
            print("❌ Retriever not initialized")
            return [], ""
            
        print(f"\n🔍 Query: {query}")
        print("="*60)

        try:
            # Clean the query
            query_cleaned = self.clean_bangla_text(query)
            
            # Retrieve documents
            retrieved_docs = self.retriever.invoke(query_cleaned)
            
            if retrieved_docs:
                # Deduplicate similar results
                unique_docs = self.deduplicate_results(retrieved_docs)
                
                print(f"✅ Retrieved {len(retrieved_docs)} documents, {len(unique_docs)} unique")
                
                # Format the context
                formatted_context = self.format_docs(unique_docs)
                
                # Display results with quality metrics
                for i, doc in enumerate(unique_docs, 1):
                    bangla_chars = len(re.findall(r'[অ-হ]', doc.page_content))
                    total_chars = len(doc.page_content)
                    bangla_ratio = bangla_chars / total_chars if total_chars > 0 else 0
                    
                    preview = doc.page_content[:200].replace('\n', ' ')
                    print(f"\n📄 Document {i}:")
                    print(f"Length: {len(doc.page_content)} chars")
                    print(f"Bangla ratio: {bangla_ratio:.2f}")
                    print(f"Preview: {preview}...")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"Metadata: {doc.metadata}")
                    print("-"*50)
                
                return unique_docs, formatted_context
            else:
                print("❌ No relevant documents found")
                return [], ""
                
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return [], ""

    def deduplicate_results(self, docs):
        """Remove duplicate or very similar results"""
        if not docs:
            return docs
            
        unique_docs = []
        seen_signatures = set()
        
        for doc in docs:
            # Create a signature from first 100 characters
            signature = doc.page_content[:100].strip()
            
            # Skip if we've seen this signature
            if signature in seen_signatures:
                continue
                
            # Check for substantial overlap with existing docs
            is_duplicate = False
            for existing_doc in unique_docs:
                overlap = self.calculate_text_overlap(doc.page_content, existing_doc.page_content)
                if overlap > 0.7:  # 70% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_signatures.add(signature)
        
        return unique_docs

    def calculate_text_overlap(self, text1, text2):
        """Calculate overlap ratio between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def search_in_chunks(self, search_term):
        """Debug method to search for content in chunks"""
        print(f"\n🔍 SEARCHING FOR: '{search_term}' in chunks")
        print("="*60)

        found_chunks = []
        for i, chunk in enumerate(self._documents):
            if search_term in chunk:
                found_chunks.append((i, chunk))
                print(f"✅ FOUND in chunk {i}:")
                print(f"Content: {chunk[:200]}...")
                print("-"*40)

        if not found_chunks:
            print(f"❌ '{search_term}' NOT FOUND in any chunk")
            
            # Also search in raw text
            if search_term in self._raw_text_content:
                print(f"✅ But '{search_term}' EXISTS in raw text!")
                start_idx = self._raw_text_content.find(search_term)
                context = self._raw_text_content[max(0, start_idx-300):start_idx+500]
                print(f"Raw context: {context}")

        return found_chunks

    def save_query_results(self, query, retrieved_docs, formatted_context, filename):
        """Save query and retrieved documents to file"""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"QUERY: {query}\n")
                f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
                f.write(f"RETRIEVED DOCUMENTS: {len(retrieved_docs)}\n")
                f.write("-"*60 + "\n")

                for i, doc in enumerate(retrieved_docs, 1):
                    f.write(f"\nDOCUMENT {i}:\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Length: {len(doc.page_content)} chars\n")
                    f.write(f"Content: {doc.page_content}\n")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        f.write(f"Metadata: {doc.metadata}\n")
                    f.write("-"*30 + "\n")

                f.write(f"\nFORMATTED CONTEXT:\n")
                f.write("-"*40 + "\n")
                f.write(formatted_context + "\n")
                f.write("="*80 + "\n\n")

            print(f"💾 Query results saved to: {filename}")

        except Exception as e:
            print(f"❌ Error saving query results: {e}")


def main():
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"optimized_bangla_rag_log_{timestamp}.txt"
    logger = OutputLogger(log_file)
    sys.stdout = logger

    print("="*80)
    print("🚀 OPTIMIZED BANGLA RAG RETRIEVAL SYSTEM")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Initialize retriever with Bangla-specific embeddings
    retriever = OptimizedBanglaRAGRetriever(embedding_type="bangla_sbert")
    pdf_path = "HSC26-Bangla1st-Paper.pdf"

    try:
        # Always recreate for testing the optimization
        print("\n🔹 STEP 1: EXTRACTING TEXT")
        print("-"*50)

        # Extract text from PDF
        text = retriever.extract_text_from_pdf(pdf_path)

        if not text or not text.strip():
            raise ValueError("No text extracted from PDF")

        # Save raw extracted text
        raw_text_file = f"optimized_raw_text_{timestamp}.txt"
        retriever.save_raw_text(raw_text_file)

        print("\n🔹 STEP 2: CREATING OPTIMIZED DOCUMENTS")
        print("-"*50)

        # Create document chunks
        retriever.create_documents(text)

        print("\n🔹 STEP 3: UPDATING VECTOR STORE")
        print("-"*50)

        # Update Chroma database
        retriever.update_chroma_db()

        print("\n🔹 STEP 4: INITIALIZING RETRIEVER")
        print("-"*50)

        # Initialize retriever
        retriever.get_retriever(k=5)

        print("\n🔹 STEP 5: TESTING OPTIMIZED QUERIES")
        print("-"*50)
        
        test_queries = [
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "অপরিচিতা গল্পের মূল বিষয় কী?",
        ]

        # File to store all query results
        query_results_file = f"optimized_query_results_{timestamp}.txt"

        # Initialize query results file
        with open(query_results_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("OPTIMIZED BANGLA RAG QUERY RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*20} QUERY {i} {'='*20}")

            # Optimized retrieval with deduplication
            retrieved_docs, formatted_context = retriever.retrieve_context(query, k=5)

            # Save query results
            retriever.save_query_results(query, retrieved_docs, formatted_context, query_results_file)

        # Optional content verification
        print("\n🔹 STEP 6: CONTENT VERIFICATION")
        print("-"*50)
        
        debug_terms = ["সুপুরুষ", "কল্যাণী", "বয়স"]
        for term in debug_terms:
            retriever.search_in_chunks(term)

        print("\n" + "="*80)
        print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY")
        print("="*80)

        # Print file summary
        files_created = [
            f"📝 Process log: {log_file}",
            f"📄 Raw text: {raw_text_file}",
            f"🔍 Query results: {query_results_file}",
            f"🗄️ Vector store: {retriever.persist_directory}"
        ]

        for file_info in files_created:
            print(file_info)

        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n✅ All logs saved to: {log_file}")


if __name__ == "__main__":
    main()