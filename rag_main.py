# -*- coding: utf-8 -*-
import os
import re
import unicodedata
import torch
import numpy as np
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

# LangGraph and LangChain imports
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# RAG components
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

# PDF processing
from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document

# Optional dependencies
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- Enhanced Agent State ----
class AgentState(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setdefault("messages", [])
        self.setdefault("chat_history", [])
        self.setdefault("session_id", "default")
        self.setdefault("last_query", "")
        self.setdefault("context_used", "")

# ---- Production-Grade Memory Manager ----
class ConversationMemoryManager:
    def __init__(self, max_history: int = 20, max_sessions: int = 1000):
        self.max_history = max_history
        self.max_sessions = max_sessions
        self.conversations = {}
        self._lock = asyncio.Lock()
    
    async def add_exchange(self, session_id: str, query: str, response: str, context: str):
        """Thread-safe add exchange to memory"""
        async with self._lock:
            try:
                if session_id not in self.conversations:
                    self.conversations[session_id] = []
                
                exchange = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "response": response,
                    "context_preview": context[:200] + "..." if len(context) > 200 else context
                }
                
                self.conversations[session_id].append(exchange)
                
                # Keep only recent exchanges
                if len(self.conversations[session_id]) > self.max_history:
                    self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
                
                # Manage session count
                if len(self.conversations) > self.max_sessions:
                    oldest_session = min(self.conversations.keys())
                    del self.conversations[oldest_session]
                    logger.info(f"Removed oldest session: {oldest_session}")
                    
            except Exception as e:
                logger.error(f"Error adding exchange to memory: {e}")
    
    async def get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation history"""
        async with self._lock:
            try:
                if session_id not in self.conversations:
                    return ""
                
                history_parts = []
                for exchange in self.conversations[session_id][-5:]:
                    history_parts.append(f"Previous Q: {exchange['query']}")
                    history_parts.append(f"Previous A: {exchange['response'][:100]}...")
                
                return "\n".join(history_parts)
            except Exception as e:
                logger.error(f"Error retrieving conversation context: {e}")
                return ""
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get full session history"""
        return self.conversations.get(session_id, [])
    
    def clear_session(self, session_id: str) -> bool:
        """Clear specific session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False

# ---- Production-Grade Hybrid RAG Retriever ----
class ProductionBanglaRAGRetriever:
    def __init__(self, embedding_type: str = "bangla_sbert", persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type
        self.vectorstore = None
        self.bm25 = None
        self.reranker = None
        self.is_initialized = False
        
        # Device setup
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self._device}")
        
        # Initialize components
        self._init_embeddings()
        self._init_reranker()
        self._init_text_splitter()
        
        # Storage
        self._documents = []
        self._chunk_texts = []
    
    def _init_embeddings(self):
        """Initialize embeddings with fallback"""
        try:
            if self.embedding_type == "bangla_sbert":
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
                    model_kwargs={"device": self._device},
                    encode_kwargs={"normalize_embeddings": True}
                )
                logger.info("✅ Bangla SBERT embeddings loaded")
            else:
                self.embeddings = OllamaEmbeddings(model="all-minilm:latest")
                logger.info("✅ Ollama embeddings loaded")
        except Exception as e:
            logger.warning(f"Embedding initialization failed, using fallback: {e}")
            self.embeddings = OllamaEmbeddings(model="all-minilm:latest")
    
    def _init_reranker(self):
        """Initialize cross-encoder reranker"""
        if CrossEncoder:
            try:
                self.reranker = CrossEncoder(
                    'cross-encoder/ms-marco-MiniLM-L-6-v2', 
                    device=self._device
                )
                logger.info("✅ Cross-encoder reranker loaded")
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}")
                self.reranker = None
        else:
            logger.warning("CrossEncoder not available")
    
    def _init_text_splitter(self):
        """Initialize text splitter with Bangla-aware separators"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "।\n", "।", "?\n", "!\n", ". ", "? ", "! ", "\n", " "],
            chunk_size=700,
            chunk_overlap=150,
            length_function=len,
        )
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Bangla or English"""
        try:
            bangla_chars = len(re.findall(r'[অ-হ]', text))
            english_chars = len(re.findall(r'[A-Za-z]', text))
            total_chars = bangla_chars + english_chars
            
            if total_chars == 0:
                return "unknown"
            
            bangla_ratio = bangla_chars / total_chars
            return "bangla" if bangla_ratio > 0.5 else "english"
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"
    
    def clean_multilingual_text(self, text: str) -> str:
        """Enhanced cleaning for Bangla and English text"""
        if not text:
            return ""
        
        try:
            # Unicode normalization
            text = unicodedata.normalize('NFC', text)
            
            # Language-aware cleaning patterns
            cleaning_patterns = [
                (r'\s+', ' '),                      # Multiple spaces
                (r'\s*।\s*', '। '),                # Bangla sentence ender
                (r'\s*\?\s*', '? '),               # Question marks
                (r'\s*!\s*', '! '),                # Exclamation marks
                (r'([অ-হ])([A-Za-z])', r'\1 \2'),   # Bangla-English separation
                (r'([A-Za-z])([অ-হ])', r'\1 \2'),   # English-Bangla separation
                (r'([০-৯])([A-Za-z])', r'\1 \2'),   # Bangla digit-English separation
                (r'([A-Za-z])([০-৯])', r'\1 \2'),   # English-Bangla digit separation
                (r'(\d)([অ-হ])', r'\1 \2'),        # English digit-Bangla separation
                (r'([অ-হ])(\d)', r'\1 \2'),        # Bangla-English digit separation
            ]
            
            for pattern, replacement in cleaning_patterns:
                text = re.sub(pattern, replacement, text, flags=re.UNICODE)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Async PDF text extraction"""
        logger.info(f"📄 Extracting text from: {pdf_path}")
        
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Run PDF extraction in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._extract_pdf_sync, pdf_path)
            return result
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def _extract_pdf_sync(self, pdf_path: str) -> str:
        """Synchronous PDF extraction"""
        try:
            doc = Document(document_path=pdf_path, language="ben+eng")
            pdf2text = PDF2Text(document=doc)
            raw_content = pdf2text.extract()
            
            full_text = ""
            section_count = 0
            
            if isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, str) and item.strip():
                        cleaned_text = self.clean_multilingual_text(item)
                        if cleaned_text and len(cleaned_text) > 20:
                            full_text += cleaned_text + "\n\n"
                            section_count += 1
                    elif isinstance(item, dict) and 'text' in item:
                        text_content = item['text']
                        if text_content and text_content.strip():
                            cleaned_text = self.clean_multilingual_text(text_content)
                            if cleaned_text and len(cleaned_text) > 20:
                                full_text += cleaned_text + "\n\n"
                                section_count += 1
            elif isinstance(raw_content, str):
                full_text = self.clean_multilingual_text(raw_content)
                section_count = 1
            
            logger.info(f"✅ Extracted text from {section_count} sections")
            with open("extracted_raw_text.txt", "w", encoding="utf-8") as f:
                f.write(full_text)
            return full_text
            
        except Exception as e:
            logger.error(f"Sync PDF extraction failed: {e}")
            return ""
    
    async def create_enhanced_documents(self, text: str) -> List[str]:
        """Create high-quality document chunks"""
        if not text.strip():
            raise ValueError("No text provided for document creation")
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._create_documents_sync, text)
            return result
        except Exception as e:
            logger.error(f"Document creation failed: {e}")
            raise Exception(f"Document creation failed: {str(e)}")
    
    def _create_documents_sync(self, text: str) -> List[str]:
        """Synchronous document creation with quality filtering"""
        raw_chunks = self.text_splitter.split_text(text)
        
        quality_chunks = []
        stats = {"total": len(raw_chunks), "filtered": 0, "accepted": 0}
        
        for chunk in raw_chunks:
            chunk = chunk.strip()
            
            # Quality filters
            if len(chunk) < 100:  # Minimum length
                stats["filtered"] += 1
                continue
            
            # Language content check
            bangla_chars = len(re.findall(r'[অ-হ]', chunk))
            english_chars = len(re.findall(r'[A-Za-z]', chunk))
            total_chars = len(re.sub(r'\s', '', chunk))
            
            if total_chars > 0:
                content_ratio = (bangla_chars + english_chars) / total_chars
                if content_ratio < 0.4:  # At least 40% meaningful content
                    stats["filtered"] += 1
                    continue
            
            quality_chunks.append(chunk)
            stats["accepted"] += 1
        
        logger.info(f"📊 Chunk statistics: {stats}")
        
        self._documents = quality_chunks
        self._chunk_texts = quality_chunks
        return quality_chunks
    
    async def create_vectorstore(self):
        """Create vector store asynchronously"""
        if not self._documents:
            raise ValueError("No documents available for vector store creation")
        
        try:
            # Remove existing store
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
                logger.info("🗑️ Removed existing vector store")
            
            # Create new vector store
            loop = asyncio.get_event_loop()
            self.vectorstore = await loop.run_in_executor(
                None, 
                lambda: Chroma.from_texts(
                    texts=self._documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            )
            
            logger.info(f"✅ Vector store created with {len(self._documents)} documents")
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise Exception(f"Vector store creation failed: {str(e)}")
    
    async def create_bm25(self):
        """Create BM25 index asynchronously"""
        if BM25Okapi and self._chunk_texts:
            try:
                loop = asyncio.get_event_loop()
                tokenized_chunks = await loop.run_in_executor(
                    None, 
                    lambda: [chunk.split() for chunk in self._chunk_texts]
                )
                self.bm25 = await loop.run_in_executor(
                    None, 
                    lambda: BM25Okapi(tokenized_chunks)
                )
                logger.info("✅ BM25 index created")
            except Exception as e:
                logger.warning(f"BM25 creation failed: {e}")
                self.bm25 = None
        else:
            logger.warning("BM25 not available")
            self.bm25 = None
    
    async def hybrid_retrieve(self, query: str, conversation_history: str = "", k: int = 10) -> List[LangchainDocument]:
        """Enhanced hybrid retrieval with error handling"""
        try:
            # Expand query with conversation context
            enhanced_query = f"{conversation_history}\n\nCurrent question: {query}" if conversation_history else query
            
            dense_results = []
            bm25_results = []
            
            # Dense retrieval
            if self.vectorstore:
                try:
                    dense_results = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.vectorstore.similarity_search(enhanced_query, k=k*2)
                    )
                except Exception as e:
                    logger.warning(f"Dense retrieval failed: {e}")
            
            # BM25 retrieval
            if self.bm25:
                try:
                    tokenized_query = query.split()
                    bm25_scores = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.bm25.get_scores(tokenized_query)
                    )
                    
                    top_indices = np.argsort(bm25_scores)[::-1][:k*2]
                    for idx in top_indices:
                        if idx < len(self._chunk_texts) and bm25_scores[idx] > 0:
                            bm25_results.append(LangchainDocument(
                                page_content=self._chunk_texts[idx],
                                metadata={"bm25_score": float(bm25_scores[idx])}
                            ))
                except Exception as e:
                    logger.warning(f"BM25 retrieval failed: {e}")
            
            # Combine and deduplicate
            all_results = dense_results + bm25_results
            unique_results = []
            seen = set()
            
            for doc in all_results:
                signature = doc.page_content[:100]
                if signature not in seen:
                    unique_results.append(doc)
                    seen.add(signature)
            
            return unique_results[:k*2]
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    async def rerank_documents(self, query: str, docs: List[LangchainDocument], top_k: int = 5) -> List[LangchainDocument]:
        """Rerank documents with cross-encoder"""
        if not self.reranker or not docs:
            return docs[:top_k]
        
        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.reranker.predict(pairs)
            )
            
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in scored_docs[:top_k]]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return docs[:top_k]
    
    async def retrieve(self, query: str, conversation_history: str = "", k: int = 5) -> List[LangchainDocument]:
        """Main retrieval method"""
        try:
            if not self.is_initialized:
                logger.error("Retriever not initialized")
                return []
            
            # Hybrid retrieval
            hybrid_results = await self.hybrid_retrieve(query, conversation_history, k=k)
            
            # Reranking
            reranked_results = await self.rerank_documents(query, hybrid_results, top_k=k)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def initialize_from_pdf(self, pdf_path: str):
        """Initialize the retriever with PDF data"""
        try:
            logger.info("🔹 Building knowledge base...")
            
            # Extract text
            text = await self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                raise ValueError("No text extracted from PDF")
            
            # Create documents
            await self.create_enhanced_documents(text)
            
            # Create indices
            await self.create_vectorstore()
            await self.create_bm25()
            
            self.is_initialized = True
            logger.info("✅ Knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise Exception(f"Initialization failed: {str(e)}")

# ---- Response Generation ----
async def generate_response(query: str, context: str, language: str) -> str:
    """Generate response using LLM"""
    try:
        from langchain.chat_models import init_chat_model
        
        CHAT_MODEL = 'llama3.1:latest'
        
        # Create prompt
        prompt_template = """You are a helpful assistant that can answer questions in both English and Bangla. 
        Based on the following context, answer the user's question accurately and in the same language as the question.

        Context: {context}

        Question: {query}

        Answer:"""
        
        llm = init_chat_model(CHAT_MODEL, model_provider='ollama')
        
        # Generate response
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm.invoke(prompt_template.format(context=context, query=query))
        )
        
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        if language == "bangla":
            return "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।"
        else:
            return "Sorry, I encountered an error while generating the response."

# ---- Main RAG Class for Easy Integration ----
class BanglaRAGSystem:
    def __init__(self, pdf_path: str, embedding_type: str = "bangla_sbert"):
        self.pdf_path = pdf_path
        self.retriever = ProductionBanglaRAGRetriever(embedding_type=embedding_type)
        self.memory_manager = ConversationMemoryManager(max_history=20)
        self._is_initialized = False  # Use a private attribute

    @property
    def is_initialized(self):
        return self._is_initialized

    async def initialize(self):
        """Initialize the RAG system"""
        try:
            await self.retriever.initialize_from_pdf(self.pdf_path)
            self._is_initialized = True
            logger.info("✅ RAG System initialized successfully")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"RAG System initialization failed: {e}")
            raise
    
    async def query(
        self, 
        query: str, 
        session_id: str = "default", 
        use_memory: bool = True, 
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Main query function for API integration"""
        start_time = datetime.now()
        
        try:
            if not self.is_initialized:
                raise Exception("RAG system not initialized. Call initialize() first.")
            
            # Detect language
            language = self.retriever.detect_language(query)
            logger.info(f"Processing {language} query: {query[:50]}...")
            
            # Get conversation context if memory is enabled
            conversation_history = ""
            if use_memory:
                conversation_history = await self.memory_manager.get_conversation_context(session_id)
            
            # Retrieve relevant documents
            retrieved_docs = await self.retriever.retrieve(
                query=query,
                conversation_history=conversation_history,
                k=max_results
            )
            
            if not retrieved_docs:
                if language == "bangla":
                    answer = "দুঃখিত, আপনার প্রশ্নের জন্য কোনো প্রাসঙ্গিক তথ্য খুঁজে পাইনি।"
                else:
                    answer = "Sorry, I couldn't find relevant information for your query."
                
                return {
                    "answer": answer,
                    "sources": [],
                    "session_id": session_id,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "language_detected": language,
                    "success": True
                }
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            # --- Save query and relevant docs to file ---
            with open("query_and_relevant_docs.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*40}\nQuery: {query}\n\nRelevant Documents:\n")
                for idx, doc in enumerate(retrieved_docs, 1):
                    f.write(f"\n--- Document {idx} ---\n{doc.page_content}\n")
                f.write("\n")
            # Generate response
            answer = await generate_response(query, context, language)
            
            # Prepare source information
            sources = []
            for i, doc in enumerate(retrieved_docs):
                source_info = {
                    "index": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            # Update memory
            if use_memory:
                await self.memory_manager.add_exchange(session_id, query, answer, context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            return {
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "processing_time": processing_time,
                "language_detected": language,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "An error occurred while processing your query.",
                "sources": [],
                "session_id": session_id,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "language_detected": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self.memory_manager.get_session_history(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        return self.memory_manager.clear_session(session_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "initialized": self.is_initialized,
            "vectorstore_ready": self.retriever.vectorstore is not None,
            "bm25_ready": self.retriever.bm25 is not None,
            "reranker_ready": self.retriever.reranker is not None,
            "total_documents": len(self.retriever._documents),
            "active_sessions": len(self.memory_manager.conversations)
        }

# ---- Main Function for Testing ----
async def main():
    """Main function for testing the RAG system"""
    print("="*60)
    print("🚀 PRODUCTION BANGLA RAG SYSTEM")
    print("="*60)
    
    # Initialize RAG system
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    rag_system = BanglaRAGSystem(pdf_path=pdf_path, embedding_type="bangla_sbert")
    
    try:
        # Initialize
        await rag_system.initialize()
        
        # Test queries
        test_queries = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "What is the main theme of the story?",
            "কল্যাণীর বয়স কত ছিল?",
            "Tell me more about the previous character."
        ]
        
        session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*20} QUERY {i} {'='*20}")
            print(f"🔎 Query: {query}")
            
            # Process query
            result = await rag_system.query(
                query=query,
                session_id=session_id,
                use_memory=True,
                max_results=3
            )
            
            print(f"🤖 Answer: {result['answer']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"🌐 Language: {result['language_detected']}")
            print(f"📄 Sources: {len(result['sources'])}")
            print(f"✅ Success: {result['success']}")
            print("-" * 50)
        
        # Show system status
        print(f"\n📊 System Status: {rag_system.get_system_status()}")
        
        print("\n✅ Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        print(f"❌ Error: {e}")
        # ---- CLI Function ----
async def cli():
    """Interactive CLI for the Bangla RAG system"""
    print("="*60)
    print("📝 BANGLA RAG SYSTEM CLI")
    print("="*60)
    pdf_path = input("Enter PDF path (default: HSC26-Bangla1st-Paper.pdf): ").strip() or "HSC26-Bangla1st-Paper.pdf"
    rag_system = BanglaRAGSystem(pdf_path=pdf_path, embedding_type="bangla_sbert")
    session_id = f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        await rag_system.initialize()
        print("✅ System initialized. Type your questions below (type 'exit' to quit).")
        while True:
            query = input("\n🔎 Your question: ").strip()
            if query.lower() in {"exit", "quit"}:
                print("👋 Exiting CLI. Goodbye!")
                break
            result = await rag_system.query(
                query=query,
                session_id=session_id,
                use_memory=True,
                max_results=3
            )
            print(f"\n🤖 Answer: {result['answer']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"🌐 Language: {result['language_detected']}")
            print(f"📄 Sources: {len(result['sources'])}")
            print(f"✅ Success: {result['success']}")
    except Exception as e:
        logger.error(f"CLI failed: {e}")
        print(f"❌ Error: {e}")
        
if __name__ == "__main__":
    # For interactive CLI
    #asyncio.run(cli())
    # For automated testing
    asyncio.run(main())
