# -*- coding: utf-8 -*-
import os
import re
import unicodedata
import torch
import numpy as np
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from typing import Annotated, Sequence, TypedDict

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from langchain_community.retrievers import BM25Retriever

from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# ---- Hybrid Bangla RAG Retriever ----

class HybridBanglaRAGRetriever:
    def __init__(self, embedding_type="bangla_sbert", persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type
        self.vectorstore = None
        self.bm25 = None
        self.reranker = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_embeddings()
        self._init_reranker()
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "।", ". ", "\n", " "],
            chunk_size=600,
            chunk_overlap=150,
        )
        self._documents = []
        self._chunk_texts = []

    def _init_embeddings(self):
        if self.embedding_type == "bangla_sbert":
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
                    model_kwargs={"device": self._device},
                    encode_kwargs={"normalize_embeddings": True}
                )
            except Exception:
                self.embeddings = OllamaEmbeddings(model="all-minilm:latest")
        else:
            self.embeddings = OllamaEmbeddings(model="all-minilm:latest")

    def _init_reranker(self):
        if CrossEncoder:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self._device)
            except Exception:
                self.reranker = None

    def clean_bangla_text(self, text):
        if not text:
            return ""
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*।\s*', '। ', text)
        return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        doc = Document(document_path=pdf_path, language="ben+eng")
        pdf2text = PDF2Text(document=doc)
        raw_content = pdf2text.extract()
        full_text = ""
        for item in raw_content:
            if isinstance(item, dict) and 'text' in item:
                text_content = item['text']
                if text_content and text_content.strip():
                    cleaned_text = self.clean_bangla_text(text_content)
                    if cleaned_text and len(cleaned_text) > 20:
                        full_text += cleaned_text + "\n\n"
        return full_text

    def create_documents(self, text):
        chunks = [
            chunk for chunk in self.text_splitter.split_text(text)
            if len(chunk.strip()) >= 80
        ]
        self._documents = chunks
        self._chunk_texts = chunks
        return chunks

    def create_vectorstore(self):
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        self.vectorstore = Chroma.from_texts(
            texts=self._documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def create_bm25(self):
        if BM25Okapi:
            tokenized_chunks = [chunk.split() for chunk in self._chunk_texts]
            self.bm25 = BM25Okapi(tokenized_chunks)
        else:
            self.bm25 = None

    def hybrid_retrieve(self, query, k=8):
        # Dense retrieval
        dense_results = []
        if self.vectorstore:
            dense_results = self.vectorstore.similarity_search(query, k=k*2)
        # BM25 retrieval
        bm25_results = []
        if self.bm25:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
            for idx in top_bm25_indices:
                if idx < len(self._chunk_texts):
                    bm25_results.append(LangchainDocument(
                        page_content=self._chunk_texts[idx],
                        metadata={"bm25_score": float(bm25_scores[idx])}
                    ))
        # Combine and deduplicate
        all_results = dense_results + bm25_results
        seen = set()
        unique_results = []
        for doc in all_results:
            sig = doc.page_content[:100]
            if sig not in seen:
                unique_results.append(doc)
                seen.add(sig)
        return unique_results[:k*2]

    def rerank(self, query, docs, top_k=5):
        if self.reranker and docs:
            pairs = [[query, doc.page_content] for doc in docs]
            try:
                scores = self.reranker.predict(pairs)
                scored_docs = list(zip(docs, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in scored_docs[:top_k]]
            except Exception:
                return docs[:top_k]
        return docs[:top_k]

    def retrieve(self, query, k=5):
        hybrid_results = self.hybrid_retrieve(query, k=k)
        reranked = self.rerank(query, hybrid_results, top_k=k)
        return reranked

# ---- Agentic RAG State and Nodes ----

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

def agent(state):
    messages = state["messages"]
    from langchain.chat_models import init_chat_model
    CHAT_MODEL = 'llama3.1:latest'
    model = init_chat_model(CHAT_MODEL, model_provider='ollama')
    response = model.invoke(messages)
    return {"messages": [response]}

def retrieve_node(state):
    messages = state["messages"]
    user_query = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    if user_query is None:
        raise ValueError("No user query found in messages!")
    docs = retriever.retrieve(user_query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"messages": [ToolMessage(content=context, name="retrieve_bangla_pdf", tool_call_id="fixed-id")]}

def generate(state):
    messages = state["messages"]
    question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    last_message = messages[-1]
    docs = last_message.content
    prompt = hub.pull("rlm/rag-prompt")
    from langchain.chat_models import init_chat_model
    CHAT_MODEL = 'gemma3:12b'
    llm = init_chat_model(CHAT_MODEL, model_provider='ollama')
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [AIMessage(content=response)]}

# ---- Main Agentic RAG Pipeline ----

def main():
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    global retriever
    retriever = HybridBanglaRAGRetriever(embedding_type="bangla_sbert")
    text = retriever.extract_text_from_pdf(pdf_path)
    retriever.create_documents(text)
    retriever.create_vectorstore()
    retriever.create_bm25()

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    test_query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    inputs = {
        "messages": [
            HumanMessage(content=test_query),
        ]
    }
    print("\n" + "="*40)
    print("🔎 AGENTIC RAG TEST QUERY")
    print("="*40)
    for output in graph.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

if __name__ == "__main__":
    main()