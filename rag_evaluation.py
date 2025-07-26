# -*- coding: utf-8 -*-
import asyncio
from typing import List, Dict
from rag_main import BanglaRAGSystem  # Adjust import if your main code is in a different file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your evaluation cases with queries and expected answers
TEST_CASES = [
    {
        "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected": "শুম্ভু নাথ"
    },
    {
        "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected": "মামা"
    },
    {
        "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "expected": "১৫ বছর"
    }
]

# Simple substring matching evaluator (can be replaced with fuzzy or semantic similarity later)
def is_answer_relevant(generated_answer: str, expected_answer: str) -> bool:
    if not generated_answer or not expected_answer:
        return False
    # Lowercase and strip for simple robustness
    return expected_answer.strip().lower() in generated_answer.strip().lower()

async def evaluate_rag_system(pdf_path: str):
    rag_system = BanglaRAGSystem(pdf_path=pdf_path, embedding_type="bangla_sbert")
    
    print("Initializing RAG system...")
    await rag_system.initialize()
    print("Initialization complete.\n")
    
    total_tests = len(TEST_CASES)
    relevant_count = 0
    grounded_count = 0  # Here we treat relevance as proxy; extend with actual grounding if you implement
    print(f"Evaluating {total_tests} test cases...\n")
    
    for idx, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expected = case["expected"]
        print(f"Test Case {idx}:")
        print(f"Query: {query}")
        result = await rag_system.query(query=query, session_id=f"eval_session_{idx}", use_memory=False)
        generated = result.get("answer", "")
        print(f"Generated Answer: {generated}")
        print(f"Expected Answer: {expected}")
        
        relevant = is_answer_relevant(generated, expected)
        if relevant:
            relevant_count += 1
            grounded_count += 1  # For now, treat relevant == grounded for simplicity
            print("Result: Relevant & Grounded ✅\n")
        else:
            print("Result: Not Relevant or Grounded ❌\n")
    
    accuracy = relevant_count / total_tests * 100
    print(f"Summary:\nTotal Tests: {total_tests}\nRelevant: {relevant_count}\nGrounded: {grounded_count}\nAccuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    # Replace with the actual PDF you are using for initialization
    PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
    asyncio.run(evaluate_rag_system(pdf_path=PDF_PATH))
