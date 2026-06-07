"""
Evaluation harness for the TechNova RAG Support Assistant.
Runs every question in evaluation/questions.json through the RAG pipeline,
scores answers via keyword/substring match (Method A), and saves eval_results.json.

Usage:
    python evaluate.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Re-initialise the RAG pipeline (mirrors solution.ipynb)
# ---------------------------------------------------------------------------
VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL = "text-embedding-3-small"

print("Loading vector store...")
embedding_fn = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_fn,
)
print(f"Vector store loaded — {vectorstore._collection.count()} vectors")

openai_client = OpenAI()

SYSTEM_PROMPT = """You are a knowledgeable support assistant for TechNova, a consumer electronics company.
Your job is to answer questions from the internal support team accurately and completely.

Rules:
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. After each fact you state, cite the source file in parentheses: (source: policies/return_policy.md)
3. If the context does not contain the answer, say exactly: "I don't have that information in my knowledge base."
4. Never invent facts, prices, dates, or policy terms.
5. Be thorough — include ALL specific numbers, prices, timeframes, model numbers, and key terms from the context. Do not omit important details."""


def retrieve(question: str, k: int = 12, category: str = None) -> list:
    search_kwargs = {"k": k}
    if category:
        search_kwargs["filter"] = {"category": category}
    return vectorstore.similarity_search(question, **search_kwargs)


def rag_answer(question: str) -> tuple[str, list[str]]:
    """Return (answer_text, list_of_retrieved_sources)."""
    retrieved_chunks = retrieve(question, k=12)
    sources = [c.metadata["source"] for c in retrieved_chunks]

    context_parts = [
        f"[{c.metadata['source']}]\n{c.page_content}"
        for c in retrieved_chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    system_with_context = (
        SYSTEM_PROMPT
        + "\n\n=== KNOWLEDGE BASE CONTEXT ===\n"
        + context
        + "\n=== END CONTEXT ==="
    )

    response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_with_context},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content, sources


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_answer(answer: str, must_mention: list[str], category: str) -> float:
    """
    Method A — keyword/substring match (case-insensitive, normalised).
    Out-of-scope questions (category == 'out_of_scope') are scored on refusal.
    """
    def normalise(text: str) -> str:
        return (
            text.lower()
            .replace(",", "")   # 5,000 → 5000
            .replace("$", "")   # $189 → 189
            .replace("%", "")   # 30% → 30
            .replace("–", "-")  # en-dash → hyphen (3–5 → 3-5)
            .replace("—", "-")  # em-dash → hyphen
        )

    answer_norm = normalise(answer)

    if category == "out_of_scope":
        refusal_phrases = ["don't have", "don't know", "not have", "cannot", "no information"]
        refused = any(phrase in answer_norm for phrase in refusal_phrases)
        return 1.0 if refused else 0.0

    if not must_mention:
        return 1.0

    hits = 0
    for kw in must_mention:
        kw_norm = normalise(kw)
        if kw_norm in answer_norm:
            hits += 1

    return round(hits / len(must_mention), 4)


def retrieval_hit(retrieved_sources: list[str], expected_sources: list[str]) -> bool:
    """True if at least one expected source appears in the retrieved sources."""
    if not expected_sources:
        return True  # out-of-scope: no expected source, counts as hit
    for exp in expected_sources:
        # Strip leading 'knowledge-base/' prefix if present in retrieved path
        for src in retrieved_sources:
            if exp in src or src.endswith(exp):
                return True
    return False


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    eval_path = Path("evaluation/questions.json")
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    per_question = []

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        category = q["category"]
        must_mention = q.get("must_mention", [])
        expected_sources = q.get("expected_sources", [])

        print(f"[{i}/{len(questions)}] {qid}: {question_text[:60]}...")

        answer, retrieved_sources = rag_answer(question_text)
        score = score_answer(answer, must_mention, category)
        ret_hit = retrieval_hit(retrieved_sources, expected_sources)

        per_question.append({
            "id": qid,
            "category": category,
            "question": question_text,
            "predicted_answer": answer,
            "retrieved_sources": retrieved_sources,
            "expected_sources": expected_sources,
            "score": score,
            "retrieval_hit": ret_hit,
        })

        print(f"  score={score:.2f}  retrieval_hit={ret_hit}")

    # Aggregate
    avg_answer_score = round(sum(r["score"] for r in per_question) / len(per_question), 4)
    retrieval_precision = round(sum(r["retrieval_hit"] for r in per_question) / len(per_question), 4)

    # Find q16 (out-of-scope) refusal result
    oos = next((r for r in per_question if r["category"] == "out_of_scope"), None)
    refused_correctly = bool(oos and oos["score"] >= 1.0)

    output = {
        "summary": {
            "total_questions": len(per_question),
            "retrieval_precision": retrieval_precision,
            "answer_score_avg": avg_answer_score,
            "refused_correctly": refused_correctly,
            "passed_60_pct_threshold": avg_answer_score >= 0.60,
        },
        "per_question": per_question,
    }

    out_path = Path("eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"RESULTS SAVED TO: {out_path}")
    print(f"Answer score avg   : {avg_answer_score:.1%}")
    print(f"Retrieval precision: {retrieval_precision:.1%}")
    print(f"Refused out-of-scope correctly: {refused_correctly}")
    print(f"Passed 60% threshold: {output['summary']['passed_60_pct_threshold']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
