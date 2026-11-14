"""
evals.py: a small evaluation suite for the ask-my-docs RAG system.

Two automated checks per test case:
  - retrieval: was the chunk containing the answer actually retrieved?
  - answer:    does the generated answer contain the expected fact?

Run it, note the score, change a setting in rag.py (e.g. THRESHOLD),
run it again, and watch the score move - measuring instead of guessing.
"""

import rag
import time

# Each test case: the question, which chunk index should be retrieved,
# and a keyword the answer must contain to count as correct.
# Chunk order in planets.txt: 0=Mercury 1=Venus 2=Earth 3=Mars
#   4=Jupiter 5=Saturn 6=Uranus 7=Neptune
TEST_CASES = [
    {"question": "Which planet is the hottest?",            "expected_chunk": 1, "expected_fact": "Venus"},
    {"question": "How many moons does Jupiter have?",       "expected_chunk": 4, "expected_fact": "95"},
    {"question": "Which planet is known as the Red Planet?","expected_chunk": 3, "expected_fact": "Mars"},
    {"question": "Which planet has a famous ring system?",  "expected_chunk": 5, "expected_fact": "Saturn"},
    {"question": "Which is the smallest planet?",           "expected_chunk": 0, "expected_fact": "Mercury"},
    {"question": "Which planet is farthest from the Sun?",  "expected_chunk": 7, "expected_fact": "Neptune"},
    {"question": "Which planet supports life?",             "expected_chunk": 2, "expected_fact": "Earth"},
    {"question": "Which planet is the coldest?",            "expected_chunk": 6, "expected_fact": "Uranus"},
    {"question": "What is the tallest volcano in the solar system?", "expected_chunk": 3, "expected_fact": "Olympus Mons"},
    {"question": "Which planet rotates on its side?",       "expected_chunk": 6, "expected_fact": "Uranus"},
    {"question": "Which planet has no moons?",                        "expected_chunk": 0, "expected_fact": "Mercury"},
    {"question": "Which planet would float in water?",                "expected_chunk": 5, "expected_fact": "Saturn"},
    {"question": "Which planet has a day longer than its year?",      "expected_chunk": 1, "expected_fact": "Venus"},
    {"question": "Which planet has a storm lasting centuries?",       "expected_chunk": 4, "expected_fact": "Jupiter"},
    {"question": "Which planet has the strongest winds?",             "expected_chunk": 7, "expected_fact": "Neptune"},
    {"question": "Which planet's day is closest in length to Earth's?", "expected_chunk": 3, "expected_fact": "Mars"},
]

RETRIEVAL_ONLY = False

def evaluate_case(case, chunks, chunk_vectors):
    """Run one test case through the RAG system and score both checks."""
    question = case["question"]

    # Run retrieval and generation using the real RAG functions.
    retrieved_chunks, scored = rag.retrieve(question, chunks, chunk_vectors)
    retrieved_indices = [i for i, score in scored]
    if RETRIEVAL_ONLY:
        answer = "(skipped)"
    else:
        answer = rag.generate_answer(question, retrieved_chunks)

    # Check 1 - retrieval: was the expected chunk actually retrieved?
    retrieval_pass = case["expected_chunk"] in retrieved_indices

    # Check 2 - answer: does the answer contain the expected fact (case-insensitive)?
    answer_pass = case["expected_fact"].lower() in answer.lower()

    return {
        "question": question,
        "retrieval_pass": retrieval_pass,
        "answer_pass": answer_pass,
        "retrieved_indices": retrieved_indices,
        "answer": answer,
    }

def run_evals():
    # Load and embed once, then reuse across all test cases (uses rag's cache).
    chunks = rag.load_chunks("planets.txt")
    chunk_vectors = rag.embed_chunks(chunks)
    print(f"Loaded and embedded {len(chunks)} chunks.")
    print(f"Running {len(TEST_CASES)} test cases "
          f"(THRESHOLD={rag.THRESHOLD}, MIN_CHUNKS={rag.MIN_CHUNKS})\n")

    retrieval_passes = 0
    answer_passes = 0

    for case in TEST_CASES:
        result = evaluate_case(case, chunks, chunk_vectors)

        retrieval_passes += result["retrieval_pass"]
        answer_passes += result["answer_pass"]

        r_mark = "PASS" if result["retrieval_pass"] else "FAIL"
        a_mark = "PASS" if result["answer_pass"] else "FAIL"
        print(f"[retrieval {r_mark} | answer {a_mark}] {result['question']}")
        if not result["retrieval_pass"] or not result["answer_pass"]:
            print(f"    retrieved: {result['retrieved_indices']}  answer: {result['answer'][:80]}")

        time.sleep(13)  # stay under the free-tier limit of ~5 generation calls/minute

    total = len(TEST_CASES)
    print(f"\nRetrieval: {retrieval_passes}/{total} "
          f"({100 * retrieval_passes / total:.0f}%)")
    print(f"Answer:    {answer_passes}/{total} "
          f"({100 * answer_passes / total:.0f}%)")


if __name__ == "__main__":
    run_evals()