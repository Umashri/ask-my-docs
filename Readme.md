# ask-my-docs

A minimal Retrieval-Augmented Generation (RAG) system, built from scratch to understand how Applied AI engineering actually works — not to ship a product.

I deliberately used **no RAG frameworks** (no LangChain, no LlamaIndex, no vector database). Everything — chunking, embedding, similarity search, retrieval logic — is written directly with `numpy` and the Gemini API, so that none of the core mechanism is hidden behind a library. The goal was understanding, not convenience.

## What it does

Given a set of documents, it answers questions using only the content of those documents:

1. **Load & chunk** — split the source text into passages.
2. **Embed** — convert each chunk into a vector that captures its meaning (via Gemini's embedding model), cached to disk so embedding only happens once.
3. **Retrieve** — embed the question, then rank chunks by cosine similarity to find the most relevant ones.
4. **Generate** — pass the retrieved chunks to the model as context and ask it to answer *only* from that context, saying "I don't know" when the answer isn't present.

## The interesting part: engineering the retrieval

Retrieval quality — not the API calls — is the hard problem in RAG. This project worked through three approaches:

**1. Fixed top-k.** Always retrieve the top *k* chunks by similarity. Simple, but it fails on questions that need to compare across many chunks: when several chunks score within a small margin of each other, the genuinely relevant one can sit at rank 3 or 4, and top-2 silently drops it — leading to a wrong or incomplete answer.

**2. Similarity threshold.** Retrieve every chunk scoring above a cutoff. This adapts to the question — but there is no single threshold that works for every case. On this dataset the scores were bunched in a narrow band (relevant and irrelevant chunks only ~0.05 apart), so a cutoff that caught the right chunks on one question rejected them on another. A threshold of 0.65 returned zero chunks for a valid question whose best match scored 0.63 — throwing away a correct answer.

**3. Hybrid: threshold with a top-k floor** (the final approach). Keep everything above the threshold, but if fewer than `MIN_CHUNKS` clear the bar, fall back to the top *k*. This trims noise when similarity scores are strong, and guarantees the model is never starved of context when they're weak. It combines the precision of thresholding with the reliability of top-k.

The takeaway: there is no clean knob that "solves" retrieval. Each approach is a tradeoff tuned against the actual data and question types, and the only way to know it works is to test real questions and read the real scores.

## Where RAG stops working

RAG retrieves a *relevant subset* of the data. So it struggles with questions that require looking at **all** the data at once — "which has the most X", "how many total", and other aggregate/count questions. The needed facts are spread across every chunk, but retrieval only ever fetches a few. Production systems typically route these to a structured/database query instead of similarity search. Knowing this boundary is as important as knowing what RAG does well.

## Evaluation

`evals.py` measures the system instead of spot-checking it. Each test case has a question, the chunk that should be retrieved, and a fact the answer must contain. Two checks run per case:

- **Retrieval:** was the expected chunk among those retrieved?
- **Answer:** does the generated answer contain the expected fact?

Scoring them separately shows where a failure comes from: the wrong chunk was fetched, or the right chunk was fetched and the model still answered badly.

### What it found

The first ten questions scored 10/10 on both checks, and kept scoring 10/10 on retrieval under every setting I tried, including retrieving only the single top chunk. A test set that scores 100% under every configuration cannot tell a good configuration from a bad one, so I added six harder questions that are indirect or use words shared across several chunks.

On those six, retrieval-only runs gave:

| Retrieval setting | Score |
|---|---|
| Top chunk only (`THRESHOLD=0.9`, `MIN_CHUNKS=1`) | 5/6 (83%) |
| Threshold with a floor (`THRESHOLD=0.65`, `MIN_CHUNKS=2`) | 6/6 (100%) |

The failing case was "Which planet's day is closest in length to Earth's?": the top match was Mercury, and Mars, the correct chunk, ranked second. This is one question out of six, so it supports the hybrid retrieval design without proving it.

### Limits

- The answer check is a keyword match. It cannot tell "Venus is the hottest" from "Venus is not the hottest", so negation-style questions are left out of the test set.
- Retrieval passing does not guarantee a correct answer when a wrong chunk ranks above the right one.
- Sixteen cases is a small sample.

### Running the evals

    python3 evals.py

Set `RETRIEVAL_ONLY = True` in `evals.py` to skip answer generation. That mode needs no chat-model quota and is the quick way to compare retrieval settings. A full run makes one generation call per case, paced to stay inside the free-tier rate limit, with retry and backoff when a call is rate limited.

## Stack

- Python
- Google Gemini API (`gemini-embedding-001` for embeddings, `gemini-3.6-flash` for generation)
- `numpy` for cosine-similarity search
- No frameworks, by design

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install google-generativeai numpy python-dotenv
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

Then:

```bash
python3 rag.py
```

Ask questions interactively; type `quit` to exit.

## Possible next steps

- A real vector database for scaling beyond a small document set.
- Routing aggregate/count questions to a structured query path.
- A larger test set, and an LLM-as-judge answer check for cases a keyword match cannot score.
- Migrate from the deprecated `google.generativeai` package to `google.genai`.
