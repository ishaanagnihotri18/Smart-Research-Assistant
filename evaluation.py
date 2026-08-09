import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from generate import create_llm, generate_answer
from rag_utils import (
    create_embeddings,
    create_vector_store,
    extract_text_from_pdf,
    split_text,
    search_documents,
)

load_dotenv()

PDF_PATH = "Rag_based_Assessment.pdf"

QUESTIONS = [
    (
        "What does Hugging Face provide?",
        "Hugging Face provides open-source embedding models and enables cost-effective or offline setups.",
    ),
    (
        "What is RAG?",
        "RAG combines information retrieval with language generation using retrieved document context.",
    ),
    (
        "Why are embeddings used in the system?",
        "Embeddings represent text as numerical vectors so semantically similar document chunks can be retrieved.",
    ),
    (
        "What is ChromaDB used for?",
        "ChromaDB is used for persistent vector storage and retrieval of relevant document information.",
    ),
    (
        "Why is chunking used in the RAG pipeline?",
        "Chunking divides documents into smaller pieces so relevant sections can be retrieved efficiently.",
    ),
]


def load_store():
    with open(PDF_PATH, "rb") as f:
        pages = extract_text_from_pdf(f)

    chunks = split_text(pages)
    embeddings = create_embeddings()

    return create_vector_store(
        chunks,
        embeddings,
        PDF_PATH,
    )


def collect_data():
    store = load_store()
    llm = create_llm()
    data = []

    for question, reference in QUESTIONS:

        docs = search_documents(
            store,
            question,
            top_k=2,
        )

        answer = generate_answer(
            llm,
            question,
            docs,
        )

        data.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": [
                    doc.page_content for doc in docs
                ],
                "reference": reference,
            }
        )

    return data


def create_evaluator():
    client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    return llm_factory(
        "openai/gpt-oss-20b",
        provider="openai",
        client=client,
        max_tokens=2048,
    )


async def score(metric, **kwargs):
    for attempt in range(3):
        try:
            return await metric.ascore(**kwargs)

        except Exception as e:

            if "429" not in str(e):
                raise

            print(
                "Groq rate limit reached. "
                "Waiting 60 seconds..."
            )

            await asyncio.sleep(60)

    return None


async def evaluate(data):

    llm = create_evaluator()

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    metrics = {
        "faithfulness": Faithfulness(llm=llm),

        "answer_relevancy": AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=1,
        ),

        "context_precision": ContextPrecision(
            llm=llm
        ),

        "context_recall": ContextRecall(
            llm=llm
        ),
    }

    results = []

    for i, sample in enumerate(data, 1):

        print(
            f"Evaluating question {i}/{len(data)}..."
        )

        contexts = sample["retrieved_contexts"]
        question = sample["user_input"]
        answer = sample["response"]
        reference = sample["reference"]

        f = await score(
            metrics["faithfulness"],
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        await asyncio.sleep(20)

        r = await score(
            metrics["answer_relevancy"],
            user_input=question,
            response=answer,
        )

        await asyncio.sleep(20)

        p = await score(
            metrics["context_precision"],
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )

        await asyncio.sleep(20)

        c = await score(
            metrics["context_recall"],
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )

        results.append(
            {
                "question": question,

                "faithfulness": (
                    f.value
                    if f is not None
                    else None
                ),

                "answer_relevancy": (
                    r.value
                    if r is not None
                    else None
                ),

                "context_precision": (
                    p.value
                    if p is not None
                    else None
                ),

                "context_recall": (
                    c.value
                    if c is not None
                    else None
                ),
            }
        )

    return results


def print_results(results):

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    print("\n" + "=" * 55)
    print("RAG EVALUATION RESULTS")
    print("=" * 55)

    for i, result in enumerate(results, 1):

        print(
            f"\nQuestion {i}: "
            f"{result['question']}"
        )

        for metric in metrics:

            value = result[metric]

            if value is not None:
                print(
                    f"{metric}: {value:.3f}"
                )
            else:
                print(
                    f"{metric}: N/A"
                )

    print("\n" + "-" * 55)
    print("AVERAGE SCORES")
    print("-" * 55)

    for metric in metrics:

        values = [
            r[metric]
            for r in results
            if r[metric] is not None
        ]

        if values:
            print(
                f"{metric}: "
                f"{sum(values) / len(values):.3f}"
            )
        else:
            print(f"{metric}: N/A")


if __name__ == "__main__":

    print("Collecting evaluation data...")

    data = collect_data()

    print("Running Ragas evaluation...")

    results = asyncio.run(
        evaluate(data)
    )

    print_results(results)