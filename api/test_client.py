"""
api/test_client.py

Simple test client for the FastAPI backend.

Run the server first:
    uvicorn api.main:app --reload --port 8000

Then run this script:
    python api/test_client.py
"""

import requests
import json
from datetime import datetime


API_BASE_URL = "http://localhost:8000"


def print_header(title: str, char: str = "═") -> None:
    """Print a section header."""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def test_health_check():
    """Test the health check endpoint."""
    print_header("Testing Health Check Endpoint")

    response = requests.get(f"{API_BASE_URL}/health")

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nService Status: {data['status']}")
        print(f"Service Name  : {data['service']}")
        print(f"Version       : {data['version']}")
        print(f"Graph Loaded  : {data['graph_loaded']}")
        print(f"\nConfiguration:")
        for key, value in data['config'].items():
            print(f"  {key:<25} : {value}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_query(question: str, thread_id: str = "test-session"):
    """Test the query endpoint."""
    print_header(f"Testing Query Endpoint")

    payload = {
        "question": question,
        "thread_id": thread_id
    }

    print(f"Question  : {question}")
    print(f"Thread ID : {thread_id}\n")
    print("Sending request...")

    start_time = datetime.now()
    response = requests.post(
        f"{API_BASE_URL}/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}s\n")

    if response.status_code == 200:
        data = response.json()

        print_header("Response Data", char="─")

        print(f"Status: {data['status']}\n")

        print("ANSWER")
        print("─" * 80)
        print(data['answer'])
        print()

        print(f"\nSOURCE CHUNKS ({len(data['sources'])} chunks)")
        print("─" * 80)
        for i, source in enumerate(data['sources'], start=1):
            print(f"\n[{i}] {source['chunk_id']}")
            print(f"    Page   : {source['page_num']}")
            print(f"    Label  : {source['label']}")
            print(f"    Text   : {source['text'][:150]}...")

        print(f"\n\nRETRIEVAL METADATA")
        print("─" * 80)
        metadata = data['metadata']
        print(f"Retrieval Attempt : {metadata['retrieval_attempt']}")
        print(f"Dense Count       : {metadata['dense_count']}")
        print(f"Sparse Count      : {metadata['sparse_count']}")
        print(f"Fused Count       : {metadata['fused_count']}")
        print(f"Top RRF Score     : {metadata['top_rrf_score']:.4f}")

        print(f"\n\nDENSE SCORES (Top 5)")
        print("─" * 80)
        for i, score_info in enumerate(metadata['dense_scores'], start=1):
            print(f"{i}. {score_info['chunk_id']} | score={score_info['score']:.4f} | page={score_info['page_num']}")

        print(f"\n\nSPARSE SCORES (Top 5)")
        print("─" * 80)
        for i, score_info in enumerate(metadata['sparse_scores'], start=1):
            print(f"{i}. {score_info['chunk_id']} | score={score_info['score']:.4f} | page={score_info['page_num']}")

        print(f"\n\nRRF SCORES (Fused)")
        print("─" * 80)
        for i, score_info in enumerate(metadata['rrf_scores'], start=1):
            print(f"{i}. {score_info['chunk_id']} | rrf_score={score_info['rrf_score']:.4f} | page={score_info['page_num']}")

        print("\n" + "═" * 80)

        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    print_header("FastAPI RAG Backend - Test Client", char="═")

    # Test health check
    if not test_health_check():
        print("\n❌ Health check failed. Make sure the server is running:")
        print("   uvicorn api.main:app --reload --port 8000")
        return

    print("\n✓ Health check passed!\n")

    # Test queries
    test_questions = [
        "What is agentic AI?",
        "How do multi-agent systems work together?",
        "What are the key orchestration challenges for agentic AI?",
    ]

    for i, question in enumerate(test_questions, start=1):
        print(f"\n{'=' * 80}")
        print(f"  TEST QUERY {i}/{len(test_questions)}")
        print(f"{'=' * 80}")
        test_query(question, thread_id=f"test-{i}")

        if i < len(test_questions):
            input("\n\nPress Enter to continue to next test...")

    print("\n\n✓ All tests completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Exiting...")
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to the API server.")
        print("   Make sure the server is running:")
        print("   uvicorn api.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
