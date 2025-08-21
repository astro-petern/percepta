#!/usr/bin/env python3
"""Morphik Evaluator

Morphik-specific implementation of the RAG evaluation framework.
Inherits from BaseRAGEvaluator and implements Morphik-specific
ingest and query methods.

Usage:
    python morphik_eval.py
    python morphik_eval.py --output morphik_answers_v2.csv
    python morphik_eval.py --skip-ingestion
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import List

from percepta_base_eval import PerceptaBaseRAGEvaluator
from dotenv import load_dotenv
from morphik import Morphik

# Load environment variables
load_dotenv(override=True)


class PerceptaIntvEvaluator(PerceptaBaseRAGEvaluator):
    """Morphik-specific RAG evaluator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional: Set query delay to avoid overwhelming Morphik
        self.query_delay = 0.0  # No delay needed for Morphik

    def setup_client(self, **kwargs) -> Morphik:
        """Initialize Morphik client."""
        morphik_uri = os.getenv("MORPHIK_URI")
        if not morphik_uri:
            raise ValueError(
                "MORPHIK_URI environment variable not set. " "Please set it with: export MORPHIK_URI=your_morphik_uri"
            )

        print(f"Connecting to Morphik at: {morphik_uri}")

        try:
            db = Morphik(morphik_uri, timeout=30000)
            print("✓ Connected to Morphik successfully")
            return db
        except Exception as e:
            raise ConnectionError(f"Error connecting to Morphik: {e}. Make sure Morphik server is running.")

    def ingest(self, client: Morphik, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents into Morphik."""
        # List available documents
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        print(f"Found {len(doc_files)} documents to ingest:")
        for doc_file in doc_files:
            print(f"  - {doc_file.name}")

        # Ingest documents using ingest_directory
        try:
            ingested_docs = client.ingest_directory(
                directory=docs_dir,
                metadata={"source": "financial_eval", "type": "financial_document"},
                use_colpali=True,
            )

            print(f"✓ Successfully ingested {len(ingested_docs)} documents")

            # Wait for processing to complete
            print("Waiting for document processing to complete...")
            for doc in ingested_docs:
                client.wait_for_document_completion(doc.external_id, timeout_seconds=300)

            print("✓ All documents processed successfully")

            return [doc.external_id for doc in ingested_docs]

        except Exception as e:
            raise RuntimeError(f"Error ingesting documents: {e}")

    def query(self, client: Morphik, question: str, **kwargs) -> str:
        """Query Morphik with a question, with retry logic for intermittent errors."""
        # Default query parameters optimized for financial documents
        query_params = {
            "k": 10,  # Retrieve more chunks for complex questions
            "padding": 3,  # Add context padding around chunks
            "min_score": 0.075,  # Lower threshold for financial data
            "llm_config": {"model": "o4-mini", "api_key": os.getenv("OPENAI_API_KEY")},
        }

        # Override with any provided kwargs
        query_params.update(kwargs)

        max_retries = 3
        base_delay = 1.0  # Base delay in seconds
        
        for attempt in range(max_retries + 1):
            try:
                response = client.query(query=question, **query_params)
                return response.completion
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if this is a retryable error (5xx server errors)
                is_retryable = (
                    "500" in error_msg or 
                    "502" in error_msg or 
                    "503" in error_msg or 
                    "504" in error_msg or
                    "internal server error" in error_msg or
                    "bad gateway" in error_msg or
                    "service unavailable" in error_msg or
                    "gateway timeout" in error_msg
                )
                
                if attempt < max_retries and is_retryable:
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    print(f"  ⚠️  Query attempt {attempt + 1} failed with retryable error: {e}")
                    print(f"  🔄 Retrying in {delay:.1f} seconds... (attempt {attempt + 2}/{max_retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    # Either we've exhausted retries or it's a non-retryable error
                    if is_retryable:
                        raise RuntimeError(f"Error querying Morphik after {max_retries + 1} attempts: {e}")
                    else:
                        raise RuntimeError(f"Error querying Morphik: {e}")
        
        # This should never be reached, but just in case
        raise RuntimeError("Unexpected error in retry loop")


def main():
    """Main entry point for Morphik evaluation."""
    # Create CLI parser using base class helper
    parser = PerceptaIntvEvaluator.create_cli_parser("morphik")
    args = parser.parse_args()

    # Create evaluator instance
    evaluator = PerceptaIntvEvaluator(
        system_name="morphik", docs_dir=args.docs_dir, questions_file=args.questions, output_file=args.output
    )

    # Run evaluation
    try:
        output_file = evaluator.run_evaluation(
            skip_ingestion=args.skip_ingestion,
            use_parallel=args.parallel,
            max_workers=args.max_workers
        )
        print("\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
