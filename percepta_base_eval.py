#!/usr/bin/env python3
"""Base Evaluation Class

This module provides a base class for implementing RAG system evaluations.
Each system (Morphik, OpenAI, etc.) can inherit from this class and implement
the `ingest` and `query` methods specific to their system.

The base class handles:
- Loading questions from CSV
- Managing the evaluation workflow
- Saving results to CSV
- Progress tracking and error handling
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import csv
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        import datetime
        ct = datetime.datetime.fromtimestamp(record.created)
        return ct.strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds

root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler()
    formatter = CustomFormatter('%(asctime)s | %(levelname)s | %(filename)s | %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class PerceptaBaseRAGEvaluator(abc.ABC):
    """Base class for RAG system evaluators.

    Subclasses must implement `ingest` and `query` methods.
    """

    def __init__(self, system_name: str, docs_dir: Path, questions_file: Path, output_file: str = None):
        """Initialize the evaluator.

        Args:
            system_name: Name of the RAG system (e.g., "morphik", "openai")
            docs_dir: Directory containing documents to ingest
            questions_file: CSV file with evaluation questions
            output_file: Output CSV file for answers (defaults to {system_name}_answers.csv)
        """
        self.system_name = system_name
        self.docs_dir = Path(docs_dir)
        self.questions_file = Path(questions_file)
        self.output_file = output_file or f"{system_name}_answers.csv"

        # Validate inputs
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")

    @abc.abstractmethod
    def setup_client(self, **kwargs) -> Any:
        """Initialize the RAG system client.

        Returns:
            Client object for the RAG system
        """
        pass

    @abc.abstractmethod
    def ingest(self, client: Any, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents into the RAG system.

        Args:
            client: The RAG system client
            docs_dir: Directory containing documents to ingest
            **kwargs: Additional system-specific parameters

        Returns:
            List of document IDs or identifiers
        """
        pass

    @abc.abstractmethod
    def query(self, client: Any, question: str, **kwargs) -> str:
        """Query the RAG system with a question.

        Args:
            client: The RAG system client
            question: Question to ask
            **kwargs: Additional system-specific parameters

        Returns:
            Answer string from the RAG system
        """
        pass

    def load_questions(self) -> List[Dict[str, str]]:
        """Load questions from CSV file."""
        questions = []

        with open(self.questions_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({"question": row["question"].strip(), "golden_answer": row.get("answer", "").strip()})

        return questions

    async def batch_query(self, client: Any, questions: List[str], max_workers: int = 10, **kwargs) -> List[str]:
        """Run multiple queries concurrently using ThreadPoolExecutor.
        
        Args:
            client: The RAG system client
            questions: List of question strings
            max_workers: Maximum number of concurrent workers
            **kwargs: Additional parameters for query method
            
        Returns:
            List of answer strings
        """
        loop = asyncio.get_event_loop()
        
        def run_query_with_index(q_idx, q):
            question_num = q_idx + 1
            try:
                logger.info(f"  ▸ Starting query {question_num}/{len(questions)}: {q[:60]}...")
                result = self.query(client, q, **kwargs)
                logger.info(f"  ✓ Completed query {question_num}/{len(questions)} ({len(result)} chars)")
                return q_idx, result
            except Exception as e:
                logger.error(f"  ✗ Failed query {question_num}/{len(questions)}: {str(e)}")
                return q_idx, f"[Error: {str(e)}]"
        
        logger.info(f"🚀 Starting {len(questions)} queries with {max_workers} workers")
        
        # Create results array to maintain order
        results = [None] * len(questions)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            tasks = [
                loop.run_in_executor(executor, run_query_with_index, i, q) 
                for i, q in enumerate(questions)
            ]
            
            # Process completions as they arrive
            for task in asyncio.as_completed(tasks):
                q_idx, result = await task
                results[q_idx] = result
                completed += 1
                
                # Show overall progress periodically
                progress_interval = max(1, len(questions) // 10)  # Show progress every 10%
                if completed % progress_interval == 0 or completed == len(questions):
                    logger.info(f"  📊 Progress: {completed}/{len(questions)} queries completed ({completed/len(questions)*100:.1f}%)")
        
        logger.info(f"✅ All {len(questions)} queries completed")
        return results

    def generate_answers(
        self, client: Any, questions: List[Dict[str, str]], skip_ingestion: bool = False, use_parallel: bool = False, max_workers: int = 10, **kwargs
    ) -> List[Dict[str, str]]:
        """Generate answers for all questions.

        Args:
            client: The RAG system client
            questions: List of question dictionaries
            skip_ingestion: Skip document ingestion step
            use_parallel: Whether to use parallel processing for queries
            max_workers: Maximum number of concurrent workers (only used if use_parallel=True)
            **kwargs: Additional parameters for ingest/query methods

        Returns:
            List of result dictionaries with question and answer
        """
        # Ingest documents if not skipped
        if not skip_ingestion:
            logger.info(f"📦 Ingesting documents from {self.docs_dir}")
            doc_ids = self.ingest(client, self.docs_dir, **kwargs)
            logger.info(f"✓ Ingested {len(doc_ids)} documents")
        else:
            logger.info("⏭️ Skipping document ingestion")

        logger.info(f"📝 Generating answers for {len(questions)} questions")
        if use_parallel:
            logger.info(f"⚡ Using parallel processing with {max_workers} workers")
        
        if use_parallel:
            # Use parallel processing
            question_strings = [q_data["question"] for q_data in questions]
            
            try:
                logger.info("⚡ Parallel processing mode activated")
                start_time = time.time()
                
                # Run the batch query in an async context
                answers = asyncio.run(self.batch_query(client, question_strings, max_workers=max_workers, **kwargs))
                
                end_time = time.time()
                total_time = end_time - start_time
                
                logger.info(f"⏱️ Parallel processing completed in {total_time:.2f}s")
                logger.info("📋 Processing results and formatting output")
                
                results = []
                for i, (q_data, answer) in enumerate(zip(questions, answers), 1):
                    question = q_data["question"]
                    
                    # Handle empty or error responses
                    if not answer or answer.strip().lower() in ["", "none", "n/a"]:
                        answer = "[No answer generated]"
                    
                    results.append({"question": question, "answer": answer.strip()})
                    
                    # Show processing progress for result formatting
                    if i % max(1, len(questions) // 20) == 0 or i == len(questions):
                        logger.info(f"  📋 Formatted {i}/{len(questions)} results ({i/len(questions)*100:.1f}%)")
                
                avg_time_per_query = total_time / len(questions)
                logger.info(f"🎯 Average time per query: {avg_time_per_query:.2f}s")
                return results
                
            except Exception as e:
                logger.error(f"✗ Error in parallel processing: {e}")
                logger.info("🔄 Falling back to sequential processing")
                use_parallel = False
        
        if not use_parallel:
            # Use sequential processing (original implementation)
            logger.info("🔄 Sequential processing mode activated")
            start_time = time.time()
            results = []

            for i, q_data in enumerate(questions, 1):
                question = q_data["question"]

                logger.info(f"  ▸ Processing question {i}/{len(questions)}: {question[:60]}...")

                try:
                    query_start = time.time()
                    answer = self.query(client, question, **kwargs)
                    query_time = time.time() - query_start

                    # Handle empty or error responses
                    if not answer or answer.strip().lower() in ["", "none", "n/a"]:
                        answer = "[No answer generated]"

                    results.append({"question": question, "answer": answer.strip()})

                    logger.info(f"    ✓ Generated answer ({len(answer)} chars) in {query_time:.2f}s")
                    
                    # Show overall progress periodically
                    if i % max(1, len(questions) // 10) == 0 or i == len(questions):
                        elapsed = time.time() - start_time
                        avg_time = elapsed / i
                        estimated_total = avg_time * len(questions)
                        remaining = estimated_total - elapsed
                        logger.info(f"  📊 Progress: {i}/{len(questions)} ({i/len(questions)*100:.1f}%) - ETA: {remaining:.1f}s")

                    # Optional delay to avoid overwhelming systems
                    if hasattr(self, "query_delay") and self.query_delay > 0:
                        time.sleep(self.query_delay)

                except Exception as e:
                    logger.error(f"    ✗ Error generating answer: {e}")
                    results.append({"question": question, "answer": f"[Error: {str(e)}]"})

            total_time = time.time() - start_time
            avg_time_per_query = total_time / len(questions)
            logger.info(f"⏱️ Sequential processing completed in {total_time:.2f}s")
            logger.info(f"🎯 Average time per query: {avg_time_per_query:.2f}s")

        return results

    def save_results(self, results: List[Dict[str, str]]) -> None:
        """Save results to CSV file."""
        logger.info(f"💾 Saving results to {self.output_file}")

        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"✅ Saved {len(results)} answers to {self.output_file}")

    def run_evaluation(
        self,
        skip_ingestion: bool = False,
        use_parallel: bool = False,
        max_workers: int = 10,
        client_kwargs: Optional[Dict] = None,
        ingest_kwargs: Optional[Dict] = None,
        query_kwargs: Optional[Dict] = None,
    ) -> str:
        """Run the complete evaluation workflow.

        Args:
            skip_ingestion: Skip document ingestion step
            use_parallel: Whether to use parallel processing for queries
            max_workers: Maximum number of concurrent workers (only used if use_parallel=True)
            client_kwargs: Parameters for setup_client()
            ingest_kwargs: Parameters for ingest()
            query_kwargs: Parameters for query()

        Returns:
            Path to the output CSV file
        """
        client_kwargs = client_kwargs or {}
        ingest_kwargs = ingest_kwargs or {}
        query_kwargs = query_kwargs or {}

        logger.info("=" * 60)
        logger.info(f"{self.system_name.upper()} PERCEPTA EVALUATION")
        logger.info("=" * 60)
        logger.info(f"System: {self.system_name}")
        logger.info(f"Documents: {self.docs_dir}")
        logger.info(f"Questions: {self.questions_file}")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"Skip ingestion: {skip_ingestion}")
        logger.info("=" * 60)

        # Setup client
        logger.info(f"🔧 Setting up {self.system_name} client")
        client = self.setup_client(**client_kwargs)
        logger.info("✅ Client setup complete")

        # Store client for potential cleanup
        self._client = client

        # Load questions
        logger.info(f"📄 Loading questions from {self.questions_file}")
        questions = self.load_questions()
        logger.info(f"✅ Loaded {len(questions)} questions")

        # Generate answers
        results = self.generate_answers(
            client, questions, skip_ingestion=skip_ingestion, use_parallel=use_parallel, max_workers=max_workers, **{**ingest_kwargs, **query_kwargs}
        )

        # Save results
        self.save_results(results)

        logger.info("=" * 60)
        logger.info("🎉 EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📁 Generated answers saved to: {self.output_file}")
        logger.info("📋 Next steps:")
        logger.info(f"  1. Run evaluation: python evaluate.py {self.output_file}")
        logger.info("  2. Check results in eval_results.csv")
        logger.info("=" * 60)

        return self.output_file

    @classmethod
    def create_cli_parser(cls, system_name: str) -> argparse.ArgumentParser:
        """Create a standard CLI parser for evaluation scripts."""
        default_docs_dir = Path(__file__).parent / "docs"
        default_questions_file = Path(__file__).parent / "questions_and_answers.csv"
        default_output_file = f"{system_name}_answers.csv"

        parser = argparse.ArgumentParser(
            description=f"Generate {system_name} answers for financial document evaluation"
        )
        parser.add_argument(
            "--docs-dir",
            type=Path,
            default=default_docs_dir,
            help=(f"Directory containing financial documents " f"(default: {default_docs_dir})"),
        )
        parser.add_argument(
            "--questions",
            type=Path,
            default=default_questions_file,
            help=f"CSV file with questions (default: {default_questions_file})",
        )
        parser.add_argument(
            "--output",
            default=default_output_file,
            help=f"Output CSV file for answers (default: {default_output_file})",
        )
        parser.add_argument(
            "--skip-ingestion", action="store_true", help="Skip document ingestion (use existing documents)"
        )
        parser.add_argument(
            "--parallel", action="store_true", help="Use parallel processing for queries"
        )
        parser.add_argument(
            "--max-workers", type=int, default=10, help="Maximum number of concurrent workers for parallel processing (default: 10)"
        )

        return parser


# Example usage template for implementing a new evaluator:
"""
class MySystemEvaluator(BaseRAGEvaluator):
    def setup_client(self, **kwargs):
        # Initialize your system's client
        return MySystemClient(**kwargs)

    def ingest(self, client, docs_dir, **kwargs):
        # Ingest documents into your system
        doc_files = list(docs_dir.glob("*.pdf"))
        doc_ids = []
        for doc_file in doc_files:
            doc_id = client.ingest_document(doc_file)
            doc_ids.append(doc_id)
        return doc_ids

    def query(self, client, question, **kwargs):
        # Query your system
        response = client.query(question, **kwargs)
        return response.answer

# Usage:
if __name__ == "__main__":
    parser = MySystemEvaluator.create_cli_parser("mysystem")
    args = parser.parse_args()

    evaluator = MySystemEvaluator(
        system_name="mysystem",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output
    )

    evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
"""
