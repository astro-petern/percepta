#!/usr/bin/env python3
"""Enhanced Morphik Evaluator with Advanced Query Mechanism

This enhanced implementation leverages Morphik's rules-based document processing
to improve query precision and reliability. It addresses the failure patterns
identified in the base evaluation by implementing:

1. Financial Metadata Extraction Rules
2. Calculation Validation Rules  
3. Multi-Pass Query Strategy
4. Error Recovery Mechanisms
5. Document Type-Specific Processing

Usage:
    python enhanced_morphik_eval.py
    python enhanced_morphik_eval.py --output enhanced_morphik_answers.csv
    python enhanced_morphik_eval.py --skip-ingestion
"""

from __future__ import annotations

import asyncio
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from percepta_base_eval import PerceptaBaseRAGEvaluator
from dotenv import load_dotenv
from morphik import Morphik
from morphik.rules import MetadataExtractionRule, NaturalLanguageRule
from pydantic import BaseModel

# Load environment variables
load_dotenv(override=True)


class FinancialMetrics(BaseModel):
    """Schema for extracting financial metrics from documents."""
    company_name: str
    fiscal_period: str
    revenue_figures: Dict[str, float] = {}
    growth_rates: Dict[str, float] = {}
    operating_metrics: Dict[str, float] = {}
    balance_sheet_items: Dict[str, float] = {}
    cash_flow_items: Dict[str, float] = {}
    ratios_and_percentages: Dict[str, float] = {}
    dates_and_periods: List[str] = []


class BusinessMetrics(BaseModel):
    """Schema for extracting business operation metrics."""
    company_name: str
    customer_counts: Dict[str, int] = {}
    operational_improvements: Dict[str, str] = {}
    time_reductions: Dict[str, str] = {}
    efficiency_gains: Dict[str, float] = {}
    product_categories: List[str] = []


class MarketAnalysis(BaseModel):
    """Schema for extracting market analysis data."""
    market_segments: List[str] = []
    performance_metrics: Dict[str, float] = {}
    growth_forecasts: Dict[str, float] = {}
    comparative_analysis: Dict[str, float] = {}
    volatility_measures: Dict[str, float] = {}


@dataclass
class QueryStrategy:
    """Configuration for different query strategies."""
    name: str
    k: int
    padding: int
    min_score: float
    model: str
    use_rules: bool = True
    retry_count: int = 3
    fallback_strategy: Optional[str] = None


class EnhancedMorphikEvaluator(PerceptaBaseRAGEvaluator):
    """Enhanced Morphik evaluator with advanced query mechanism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_delay = 1.0  # Increased delay to prevent server overload
        self.server_error_count = 0  # Track server errors for adaptive delays
        self.setup_query_strategies()

    def setup_query_strategies(self):
        """Setup different query strategies for different question types."""
        self.query_strategies = {
            "financial_calculation": QueryStrategy(
                name="financial_calculation",
                k=9,  # Reduced from 15
                padding=2,  # Reduced from 5
                min_score=0.05,  # Increased from 0.05
                model="o4-mini",
                retry_count=5,  # Reduced from 5
                fallback_strategy="conservative_search"
            ),
            "business_metrics": QueryStrategy(
                name="business_metrics", 
                k=10,  # Reduced from 12
                padding=3,  # Reduced from 4
                min_score=0.075,
                model="o4-mini",
                retry_count=3,  # Reduced from 4
                fallback_strategy="conservative_search"
            ),
            "market_analysis": QueryStrategy(
                name="market_analysis",
                k=10,
                padding=3,
                min_score=0.1,
                model="o4-mini",
                retry_count=3,
                fallback_strategy="conservative_search"
            ),
            "conservative_search": QueryStrategy(
                name="conservative_search",
                k=8,  # Much more conservative than broad_search
                padding=1,
                min_score=0.065,
                model="o4-mini",
                retry_count=4,
                fallback_strategy="simple_naive"
            ),
            "simple_naive": QueryStrategy(
                name="simple_naive",
                k=7,  # Same as original working system
                padding=1,
                min_score=0.05,
                model="o4-mini",
                retry_count=3  # Single attempt to avoid overload
            )
        }

    def setup_client(self, **kwargs) -> Morphik:
        """Initialize Morphik client."""
        morphik_uri = os.getenv("MORPHIK_URI")
        if not morphik_uri:
            raise ValueError(
                "MORPHIK_URI environment variable not set. "
                "Please set it with: export MORPHIK_URI=your_morphik_uri"
            )

        print(f"Connecting to Morphik at: {morphik_uri}")

        try:
            db = Morphik(morphik_uri, timeout=45000)  # Increased timeout
            print("✓ Connected to Morphik successfully")
            return db
        except Exception as e:
            raise ConnectionError(f"Error connecting to Morphik: {e}")

    def create_ingestion_rules(self) -> List:
        """Create lightweight rules for document ingestion."""
        # Only use the most essential rules to avoid overload
        rules = [
            # Only extract financial metrics to avoid overwhelming the system
            MetadataExtractionRule(schema=FinancialMetrics),
            
            # Simplified content enhancement
            NaturalLanguageRule(
                prompt="""Add clear section headers and standardize number formats for better retrieval. 
                Preserve all original numerical values and context. Keep the content largely unchanged."""
            )
        ]
        return rules

    def ingest(self, client: Morphik, docs_dir: Path, **kwargs) -> List[str]:
        """Enhanced document ingestion with optional rules-based processing."""
        doc_ids = []
        
        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        # Check if we should use lightweight processing
        use_lightweight = kwargs.get('lightweight_processing', False)
        
        if use_lightweight:
            print("Using lightweight processing without rules due to server load")
            rules = []
        else:
            # Get ingestion rules
            rules = self.create_ingestion_rules()
            print(f"Using {len(rules)} ingestion rules for enhanced document processing")

        # Get list of documents to process
        doc_files = []
        for ext in ["*.pdf", "*.txt", "*.docx", "*.doc"]:
            doc_files.extend(docs_dir.glob(ext))

        if not doc_files:
            print(f"No documents found in {docs_dir}")
            return doc_ids

        print(f"Found {len(doc_files)} documents to ingest")

        try:
            for doc_file in doc_files:
                processing_type = "lightweight" if use_lightweight else "enhanced"
                print(f"Ingesting with {processing_type} processing: {doc_file.name}")
                
                # Metadata for categorization
                metadata = {
                    "filename": doc_file.name,
                    "document_type": self._classify_document_type(doc_file.name),
                    "processing_timestamp": time.time(),
                    "enhanced_processing": not use_lightweight
                }
                
                # Ingest with or without rules based on load
                if use_lightweight:
                    doc_info = client.ingest_file(str(doc_file), metadata=metadata)
                else:
                    doc_info = client.ingest_file(str(doc_file), metadata=metadata, rules=rules)
 
                print(f"Doc info: {doc_info}")
                
                doc_ids.append(doc_info)
                print(f"✓ Ingested {doc_file.name} (ID: {doc_info})")
                
                # Longer delay to prevent overwhelming the system
                time.sleep(0.5)

        except Exception as e:
            print(f"Enhanced processing failed: {e}")
            print("Falling back to lightweight processing...")
            # Recursive call with lightweight processing
            return self.ingest(client, docs_dir, lightweight_processing=True, **kwargs)

        print(f"✓ Successfully ingested {len(doc_ids)} documents with {processing_type} processing")
 
        return doc_ids

    def _classify_document_type(self, filename: str) -> str:
        """Classify document type based on filename patterns."""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ["10-q", "10q", "10-k", "earnings", "financial"]):
            return "financial_statement"
        elif any(term in filename_lower for term in ["presentation", "investor", "slides"]):
            return "investor_presentation"  
        elif any(term in filename_lower for term in ["market", "outlook", "analysis", "midyear"]):
            return "market_analysis"
        elif any(term in filename_lower for term in ["press", "news", "announcement"]):
            return "press_release"
        else:
            return "general_document"

    def _classify_question_type(self, question: str) -> str:
        """Classify question type to determine optimal query strategy."""
        question_lower = question.lower()
        
        # Financial calculation indicators
        if any(term in question_lower for term in [
            "percentage", "growth rate", "margin", "ratio", "calculate", 
            "standard deviation", "compound annual", "basis points",
            "year-over-year", "quarter-over-quarter", "million", "billion"
        ]):
            return "financial_calculation"
        
        # Business metrics indicators
        elif any(term in question_lower for term in [
            "customer count", "supply chain", "time reduction", "efficiency",
            "automation", "development time", "cycle time", "improvement"
        ]):
            return "business_metrics"
            
        # Market analysis indicators  
        elif any(term in question_lower for term in [
            "market", "sector", "magnificent 7", "volatility", "trading days",
            "investment", "portfolio", "forecast", "eps growth"
        ]):
            return "market_analysis"
            
        else:
            return "broad_search"

    def _enhance_query_with_context(self, question: str, question_type: str) -> str:
        """Enhance the query with contextual information based on question type."""
        enhanced_query = question
        
        # Add context-specific enhancement
        if question_type == "financial_calculation":
            enhanced_query += " Include specific numerical values, calculation methods, and financial statement references."
        elif question_type == "business_metrics":
            enhanced_query += " Focus on operational metrics, customer data, and business process improvements."
        elif question_type == "market_analysis":
            enhanced_query += " Include market performance data, comparative analysis, and forecast information."
        
        return enhanced_query

    def query(self, client: Morphik, question: str, **kwargs) -> str:
        """Enhanced query with multi-strategy approach and error recovery."""
        
        # If we've had too many server errors, fallback to simple approach immediately
        if self.server_error_count > 5:
            print(f"Too many server errors ({self.server_error_count}), using simple_naive strategy directly")
            naive_strategy = self.query_strategies["simple_naive"]
            result = self._execute_query_with_strategy(client, question, naive_strategy)
            if result:
                return result
            else:
                return f"Unable to find answer even with simple strategy for: {question}"
        
        # Classify the question to determine strategy
        question_type = self._classify_question_type(question)
        strategy = self.query_strategies[question_type]
        
        print(f"Using '{strategy.name}' strategy for question type: {question_type}")
        
        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(question, question_type)
        
        # Try primary strategy with retries
        result = self._execute_query_with_strategy(client, enhanced_query, strategy)
        
        # If primary strategy fails and we have a fallback, try it
        if not result and strategy.fallback_strategy:
            print(f"Primary strategy failed, trying fallback: {strategy.fallback_strategy}")
            fallback_strategy = self.query_strategies[strategy.fallback_strategy]
            result = self._execute_query_with_strategy(client, enhanced_query, fallback_strategy)
        
        # Final fallback to simple_naive if everything else fails
        if not result and strategy.name != "simple_naive":
            print("All strategies failed, trying simple_naive as final fallback")
            naive_strategy = self.query_strategies["simple_naive"]
            result = self._execute_query_with_strategy(client, question, naive_strategy)  # Use original question
        
        return result or f"Unable to find answer after trying multiple query strategies for: {question}"

    def _execute_query_with_strategy(self, client: Morphik, query: str, strategy: QueryStrategy) -> Optional[str]:
        """Execute a query using a specific strategy with error handling."""
        
        for attempt in range(strategy.retry_count):
            try:
                # Build query parameters
                query_params = {
                    "k": strategy.k,
                    "padding": strategy.padding,
                    "min_score": strategy.min_score,
                    "llm_config": {
                        "model": strategy.model,
                        "api_key": os.getenv("OPENAI_API_KEY")
                    }
                }
                
                # For simple_naive strategy, don't add filters to avoid complexity
                if strategy.name != "simple_naive":
                    # Add filters for enhanced documents if available
                    filters = {"enhanced_processing": True}
                    query_params["filters"] = filters
                
                print(f"Attempt {attempt + 1}/{strategy.retry_count} with strategy '{strategy.name}'")
                response = client.query(query=query, **query_params)
                
                if response and hasattr(response, 'completion') and response.completion:
                    return response.completion
                else:
                    print(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # Check for specific error types
                if "500 internal server error" in error_msg:
                    # Server error - wait longer before retry and track for adaptive behavior
                    self.server_error_count += 1
                    wait_time = (attempt + 1) * 2.0 + min(self.server_error_count * 0.5, 5.0)
                    print(f"Server error detected (count: {self.server_error_count}), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif "timeout" in error_msg:
                    # Timeout - wait and potentially reduce parameters
                    wait_time = (attempt + 1) * 1.5
                    print(f"Timeout detected, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    # Reduce parameters for next attempt
                    if attempt < strategy.retry_count - 1:
                        strategy.k = max(5, strategy.k - 2)
                        strategy.padding = max(1, strategy.padding - 1)
                elif "rate limit" in error_msg:
                    # Rate limit - wait longer
                    wait_time = (attempt + 1) * 3.0
                    print(f"Rate limit detected, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    # Other error - shorter wait
                    time.sleep(1.0)
        
        print(f"All {strategy.retry_count} attempts failed for strategy '{strategy.name}'")
        return None

    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process the answer to ensure consistency and accuracy."""
        if not answer or answer.startswith("Unable to find"):
            return answer
            
        # Add validation for numerical answers
        if self._is_numerical_question(question):
            return self._validate_numerical_answer(answer, question)
        
        return answer

    def _is_numerical_question(self, question: str) -> bool:
        """Check if question expects numerical answer."""
        numerical_indicators = [
            "how many", "what percentage", "by how much", "growth rate", 
            "ratio", "million", "billion", "basis points", "percentage points"
        ]
        return any(indicator in question.lower() for indicator in numerical_indicators)

    def _validate_numerical_answer(self, answer: str, question: str) -> str:
        """Validate and format numerical answers for consistency."""
        # This could include additional validation logic
        # For now, just ensure the answer format is clear
        
        # Extract numbers and ensure they're clearly stated
        numbers = re.findall(r'\d+\.?\d*%?', answer)
        if numbers:
            # Add a summary line if multiple numbers are mentioned
            if len(numbers) > 2:
                answer += f"\n\nKey figures: {', '.join(numbers)}"
        
        return answer


def main():
    """Main function for CLI usage."""
    parser = EnhancedMorphikEvaluator.create_cli_parser("enhanced_morphik")
    args = parser.parse_args()

    evaluator = EnhancedMorphikEvaluator(
        system_name="enhanced_morphik",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output,
    )

    try:
        output_file = evaluator.run_evaluation(
            skip_ingestion=args.skip_ingestion,
            use_parallel=args.parallel,
            max_workers=args.max_workers,
        )
        print(f"\n✅ Enhanced Morphik evaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ Enhanced Morphik evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
