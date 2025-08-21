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
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from percepta_base_eval import PerceptaBaseRAGEvaluator
from query_enhancement import QueryProcessor, QueryClassifier, QueryEnhancer, AnswerValidator
from dotenv import load_dotenv
from morphik import Morphik
from morphik.rules import MetadataExtractionRule, NaturalLanguageRule
from pydantic import BaseModel

load_dotenv(override=True)

# Use the base class logging configuration
logger = logging.getLogger(__name__)


class FinancialMetrics(BaseModel):
    """Schema for extracting comprehensive financial metrics from documents."""
    company_name: str
    fiscal_period: str
    revenue_figures: Dict[str, float] = {}
    growth_rates: Dict[str, float] = {}
    operating_metrics: Dict[str, float] = {}
    balance_sheet_items: Dict[str, float] = {}
    cash_flow_items: Dict[str, float] = {}
    ratios_and_percentages: Dict[str, float] = {}
    dates_and_periods: List[str] = []
    key_financial_numbers: List[Dict[str, Any]] = []
    percentage_metrics: List[Dict[str, Any]] = []
    currency_amounts: List[Dict[str, Any]] = []
    growth_calculations: List[Dict[str, Any]] = []
    comparative_data: List[Dict[str, Any]] = []


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
        self.query_delay = 1.0
        self.server_error_count = 0
        self.setup_query_strategies()
        
        try:
            self.query_processor = QueryProcessor()
            self.query_classifier = QueryClassifier()
            self.query_enhancer = QueryEnhancer()
            self.answer_validator = AnswerValidator()
            self.advanced_processing_available = True
            logger.info("Advanced query processing components initialized")
        except Exception as e:
            logger.warning(f"Could not initialize advanced query processing: {e}")
            logger.info("Falling back to basic query processing")
            self.advanced_processing_available = False

    def setup_query_strategies(self):
        """Setup query strategies for different question types."""
        self.query_strategies = {
            "financial_calculation": QueryStrategy(
                name="financial_calculation",
                k=12,
                padding=3,
                min_score=0.05,
                model="o4-mini",
                retry_count=5,
                fallback_strategy="conservative_search"
            ),
            "business_metrics": QueryStrategy(
                name="business_metrics", 
                k=12,
                padding=3,
                min_score=0.075,
                model="o4-mini",
                retry_count=3,
                fallback_strategy="conservative_search"
            ),
            "market_analysis": QueryStrategy(
                name="market_analysis",
                k=12,
                padding=3,
                min_score=0.1,
                model="o4-mini",
                retry_count=3,
                fallback_strategy="conservative_search"
            ),
            "conservative_search": QueryStrategy(
                name="conservative_search",
                k=10,
                padding=2,
                min_score=0.075,
                model="o4-mini",
                retry_count=4,
                fallback_strategy="simple_naive"
            ),
            "simple_naive": QueryStrategy(
                name="simple_naive",
                k=7,
                padding=1,
                min_score=0.05,
                model="o4-mini",
                retry_count=3
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

        logger.info(f"Connecting to Morphik at: {morphik_uri}")

        try:
            db = Morphik(morphik_uri, timeout=45000)
            logger.info("Connected to Morphik successfully")
            return db
        except Exception as e:
            raise ConnectionError(f"Error connecting to Morphik: {e}")

    def create_ingestion_rules(self) -> List:
        """Create rules for enhanced document processing."""
        rules = [
            MetadataExtractionRule(schema=FinancialMetrics),
            MetadataExtractionRule(schema=BusinessMetrics),
            MetadataExtractionRule(schema=MarketAnalysis),
            NaturalLanguageRule(
                prompt="""Transform this document to improve query retrieval accuracy:
                
                1. Add clear section headers for financial data, business metrics, and market analysis
                2. Standardize all numerical formats (percentages, currency, ratios)
                3. Extract and highlight key calculations with step-by-step breakdowns
                4. Add context labels to all numbers (e.g., "Q1 2024 Revenue: $X million")
                5. Create summary sections for growth rates, margins, and key metrics
                6. Preserve ALL original numerical values and their context
                7. Add cross-references between related financial data points
                
                Focus on making numerical data easily discoverable and contextually rich.
                Keep the document comprehensive but well-organized for retrieval."""
            )
        ]
        return rules

    def ingest(self, client: Morphik, docs_dir: Path, **kwargs) -> List[str]:
        """Enhanced document ingestion with optional rules-based processing."""
        doc_ids = []
        
        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        use_lightweight = kwargs.get('lightweight_processing', False)
        
        if use_lightweight:
            logger.info("Using lightweight processing without rules")
            rules = []
        else:
            rules = self.create_ingestion_rules()
            logger.info(f"Using {len(rules)} ingestion rules for enhanced processing")

        # Get list of documents to process
        doc_files = []
        for ext in ["*.pdf", "*.txt", "*.docx", "*.doc"]:
            doc_files.extend(docs_dir.glob(ext))

        if not doc_files:
            logger.warning(f"No documents found in {docs_dir}")
            return doc_ids

        logger.info(f"Found {len(doc_files)} documents to ingest")

        try:
            for doc_file in doc_files:
                processing_type = "lightweight" if use_lightweight else "enhanced"
                logger.info(f"Ingesting {doc_file.name} with {processing_type} processing")
                
                metadata = {
                    "filename": doc_file.name,
                    "document_type": self._classify_document_type(doc_file.name),
                    "processing_timestamp": time.time(),
                    "enhanced_processing": not use_lightweight
                }
                
                if use_lightweight:
                    doc_info = client.ingest_file(str(doc_file), metadata=metadata)
                else:
                    doc_info = client.ingest_file(str(doc_file), metadata=metadata, rules=rules)
 
                doc_ids.append(doc_info)
                logger.info(f"Ingested {doc_file.name} (ID: {doc_info})")
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            logger.info("Falling back to lightweight processing")
            return self.ingest(client, docs_dir, lightweight_processing=True, **kwargs)

        logger.info(f"Successfully ingested {len(doc_ids)} documents with {processing_type} processing")
 
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
        """Advanced query processing with advanced enhancement and validation."""
        
        if self.server_error_count > 5:
            logger.warning(f"Too many server errors ({self.server_error_count}), using simple strategy")
            naive_strategy = self.query_strategies["simple_naive"]
            result = self._execute_query_with_strategy(client, question, naive_strategy)
            if result:
                return result
            else:
                return f"Unable to find answer even with simple strategy for: {question}"
        
        if not hasattr(self, 'advanced_processing_available') or not self.advanced_processing_available:
            logger.info("Using basic classification")
            question_type = self._classify_question_type(question)
            strategy = self.query_strategies[question_type]
            enhanced_query = self._enhance_query_with_context(question, question_type)
            result = self._execute_query_with_strategy(client, enhanced_query, strategy)
            return result or f"Unable to find answer for: {question}"
        
        logger.info("Processing query through advanced enhancement pipeline")
        try:
            query_processing_result = self.query_processor.process_query(question)
            
            classification = query_processing_result["classification"]
            enhanced_query = query_processing_result["enhanced_query"]
            
            logger.info(f"Query classified as: {classification.primary_type.value}")
            logger.info(f"Complexity score: {classification.complexity_score:.2f}")
            
            strategy_name = self._map_classification_to_strategy(classification)
            strategy = self.query_strategies[strategy_name]
            logger.info(f"Using '{strategy_name}' strategy")
            
            if classification.complexity_score > 0.7:
                result = self._execute_multi_pass_query(client, question, enhanced_query, strategy, classification)
            else:
                result = self._execute_single_pass_query(client, enhanced_query, strategy)
            
            if result and not result.startswith("Unable to find"):
                try:
                    validation_result = self.query_processor.validate_answer(result, query_processing_result)
                    
                    if not validation_result["is_valid"]:
                        logger.warning(f"Answer validation failed: {validation_result['validation_errors']}")
                        
                        if strategy.fallback_strategy:
                            logger.info("Trying fallback strategy")
                            fallback_strategy = self.query_strategies[strategy.fallback_strategy]
                            result = self._execute_single_pass_query(client, enhanced_query, fallback_strategy)
                    
                    result = self._post_process_advanced_answer(result, question, classification)
                except Exception as validation_error:
                    logger.error(f"Error in answer validation: {validation_error}")
                    logger.info("Using basic post-processing")
                    result = self._post_process_answer(result, question)
            
        except Exception as e:
            logger.error(f"Error in advanced query processing: {e}")
            logger.info("Falling back to simple classification")
            question_type = self._classify_question_type(question)
            strategy = self.query_strategies[question_type]
            enhanced_query = self._enhance_query_with_context(question, question_type)
            result = self._execute_query_with_strategy(client, enhanced_query, strategy)
            if not result:
                return f"Unable to find answer for: {question}"
        
        if not result or result.startswith("Unable to find"):
            logger.info("Trying simple_naive as final fallback")
            naive_strategy = self.query_strategies["simple_naive"]
            result = self._execute_query_with_strategy(client, question, naive_strategy)
        
        return result or f"Unable to find answer after trying multiple advanced query strategies for: {question}"

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
                
                if strategy.name != "simple_naive":
                    filters = {"enhanced_processing": True}
                    query_params["filters"] = filters
                
                logger.debug(f"Attempt {attempt + 1}/{strategy.retry_count} with strategy '{strategy.name}'")
                response = client.query(query=query, **query_params)
                
                if response and hasattr(response, 'completion') and response.completion:
                    return response.completion
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if "500 internal server error" in error_msg:
                    self.server_error_count += 1
                    wait_time = (attempt + 1) * 2.0 + min(self.server_error_count * 0.5, 5.0)
                    logger.info(f"Server error detected (count: {self.server_error_count}), waiting {wait_time}s")
                    time.sleep(wait_time)
                elif "timeout" in error_msg:
                    wait_time = (attempt + 1) * 1.5
                    logger.info(f"Timeout detected, waiting {wait_time}s")
                    time.sleep(wait_time)
                    if attempt < strategy.retry_count - 1:
                        strategy.k = max(5, strategy.k - 2)
                        strategy.padding = max(1, strategy.padding - 1)
                elif "rate limit" in error_msg:
                    wait_time = (attempt + 1) * 3.0
                    logger.info(f"Rate limit detected, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    time.sleep(1.0)
        
        logger.error(f"All {strategy.retry_count} attempts failed for strategy '{strategy.name}'")
        return None

    def _map_classification_to_strategy(self, classification) -> str:
        """Map advanced query classification to our strategy names."""
        from query_enhancement import QueryType
        
        mapping = {
            QueryType.FINANCIAL_CALCULATION: "financial_calculation",
            QueryType.BUSINESS_METRICS: "business_metrics", 
            QueryType.MARKET_ANALYSIS: "market_analysis",
            QueryType.COMPARATIVE_ANALYSIS: "financial_calculation",  # Use financial for comparisons
            QueryType.TEMPORAL_ANALYSIS: "financial_calculation",    # Use financial for time-based queries
            QueryType.REGULATORY_COMPLIANCE: "business_metrics",     # Use business for compliance
            QueryType.GENERAL_INQUIRY: "conservative_search"         # Use conservative for general
        }
        
        return mapping.get(classification.primary_type, "conservative_search")
    
    def _execute_single_pass_query(self, client: Morphik, query: str, strategy: QueryStrategy) -> Optional[str]:
        """Execute a single-pass query with the given strategy."""
        return self._execute_query_with_strategy(client, query, strategy)
    
    def _execute_multi_pass_query(self, client: Morphik, original_question: str, enhanced_query: str, 
                                 strategy: QueryStrategy, classification) -> Optional[str]:
        """Execute multi-pass query for complex questions."""
        logger.info("Executing multi-pass query for complex question")
        
        logger.info("Pass 1: Enhanced query with high precision")
        high_precision_strategy = QueryStrategy(
            name=f"{strategy.name}_high_precision",
            k=strategy.k + 3,
            padding=strategy.padding + 2,
            min_score=strategy.min_score + 0.02,
            model=strategy.model,
            retry_count=2
        )
        result = self._execute_query_with_strategy(client, enhanced_query, high_precision_strategy)
        
        if result and not result.startswith("Unable to find"):
            return result
        
        logger.info("Pass 2: Original question with broader search")
        broad_strategy = QueryStrategy(
            name=f"{strategy.name}_broad",
            k=strategy.k + 5,
            padding=strategy.padding + 1,
            min_score=max(0.03, strategy.min_score - 0.02),
            model=strategy.model,
            retry_count=2
        )
        result = self._execute_query_with_strategy(client, original_question, broad_strategy)
        
        if result and not result.startswith("Unable to find"):
            return result
            
        logger.info("Pass 3: Key terms focused query")
        key_terms_query = f"{original_question} Focus on: {', '.join(classification.key_terms[:3])}"
        result = self._execute_query_with_strategy(client, key_terms_query, strategy)
        
        return result
    
    def _post_process_advanced_answer(self, answer: str, question: str, classification) -> str:
        """Post-process answer with advanced validation and enhancement."""
        if not answer or answer.startswith("Unable to find"):
            return answer
        
        try:
            is_valid, error_msg, suggestions = self.answer_validator.validate_answer(answer, classification)
            
            quality_score = self.query_processor._calculate_answer_quality(answer, classification)
            logger.info(f"Answer quality score: {quality_score:.2f}")
            
            # Enhance numerical answers with better formatting
            if hasattr(classification, 'expected_answer_type') and classification.expected_answer_type.value in ["numerical", "percentage", "currency"]:
                answer = self._enhance_numerical_answer(answer, classification)
            
            # Add summary for complex answers
            if hasattr(classification, 'complexity_score') and classification.complexity_score > 0.7 and len(answer.split()) > 100:
                answer = self._add_answer_summary(answer, classification)
                
        except Exception as e:
            logger.error(f"Error in advanced post-processing: {e}")
            logger.info("Using basic post-processing")
            return self._post_process_answer(answer, question)
        
        return answer
    
    def _enhance_numerical_answer(self, answer: str, classification) -> str:
        """Enhance numerical answers with better formatting and context."""
        
        percentages = re.findall(r'(\d+\.?\d*)\s*%', answer)
        currency_amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|M|B)?', answer)
        ratios = re.findall(r'(\d+\.?\d*)[:\s]*(?:to|×|x)\s*(\d+\.?\d*)', answer)
        
        key_numbers = []
        if percentages:
            key_numbers.extend([f"{p}%" for p in percentages])
        if currency_amounts:
            key_numbers.extend([f"${amt[0]}{' ' + amt[1] if amt[1] else ''}" for amt in currency_amounts])
        if ratios:
            key_numbers.extend([f"{r[0]}:{r[1]}" for r in ratios])
        
        if len(key_numbers) > 2:
            answer += f"\n\n📊 Key Figures: {', '.join(key_numbers[:5])}"
        
        return answer
    
    def _add_answer_summary(self, answer: str, classification) -> str:
        """Add a summary section for complex answers."""
        lines = answer.split('\n')
        if len(lines) > 5:
            summary = f"💡 Summary: This answer addresses {classification.primary_type.value} involving {', '.join(classification.key_terms[:3])}."
            return f"{summary}\n\n{answer}"
        return answer

    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process the answer to ensure consistency and accuracy."""
        if not answer or answer.startswith("Unable to find"):
            return answer
            
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
        numbers = re.findall(r'\d+\.?\d*%?', answer)
        if numbers:
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
        logger.info("Enhanced Morphik evaluation completed successfully")
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Enhanced Morphik evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
