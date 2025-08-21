#!/usr/bin/env python3
"""Advanced Query Enhancement Module

This module provides advanced query processing techniques to improve
RAG system performance on complex financial and business questions.

Features:
1. Query Classification and Routing
2. Multi-Pass Retrieval Strategies  
3. Context-Aware Query Expansion
4. Numerical Answer Validation
5. Error Recovery Mechanisms
6. Domain-Specific Query Optimization
"""

from __future__ import annotations

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types for optimal processing."""
    FINANCIAL_CALCULATION = "financial_calculation"
    BUSINESS_METRICS = "business_metrics"
    MARKET_ANALYSIS = "market_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    GENERAL_INQUIRY = "general_inquiry"


class AnswerType(Enum):
    """Expected answer types for validation."""
    NUMERICAL = "numerical"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    RATIO = "ratio"
    DATE = "date"
    CATEGORICAL = "categorical"
    DESCRIPTIVE = "descriptive"


@dataclass
class QueryClassification:
    """Result of query classification."""
    primary_type: QueryType
    secondary_types: List[QueryType]
    expected_answer_type: AnswerType
    complexity_score: float
    key_terms: List[str]
    numerical_expectations: List[str]


@dataclass
class QueryStrategy:
    """Strategy configuration for different query types."""
    name: str
    retrieval_params: Dict[str, Any]
    processing_steps: List[str]
    validation_rules: List[str]
    fallback_strategies: List[str]


class QueryClassifier:
    """Classifies queries to determine optimal processing strategy."""
    
    def __init__(self):
        self.financial_terms = {
            "revenue", "margin", "profit", "growth", "ebitda", "earnings",
            "cash flow", "balance sheet", "income statement", "10-q", "10-k",
            "quarter", "fiscal", "year-over-year", "q/q", "y/y"
        }
        
        self.calculation_terms = {
            "calculate", "percentage", "ratio", "rate", "basis points", "bps",
            "standard deviation", "compound", "cagr", "growth rate", "margin",
            "increase", "decrease", "change", "difference"
        }
        
        self.business_terms = {
            "customer", "operational", "efficiency", "automation", "supply chain",
            "development time", "cycle time", "improvement", "reduction",
            "acceleration", "ontology", "capability"
        }
        
        self.market_terms = {
            "market", "sector", "magnificent 7", "volatility", "trading",
            "investment", "portfolio", "forecast", "outlook", "analysis",
            "performance", "underperform", "outperform"
        }
        
        self.temporal_terms = {
            "between", "from", "to", "during", "period", "quarter", "year",
            "monthly", "quarterly", "annually", "trailing", "ended"
        }

    def classify_query(self, query: str) -> QueryClassification:
        """Classify a query to determine optimal processing approach."""
        query_lower = query.lower()
        tokens = set(re.findall(r'\b\w+\b', query_lower))
        
        financial_score = len(tokens.intersection(self.financial_terms))
        calculation_score = len(tokens.intersection(self.calculation_terms))
        business_score = len(tokens.intersection(self.business_terms))
        market_score = len(tokens.intersection(self.market_terms))
        temporal_score = len(tokens.intersection(self.temporal_terms))
        
        scores = {
            QueryType.FINANCIAL_CALCULATION: financial_score + calculation_score * 1.5,
            QueryType.BUSINESS_METRICS: business_score * 1.2,
            QueryType.MARKET_ANALYSIS: market_score * 1.2,
            QueryType.TEMPORAL_ANALYSIS: temporal_score,
        }
        
        if re.search(r'\b(how many|what percentage|by how much)\b', query_lower):
            scores[QueryType.FINANCIAL_CALCULATION] += 2
            
        if re.search(r'\b(compared to|versus|exceed|lag|outpace)\b', query_lower):
            scores[QueryType.COMPARATIVE_ANALYSIS] = max(scores.values()) * 0.8
            
        if re.search(r'\b(regulation|compliance|act|rule|requirement)\b', query_lower):
            scores[QueryType.REGULATORY_COMPLIANCE] = 3
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_type = sorted_scores[0][0] if sorted_scores[0][1] > 0 else QueryType.GENERAL_INQUIRY
        secondary_types = [item[0] for item in sorted_scores[1:3] if item[1] > 0]
        
        answer_type = self._determine_answer_type(query_lower)
        complexity_score = self._calculate_complexity(query, tokens)
        key_terms = self._extract_key_terms(query, tokens)
        numerical_expectations = self._extract_numerical_expectations(query)
        
        return QueryClassification(
            primary_type=primary_type,
            secondary_types=secondary_types,
            expected_answer_type=answer_type,
            complexity_score=complexity_score,
            key_terms=key_terms,
            numerical_expectations=numerical_expectations
        )

    def _determine_answer_type(self, query_lower: str) -> AnswerType:
        """Determine the expected type of answer."""
        if re.search(r'\b(percentage|%|percent)\b', query_lower):
            return AnswerType.PERCENTAGE
        elif re.search(r'\b(million|billion|dollar|\$)\b', query_lower):
            return AnswerType.CURRENCY
        elif re.search(r'\b(ratio|multiple|times|factor)\b', query_lower):
            return AnswerType.RATIO
        elif re.search(r'\b(date|when|quarter|year|period)\b', query_lower):
            return AnswerType.DATE
        elif re.search(r'\b(how many|count|number)\b', query_lower):
            return AnswerType.NUMERICAL
        else:
            return AnswerType.DESCRIPTIVE

    def _calculate_complexity(self, query: str, tokens: set) -> float:
        """Calculate query complexity score."""
        base_complexity = len(tokens) / 20.0
        
        if query.count('?') > 1 or query.count(',') > 3:
            base_complexity += 0.3
            
        if re.search(r'\b(calculate|derive|compute|standard deviation)\b', query.lower()):
            base_complexity += 0.4
            
        if len(re.findall(r'\b\d{4}\b', query)) > 2:
            base_complexity += 0.2
            
        company_indicators = len(re.findall(r'\b[A-Z]{3,5}\b', query))
        if company_indicators > 1:
            base_complexity += 0.3
            
        return min(1.0, base_complexity)

    def _extract_key_terms(self, query: str, tokens: set) -> List[str]:
        """Extract key terms from the query."""
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        tickers = re.findall(r'\b[A-Z]{3,5}\b', query)
        important_terms = tokens.intersection(
            self.financial_terms | self.calculation_terms | 
            self.business_terms | self.market_terms
        )
        
        return list(set(entities + tickers + list(important_terms)))

    def _extract_numerical_expectations(self, query: str) -> List[str]:
        """Extract numerical patterns that hint at expected answer format."""
        patterns = []
        
        if re.search(r'\b(percentage|%|percent)\b', query.lower()):
            patterns.append("percentage")
            
        if re.search(r'\b(million|billion|dollar|\$)\b', query.lower()):
            patterns.append("currency")
            
        if re.search(r'\b(basis points|bps)\b', query.lower()):
            patterns.append("basis_points")
            
        if re.search(r'\b(growth rate|cagr)\b', query.lower()):
            patterns.append("growth_rate")
            
        return patterns


class QueryEnhancer:
    """Enhances queries with context and domain-specific information."""
    
    def __init__(self):
        self.financial_context = {
            "revenue": "total revenue, net revenue, revenue recognition",
            "margin": "gross margin, operating margin, net margin, profit margin",
            "growth": "year-over-year growth, quarter-over-quarter growth, compound annual growth rate",
            "cash flow": "operating cash flow, free cash flow, cash flow from operations"
        }
        
        self.calculation_context = {
            "standard deviation": "population standard deviation, sample standard deviation, volatility measure",
            "percentage points": "difference in percentages, not percentage change",
            "basis points": "hundredths of a percentage point, 100 basis points = 1%"
        }

    def enhance_query(self, query: str, classification: QueryClassification) -> str:
        """Enhance query with relevant context and clarifications."""
        enhanced_query = query
        
        if classification.primary_type == QueryType.FINANCIAL_CALCULATION:
            enhanced_query += self._add_financial_context(query, classification)
        elif classification.primary_type == QueryType.BUSINESS_METRICS:
            enhanced_query += self._add_business_context(query, classification)
        elif classification.primary_type == QueryType.MARKET_ANALYSIS:
            enhanced_query += self._add_market_context(query, classification)
        
        if classification.expected_answer_type in [AnswerType.NUMERICAL, AnswerType.PERCENTAGE, AnswerType.CURRENCY]:
            enhanced_query += " Provide precise numerical values with appropriate units and context."
        
        if classification.complexity_score > 0.7:
            enhanced_query += " Break down complex calculations step by step with intermediate results."
        
        return enhanced_query

    def _add_financial_context(self, query: str, classification: QueryClassification) -> str:
        """Add financial domain context."""
        context = " Consider all relevant financial statement data including:"
        
        for term in classification.key_terms:
            if term.lower() in self.financial_context:
                context += f" {self.financial_context[term.lower()]},"
        
        if "percentage" in classification.numerical_expectations:
            context += " Express results as percentages with appropriate decimal places."
            
        return context.rstrip(",") + "."

    def _add_business_context(self, query: str, classification: QueryClassification) -> str:
        """Add business operations context."""
        return " Focus on operational metrics, efficiency improvements, and business process changes."

    def _add_market_context(self, query: str, classification: QueryClassification) -> str:
        """Add market analysis context."""
        return " Include market performance data, sector comparisons, and relevant benchmarks."


class AnswerValidator:
    """Validates answers for consistency and accuracy."""
    
    def __init__(self):
        self.numerical_patterns = {
            "percentage": re.compile(r'(\d+\.?\d*)\s*%'),
            "currency": re.compile(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|M|B)?'),
            "basis_points": re.compile(r'(\d+(?:\.\d+)?)\s*(?:basis points|bps)'),
            "ratio": re.compile(r'(\d+(?:\.\d+)?)[:\s]*(?:to|×|x)\s*(\d+(?:\.\d+)?)')
        }

    def validate_answer(self, answer: str, classification: QueryClassification) -> Tuple[bool, str, Optional[str]]:
        """Validate answer against expected format and content."""
        if not answer or answer.startswith("Unable to find"):
            return False, "No answer provided", None
        
        validation_errors = []
        suggestions = []
        
        # Validate based on expected answer type
        if classification.expected_answer_type == AnswerType.PERCENTAGE:
            if not self.numerical_patterns["percentage"].search(answer):
                validation_errors.append("Expected percentage value not found")
                suggestions.append("Ensure answer includes percentage with % symbol")
        
        elif classification.expected_answer_type == AnswerType.CURRENCY:
            if not self.numerical_patterns["currency"].search(answer):
                validation_errors.append("Expected currency value not found")
                suggestions.append("Ensure answer includes currency amount with $ symbol and units")
        
        elif classification.expected_answer_type == AnswerType.NUMERICAL:
            if not re.search(r'\d+', answer):
                validation_errors.append("Expected numerical value not found")
                suggestions.append("Ensure answer includes specific numbers")
        
        numerical_consistency = self._check_numerical_consistency(answer, classification)
        if not numerical_consistency[0]:
            validation_errors.append(numerical_consistency[1])
        
        if classification.complexity_score > 0.7:
            if len(answer.split()) < 50:
                validation_errors.append("Answer may be too brief for complex query")
                suggestions.append("Provide more detailed explanation for complex calculations")
        
        is_valid = len(validation_errors) == 0
        error_summary = "; ".join(validation_errors) if validation_errors else "Answer validation passed"
        suggestion_summary = "; ".join(suggestions) if suggestions else None
        
        return is_valid, error_summary, suggestion_summary

    def _check_numerical_consistency(self, answer: str, classification: QueryClassification) -> Tuple[bool, str]:
        """Check numerical values in answer for internal consistency."""
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        
        if len(numbers) < 2:
            return True, "Insufficient numbers for consistency check"
        
        if "percentage points" in classification.numerical_expectations:
            if re.search(r'(\d+(?:\.\d+)?)\s*%.*?(\d+(?:\.\d+)?)\s*%', answer):
                if "percentage points" not in answer.lower():
                    return False, "May be confusing percentage with percentage points"
        
        return True, "Numerical consistency check passed"


class QueryProcessor:
    """Main query processing orchestrator."""
    
    def __init__(self):
        self.classifier = QueryClassifier()
        self.enhancer = QueryEnhancer()
        self.validator = AnswerValidator()
        self.strategies = self._initialize_strategies()

    def _initialize_strategies(self) -> Dict[QueryType, QueryStrategy]:
        """Initialize query strategies for different types."""
        return {
            QueryType.FINANCIAL_CALCULATION: QueryStrategy(
                name="financial_calculation",
                retrieval_params={"k": 15, "padding": 5, "min_score": 0.05},
                processing_steps=["extract_numbers", "validate_calculations", "cross_reference"],
                validation_rules=["numerical_consistency", "unit_validation", "calculation_verification"],
                fallback_strategies=["broad_search", "document_specific_search"]
            ),
            QueryType.BUSINESS_METRICS: QueryStrategy(
                name="business_metrics",
                retrieval_params={"k": 12, "padding": 4, "min_score": 0.075},
                processing_steps=["extract_metrics", "validate_context", "time_series_check"],
                validation_rules=["metric_consistency", "unit_validation"],
                fallback_strategies=["financial_calculation", "broad_search"]
            ),
            QueryType.MARKET_ANALYSIS: QueryStrategy(
                name="market_analysis", 
                retrieval_params={"k": 10, "padding": 3, "min_score": 0.1},
                processing_steps=["extract_trends", "validate_comparisons", "source_verification"],
                validation_rules=["trend_consistency", "comparison_validity"],
                fallback_strategies=["broad_search"]
            ),
            QueryType.GENERAL_INQUIRY: QueryStrategy(
                name="general_inquiry",
                retrieval_params={"k": 20, "padding": 6, "min_score": 0.03},
                processing_steps=["broad_extraction", "relevance_ranking"],
                validation_rules=["relevance_check"],
                fallback_strategies=[]
            )
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the complete enhancement pipeline."""
        classification = self.classifier.classify_query(query)
        logger.info(f"Query classified as: {classification.primary_type.value}")
        
        enhanced_query = self.enhancer.enhance_query(query, classification)
        logger.info("Query enhanced with domain context")
        
        strategy = self.strategies.get(classification.primary_type, self.strategies[QueryType.GENERAL_INQUIRY])
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "classification": classification,
            "strategy": strategy,
            "processing_metadata": {
                "complexity_score": classification.complexity_score,
                "key_terms": classification.key_terms,
                "expected_answer_type": classification.expected_answer_type.value,
                "numerical_expectations": classification.numerical_expectations
            }
        }

    def validate_answer(self, answer: str, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an answer against the query requirements."""
        classification = processing_result["classification"]
        is_valid, error_msg, suggestions = self.validator.validate_answer(answer, classification)
        
        return {
            "is_valid": is_valid,
            "validation_errors": error_msg,
            "suggestions": suggestions,
            "answer_quality_score": self._calculate_answer_quality(answer, classification)
        }

    def _calculate_answer_quality(self, answer: str, classification: QueryClassification) -> float:
        """Calculate a quality score for the answer."""
        if not answer or answer.startswith("Unable to find"):
            return 0.0
        
        quality_score = 0.5
        
        word_count = len(answer.split())
        if classification.complexity_score > 0.7 and word_count > 30:
            quality_score += 0.2
        elif classification.complexity_score < 0.3 and word_count < 100:
            quality_score += 0.1
        
        if classification.expected_answer_type in [AnswerType.NUMERICAL, AnswerType.PERCENTAGE, AnswerType.CURRENCY]:
            if re.search(r'\d+', answer):
                quality_score += 0.2
        
        if re.search(r'\(.*\)|Source:|Note \d+', answer):
            quality_score += 0.1
        
        return min(1.0, quality_score)


if __name__ == "__main__":
    processor = QueryProcessor()
    
    test_query = """Based on Palantir's reported Q1 2024 and Q1 2025 revenue figures, what are the year-over-year growth rates for total revenue, total revenue excluding strategic commercial contracts, and US revenue; what is the standard deviation across those growth rates?"""
    
    result = processor.process_query(test_query)
    print(f"Query Type: {result['classification'].primary_type.value}")
    print(f"Complexity: {result['classification'].complexity_score:.2f}")
    print(f"Enhanced Query: {result['enhanced_query'][:200]}...")
    print(f"Strategy: {result['strategy'].name}")
