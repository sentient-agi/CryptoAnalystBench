import pandas as pd
import json
import asyncio
import os
import numpy as np
import ast
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import random
from itertools import permutations
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.llms.llm import Fireworks_LLM

load_dotenv()
from src.llms.judge import ChatJudgeLLM, parse_llm_json_response

# Trace context for scoring: span name and output base (same as llm_evaluation_system_with_context)
TRACE_SPAN_NAME = "Cyrpto Final Response"
OUTPUT_BASE = "output"


def _trace_input_to_context(value: Any) -> str:
    """Extract context from span attributes.input.value. Uses messages[-1].content when value has 'messages'; else string or json."""
    if value is None:
        return ""
    if isinstance(value, dict) and "messages" in value:
        messages = value.get("messages", [])
        if not messages:
            return ""
        content = messages[-1].get("content", "")
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

eval_datetime = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
@dataclass
class ModelScore:
    """Data class to store model scores and reasoning for each parameter."""
    temporal_relevance: int
    temporal_reasoning: str
    data_consistency: int
    data_consistency_reasoning: str
    depth: int
    depth_reasoning: str
    relevance: int
    relevance_reasoning: str

@dataclass
class QueryEvaluation:
    """Data class to store complete evaluation for a query."""
    query: str
    model_scores: Dict[str, ModelScore]
    rankings: List[Tuple[str, int]]  # (model_name, rank) - single ranking including all models
    overall_analysis: str

class LLMEvaluationSystem:
    """Comprehensive LLM evaluation system using Deepseek as judge."""
    
    def __init__(self, csv_path: str, models_to_evaluate: List[str], num_workers: int = 3):
        """
        Initialize the evaluation system.

        Args:
            csv_path: Path to the CSV file containing evaluation data
            models_to_evaluate: List of model names to evaluate (e.g., ['Sentient', 'gpt5', 'grok4', 'pplx'])
            num_workers: Number of parallel workers for processing queries (default: 3)
        """
        self.csv_path = csv_path
        self.models_to_evaluate = models_to_evaluate
        self.num_workers = num_workers
        self.df = None
        self.evaluations = []
        self.model_to_chat_id_column = {}

        # Initialize balanced randomization
        self._init_balanced_randomization()
    
    def _init_balanced_randomization(self):
        """
        Initialize balanced randomization system.
        Generates all possible permutations of model order and shuffles them for balanced distribution.
        """
        num_models = len(self.models_to_evaluate)
        # Generate all possible permutations
        self.all_permutations = list(permutations(range(num_models)))
        # Shuffle the list of permutations for randomness, but we'll cycle through all of them
        random.shuffle(self.all_permutations)
        # Counter to track which permutation to use next
        self.permutation_counter = 0
        # Lock to ensure thread-safe access to counter in parallel processing
        self.permutation_lock = asyncio.Lock()
        logger.info(f"Initialized balanced randomization with {len(self.all_permutations)} permutations for {num_models} models")
    
    async def _get_balanced_permutation(self) -> List[int]:
        """
        Get the next permutation in a balanced rotation.
        Cycles through all permutations to ensure equal distribution.
        Thread-safe for parallel query processing.
        
        Returns:
            List of indices representing the order of models
        """
        async with self.permutation_lock:
            permutation = self.all_permutations[self.permutation_counter % len(self.all_permutations)]
            self.permutation_counter += 1
            return list(permutation)
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the CSV data, reading response for each model."""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} rows from {self.csv_path}")
            
            if 'query' not in self.df.columns:
                raise ValueError("CSV must contain 'query' column")
            
            # Get response columns for each model
            response_columns = []
            for model_name in self.models_to_evaluate:
                col_name = f"{model_name}_response"
                if col_name not in self.df.columns:
                    raise ValueError(f"Missing column for model '{model_name}': {col_name}")
                response_columns.append(col_name)
            
            # Create a mapping from model name to response column
            self.model_to_column = {model: f"{model}_response" for model in self.models_to_evaluate}

            # Per-model chat_id for trace context: <model>_chat_id
            for model_name in self.models_to_evaluate:
                chat_id_col = f"{model_name}_chat_id"
                if chat_id_col in self.df.columns:
                    self.model_to_chat_id_column[model_name] = chat_id_col
            if self.model_to_chat_id_column:
                logger.info(f"Using per-model _chat_id columns for trace context from {OUTPUT_BASE}/<model>/traces/")

            # Filter rows with valid queries and at least one response
            self.df = self.df[self.df['query'].notna() & (self.df['query'] != '')]
            # Filter rows where at least one response exists
            has_responses = self.df[response_columns].notna().any(axis=1)
            self.df = self.df[has_responses]
            
            # Remove duplicate queries - keep only the first occurrence of each unique query
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=['query'], keep='first')
            duplicates_removed = initial_count - len(self.df)
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate queries. {len(self.df)} unique queries remain.")
            else:
                logger.info(f"No duplicate queries found. {len(self.df)} unique queries remain.")
            
            logger.info(f"After cleaning: {len(self.df)} rows remain with valid responses for models: {self.models_to_evaluate}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_context(self, row: pd.Series, model_name: str) -> str:
        """
        Read traces for this row/model and return the final context string to use when scoring.
        Uses model_to_chat_id_column to get chat_id, loads output/<model>/traces/<chat_id>.json,
        finds span TRACE_SPAN_NAME and returns its attributes.input.value as context; returns "" if missing.
        """
        chat_id_col = self.model_to_chat_id_column.get(model_name)
        if not chat_id_col:
            return ""
        chat_id = row.get(chat_id_col)
        if pd.isna(chat_id) or not str(chat_id).strip():
            return ""
        chat_id = str(chat_id).strip()
        trace_path = Path(OUTPUT_BASE) / model_name / "traces" / f"{chat_id}.json"
        if not trace_path.exists():
            return ""
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for trace in data.get("traces", []):
            for span in trace.get("spans", []):
                if span.get("name") == TRACE_SPAN_NAME:
                    return _trace_input_to_context(span.get("attributes.input.value")) or ""
        return ""

    def _create_scoring_prompt(self, query: str, response: str, response_num: int, trace_context: str = None) -> str:
        """Create prompt for scoring a single model response using detailed quality check criteria."""
        context_block = ""
        if trace_context and trace_context.strip():
            context_block = f"""
CONTEXT THAT WAS AVAILABLE TO THE MODEL (evaluate factual accuracy and relevance against this context only; do not use your own knowledge):
---
{trace_context}
---
"""
        return f"""Now I need you to evaluate response #{response_num} for this query:

EVALUATION DATE AND TIME: {eval_datetime}

QUERY: {query}
{context_block}
RESPONSE #{response_num}: {response}

Please score this response on the 4 parameters using the detailed evaluation criteria below (1-10 scale) and provide your reasoning.

EVALUATION CRITERIA:

1. TEMPORAL RELEVANCE (1-10): How current and timely is the information?
   - Does it reflect recent data, events, or market conditions?
   - Is the information up-to-date for crypto/blockchain context?
   - Does it avoid outdated or stale information?
   - Consider the current date ({eval_datetime}) when evaluating temporal relevance

2. DATA CONSISTENCY (1-10): How consistent and contradiction-free is the information within the response?
   - Are there any contradictions between different claims made in the response?
   - Do the statements, facts, and conclusions presented align with each other logically?
   - Are there conflicting pieces of information about the same topic within the response?
   - Do the numbers, dates, or metrics mentioned remain consistent throughout the response?
   - Are there any logical inconsistencies or self-contradictory statements?
   - Does the response maintain internal coherence without conflicting claims?

3. DEPTH (1-10): How comprehensive and detailed is the response?
   - Does it provide sufficient technical detail for blockchain concepts, protocols, and implementations?
   - Are complex blockchain concepts (consensus mechanisms, smart contracts, DeFi protocols) explained clearly?
   - Is the response well-organized with clear sections for technical analysis, market data, and risk assessment?
   - Are formulas, calculations, financial metrics, and technical specifications presented clearly?
   - Does the response address all relevant aspects of the crypto query with appropriate depth?

4. RELEVANCE (1-10): How well does the response address the specific question asked?
   - Does the response directly address the specific crypto question asked?
   - Does it provide practical information for crypto decision-making (investment, development, security)?
   - Does it appropriately highlight risks, limitations, and security considerations?
   - Is the information applicable to real-world crypto scenarios (trading, DeFi, development)?
   - Does it help users understand complex crypto concepts and make informed decisions?

IMPORTANT: Respond with ONLY valid JSON in English. Do not include any text before or after the JSON. Do not use markdown formatting. All responses must be in English only.

Required JSON format:
{{
    "temporal_relevance": {{
        "score": <1-10>,
        "reasoning": "<detailed explanation based on temporal relevance criteria>"
    }},
    "data_consistency": {{
        "score": <1-10>,
        "reasoning": "<detailed explanation based on data consistency criteria>"
    }},
    "depth": {{
        "score": <1-10>,
        "reasoning": "<detailed explanation based on depth and comprehensiveness criteria>"
    }},
    "relevance": {{
        "score": <1-10>,
        "reasoning": "<detailed explanation based on relevance and practical value criteria>"
    }}
}}"""
    
    def _create_ranking_prompt(self, query: str, model_scores: Dict[str, ModelScore]) -> str:
        """Create prompt for ranking all models based on the 4 scoring parameters' definitions."""
        # Show model responses with their reasoning for each parameter
        responses_summary = ""
        for model_name, scores in model_scores.items():
            responses_summary += f"""
{model_name} Response Analysis:
- Temporal Relevance: {scores.temporal_reasoning}
- Data Consistency: {scores.data_consistency_reasoning}
- Depth: {scores.depth_reasoning}
- Relevance: {scores.relevance_reasoning}
"""

        num_models = len(model_scores)
        rank_range = f"1 to worst ({num_models})" if num_models > 1 else "1"
        
        # Generate dynamic ranking JSON example based on number of models
        ranking_examples = ",\n        ".join([f'{{"model": "<model_name>", "rank": {i+1}}}' for i in range(num_models)])
        
        return f"""Now I need you to rank all models based on how well their responses perform according to the 4 detailed evaluation criteria for this query:

EVALUATION DATE AND TIME: {eval_datetime}

QUERY: {query}

DETAILED EVALUATION CRITERIA:

1. TEMPORAL RELEVANCE: How current and timely is the information?
   - Does it reflect recent data, events, or market conditions?
   - Is the information up-to-date for crypto/blockchain context?
   - Does it avoid outdated or stale information?
   - Consider the current date ({eval_datetime}) when evaluating temporal relevance

2. DATA CONSISTENCY: How consistent and contradiction-free is the information within the response?
   - Are there any contradictions between different claims made in the response?
   - Do the statements, facts, and conclusions presented align with each other logically?
   - Are there conflicting pieces of information about the same topic within the response?
   - Do the numbers, dates, or metrics mentioned remain consistent throughout the response?
   - Are there any logical inconsistencies or self-contradictory statements?
   - Does the response maintain internal coherence without conflicting claims?

3. DEPTH: How comprehensive and detailed is the response?
   - Does it provide sufficient technical detail for blockchain concepts, protocols, and implementations?
   - Are complex blockchain concepts (consensus mechanisms, smart contracts, DeFi protocols) explained clearly?
   - Is the response well-organized with clear sections for technical analysis, market data, and risk assessment?
   - Are formulas, calculations, financial metrics, and technical specifications presented clearly?
   - Does the response address all relevant aspects of the crypto query with appropriate depth?

4. RELEVANCE: How well does the response address the specific question asked?
   - Does the response directly address the specific crypto question asked?
   - Does it provide practical information for crypto decision-making (investment, development, security)?
   - Does it appropriately highlight risks, limitations, and security considerations?
   - Is the information applicable to real-world crypto scenarios (trading, DeFi, development)?
   - Does it help users understand complex crypto concepts and make informed decisions?

MODEL RESPONSES ANALYSIS:
{responses_summary}

Please rank them from best ({rank_range}) based on how well each response meets the 4 detailed evaluation criteria above. Consider the quality of reasoning and performance against each criterion, not just the scores.

IMPORTANT: Respond with ONLY valid JSON. Do not include any text before or after the JSON. Do not use markdown formatting. All responses must be in English only.

Required JSON format:
{{
    "rankings": [
        {ranking_examples}
    ],
    "overall_reasoning": "<comprehensive explanation of the ranking order based on the 4 detailed evaluation criteria>"
}}"""
    
    def _create_judge_llm(self) -> ChatJudgeLLM:
        """Create a new judge LLM instance with system prompt."""
        fireworks_llm = Fireworks_LLM(
            model_id="accounts/fireworks/models/deepseek-v3p1",
            api_key=None,
            temperature=0.1,
            max_tokens=10000,
        )
        system_prompt = f"""You are an expert LLM evaluator specialized in crypto/blockchain domain evaluation. 
Current evaluation date and time: {eval_datetime}

You will evaluate responses on 4 key parameters with detailed criteria:

1. TEMPORAL RELEVANCE (1-10): How current and timely is the information?
   - Does it reflect recent data, events, or market conditions?
   - Is the information up-to-date for crypto/blockchain context?
   - Does it avoid outdated or stale information?
   - Consider the current date ({eval_datetime}) when evaluating temporal relevance

2. DATA CONSISTENCY (1-10): How consistent and contradiction-free is the information within the response?
   - Are there any contradictions between different claims made in the response?
   - Do the statements, facts, and conclusions presented align with each other logically?
   - Are there conflicting pieces of information about the same topic within the response?
   - Do the numbers, dates, or metrics mentioned remain consistent throughout the response?
   - Are there any logical inconsistencies or self-contradictory statements?
   - Does the response maintain internal coherence without conflicting claims?

3. DEPTH (1-10): How comprehensive and detailed is the response?
   - Does it provide sufficient technical detail for blockchain concepts, protocols, and implementations?
   - Are complex blockchain concepts (consensus mechanisms, smart contracts, DeFi protocols) explained clearly?
   - Is the response well-organized with clear sections for technical analysis, market data, and risk assessment?
   - Are formulas, calculations, financial metrics, and technical specifications presented clearly?
   - Does the response address all relevant aspects of the crypto query with appropriate depth?

4. RELEVANCE (1-10): How well does the response address the specific question asked?
   - Does the response directly address the specific crypto question asked?
   - Does it provide practical information for crypto decision-making (investment, development, security)?
   - Does it appropriately highlight risks, limitations, and security considerations?
   - Is the information applicable to real-world crypto scenarios (trading, DeFi, development)?
   - Does it help users understand complex crypto concepts and make informed decisions?

You will maintain consistency across evaluations and provide detailed reasoning for all scores and rankings.
Always respond in valid JSON format as requested. All responses must be in English only."""
        return ChatJudgeLLM(fireworks_llm, system_prompt)
    
    async def evaluate_single_query(self, row: pd.Series, judge_llm: ChatJudgeLLM = None) -> QueryEvaluation:
        """Evaluate a single query with all models."""
        query = row['query']
        
        # Get responses from CSV using response columns
        models = {}
        for model_name in self.models_to_evaluate:
            col_name = self.model_to_column[model_name]
            response = row.get(col_name, "")
            if pd.isna(response):
                response = ""
            models[model_name] = str(response)

        # Use provided judge_llm or create a new one for this query (ensures thread isolation)
        if judge_llm is None:
            judge_llm = self._create_judge_llm()
        
        logger.info(f"Evaluating query: {query[:100]}...")
        if any(self.get_context(row, m) for m in self.models_to_evaluate):
            logger.info("Using trace context for scoring")
        
        # Clear conversation history for this new query evaluation
        judge_llm.clear_conversation_history()
        
        # Get balanced permutation for model order
        permutation = await self._get_balanced_permutation()
        # Apply permutation to models
        model_list = [self.models_to_evaluate[i] for i in permutation]
        logger.info(f"Evaluation order: {model_list}")
        
        # Step 1: Score each model in randomized order with retry logic
        model_scores = {}
        for eval_pos, model_name in enumerate(model_list, 1):
            response = models[model_name]
            if not response or not response.strip():
                logger.warning(f"Response for {model_name} is empty, using default scores")
                model_scores[model_name] = ModelScore(
                    temporal_relevance=1, temporal_reasoning="Empty response",
                    data_consistency=1, data_consistency_reasoning="Empty response",
                    depth=1, depth_reasoning="Empty response",
                    relevance=1, relevance_reasoning="Empty response"
                )
                continue
            
            max_retries = 3
            retry_count = 0
            success = False
            result = None
            
            while retry_count < max_retries and not success:
                try:
                    trace_context = self.get_context(row, model_name)
                    prompt = self._create_scoring_prompt(query, response, eval_pos, trace_context=trace_context)
                    result = await judge_llm.a_generate(prompt)
                    
                    if not result or not result.strip():
                        raise ValueError("Empty response from judge LLM")

                    scores_data = parse_llm_json_response(result)
                    
                    # Validate required fields
                    required_fields = ['temporal_relevance', 'data_consistency', 'depth', 'relevance']
                    for field in required_fields:
                        if field not in scores_data or 'score' not in scores_data[field] or 'reasoning' not in scores_data[field]:
                            raise ValueError(f"Missing required field: {field}")
                        
                        score = scores_data[field]['score']
                        if not isinstance(score, int) or score < 1 or score > 10:
                            raise ValueError(f"Invalid score for {field}: {score}")
                    
                    model_scores[model_name] = ModelScore(
                        temporal_relevance=scores_data['temporal_relevance']['score'],
                        temporal_reasoning=scores_data['temporal_relevance']['reasoning'],
                        data_consistency=scores_data['data_consistency']['score'],
                        data_consistency_reasoning=scores_data['data_consistency']['reasoning'],
                        depth=scores_data['depth']['score'],
                        depth_reasoning=scores_data['depth']['reasoning'],
                        relevance=scores_data['relevance']['score'],
                        relevance_reasoning=scores_data['relevance']['reasoning']
                    )
                    
                    logger.info(f"Scored {model_name}: TR={scores_data['temporal_relevance']['score']}, "
                              f"DC={scores_data['data_consistency']['score']}, "
                              f"D={scores_data['depth']['score']}, R={scores_data['relevance']['score']}")
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error scoring {model_name} (attempt {retry_count}/{max_retries}): {e}")
                    if result:
                        logger.info(f"Raw response: {result[:200]}...")
                    
                    if retry_count < max_retries:
                        # Add a small delay before retry
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"Failed to score {model_name} after {max_retries} attempts")
                        # Provide default scores if all retries fail
                        model_scores[model_name] = ModelScore(
                            temporal_relevance=1, temporal_reasoning="Error in evaluation after retries",
                            data_consistency=1, data_consistency_reasoning="Error in evaluation after retries",
                            depth=1, depth_reasoning="Error in evaluation after retries",
                            relevance=1, relevance_reasoning="Error in evaluation after retries"
                        )
        
        # Step 2: Create single ranking for all models
        max_retries = 3
        retry_count = 0
        success = False
        rankings = []
        ranking_result = None
        overall_reasoning = ""
        
        while retry_count < max_retries and not success:
            try:
                ranking_prompt = self._create_ranking_prompt(query, model_scores)
                ranking_result = await judge_llm.a_generate(ranking_prompt)
                
                if not ranking_result or not ranking_result.strip():
                    raise ValueError("Empty response from judge LLM")

                ranking_data = parse_llm_json_response(ranking_result)
                
                # Validate required fields
                if 'rankings' not in ranking_data:
                    raise ValueError("Missing 'rankings' field in response")
                
                rankings_list = ranking_data['rankings']
                if not isinstance(rankings_list, list) or len(rankings_list) == 0:
                    raise ValueError("Invalid rankings format")
                
                # Validate each ranking item
                for item in rankings_list:
                    if 'model' not in item or 'rank' not in item:
                        raise ValueError("Invalid ranking item format")
                    if not isinstance(item['rank'], int) or item['rank'] < 1:
                        raise ValueError(f"Invalid rank: {item['rank']}")
                
                # Create rankings for all models
                rankings = [(item['model'], item['rank']) for item in rankings_list]
                
                # Extract overall reasoning if available
                overall_reasoning = ranking_data.get('overall_reasoning', '')
                
                logger.info(f"Single ranking for all models: {rankings}")
                success = True
                
            except Exception as e:
                retry_count += 1
                logger.info(f"Error in ranking (attempt {retry_count}/{max_retries}): {e}")
                if ranking_result:
                    logger.info(f"Raw response: {ranking_result[:200]}...")
                
                if retry_count < max_retries:
                    # Add a small delay before retry
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Failed to rank after {max_retries} attempts")
                    # Fallback ranking based on total scores
                    total_scores = {
                        model: (scores.temporal_relevance + scores.data_consistency + 
                               scores.depth + scores.relevance)
                        for model, scores in model_scores.items()
                    }
                    fallback_rankings = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
                    rankings = [(model, i+1) for i, (model, _) in enumerate(fallback_rankings)]
                    overall_reasoning = "Fallback ranking based on total scores (ranking API failed)"
        
        return QueryEvaluation(
            query=query,
            model_scores=model_scores,
            rankings=rankings,
            overall_analysis=overall_reasoning
        )
    
    async def evaluate_all_queries(self, max_queries: int = None) -> List[QueryEvaluation]:
        """Evaluate all queries in the dataset using parallel workers.
        
        Args:
            max_queries: Maximum number of queries to evaluate (None for all)
        """
        if self.df is None:
            self.load_data()
        
        queries_to_evaluate = self.df.head(max_queries) if max_queries else self.df
        
        logger.info(f"Starting evaluation of {len(queries_to_evaluate)} queries with {self.num_workers} parallel workers")
        
        # Create semaphore to limit concurrent workers
        semaphore = asyncio.Semaphore(self.num_workers)
        
        async def evaluate_with_semaphore(idx: int, row: pd.Series) -> Tuple[int, QueryEvaluation]:
            """Evaluate a single query with semaphore control and isolated judge LLM."""
            async with semaphore:
                # Create a new judge LLM instance for this query to ensure thread isolation
                judge_llm = self._create_judge_llm()
                
                try:
                    evaluation = await self.evaluate_single_query(row, judge_llm)
                    logger.info(f"Completed evaluation {idx + 1}/{len(queries_to_evaluate)}")
                    return (idx, evaluation)
                except Exception as e:
                    logger.error(f"Error evaluating query {idx}: {e}")
                    raise
        
        # Create tasks for all queries
        tasks = [
            evaluate_with_semaphore(idx, row)
            for idx, (_, row) in enumerate(queries_to_evaluate.iterrows(), 1)
        ]
        
        # Execute all tasks in parallel (limited by semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        # Results are returned in the same order as tasks were submitted
        evaluations = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            idx, evaluation = result
            evaluations.append(evaluation)
        
        self.evaluations = evaluations
        logger.info(f"Completed evaluation of {len(evaluations)} queries")
        return evaluations
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all evaluations with comparative analysis."""
        if not self.evaluations:
            return "No evaluations available."
        
        # Calculate aggregate statistics
        total_queries = len(self.evaluations)
        
        # Count wins for each model
        model_wins = {model: 0 for model in self.models_to_evaluate}
        
        # Calculate average scores
        avg_scores = {model: {"TR": 0, "DC": 0, "D": 0, "R": 0} for model in self.models_to_evaluate}
        
        for eval in self.evaluations:
            # Count wins (rank 1) in main rankings
            winner = next((model for model, rank in eval.rankings if rank == 1), None)
            if winner:
                model_wins[winner] += 1
            
            # Accumulate scores
            for model_name, scores in eval.model_scores.items():
                avg_scores[model_name]["TR"] += scores.temporal_relevance
                avg_scores[model_name]["DC"] += scores.data_consistency
                avg_scores[model_name]["D"] += scores.depth
                avg_scores[model_name]["R"] += scores.relevance
            
        
        # Calculate averages
        for model in avg_scores:
            for metric in avg_scores[model]:
                avg_scores[model][metric] /= total_queries
        
        # Generate report
        report = f"""
# LLM Evaluation Summary Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Queries Evaluated: {total_queries}

## Model Performance Summary

### Win Count (Rank #1)
"""
        for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
            percentage = (wins / total_queries) * 100
            report += f"- {model}: {wins} wins ({percentage:.1f}%)\n"
        
        report += "\n### Average Scores by Model\n"
        for model, scores in avg_scores.items():
            avg_total = sum(scores.values()) / len(scores)
            report += f"\n**{model}** (Avg Total: {avg_total:.2f})\n"
            report += f"- Temporal Relevance: {scores['TR']:.2f}\n"
            report += f"- Data Consistency: {scores['DC']:.2f}\n"
            report += f"- Depth: {scores['D']:.2f}\n"
            report += f"- Relevance: {scores['R']:.2f}\n"
        
        return report
    
    # Enhanced Tag Insights Methods
    def extract_tags_from_list(self, tags_str: str) -> List[str]:
        """Extract individual tags from the tags column."""
        if pd.isna(tags_str):
            return []
        try:
            # Remove brackets and quotes, split by comma
            tags = tags_str.strip("[]").replace("'", "").replace('"', "")
            return [tag.strip() for tag in tags.split(",") if tag.strip()]
        except (AttributeError, TypeError, ValueError):
            return []
    
    def create_evaluation_results_csv(self, output_path: str = None) -> str:
        """Create a CSV file with evaluation results for enhanced tag insights analysis."""
        if not self.evaluations:
            raise ValueError("No evaluations to save. Run evaluate_all_queries first.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/output/final_evaluation_results_{timestamp}.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Create a mapping from query to original row data for preserving all fields
        query_to_original_data = {}
        if hasattr(self, 'df') and self.df is not None:
            for _, row in self.df.iterrows():
                query_to_original_data[row['query']] = row
        
        # Create results data for CSV - one row per model per query
        results_data = []
        for eval in self.evaluations:
            # Get original data for this query to preserve all fields
            original_data = query_to_original_data.get(eval.query, {})
            
            # Create a row for each model
            for model_name, model_scores in eval.model_scores.items():
                # Get this model's rank
                model_rank = next((rank for model, rank in eval.rankings if model == model_name), len(self.models_to_evaluate))
                
                # Base row with common fields
                result_row = {
                    'query': eval.query,
                    'model': model_name,
                    'rank': model_rank,
                    'temporal_relevance': model_scores.temporal_relevance,
                    'data_consistency': model_scores.data_consistency,
                    'depth': model_scores.depth,
                    'relevance': model_scores.relevance,
                    'ranking_reasoning': eval.overall_analysis,  # Same reasoning for all models in this query
                    # 'total_score': model_scores.temporal_relevance + model_scores.data_consistency + model_scores.depth + model_scores.relevance,
                    'tags': original_data.get('tags', ''),  # Preserve original tags
                }
                
                # Add all other fields from original data that might exist
                for key, value in original_data.items():
                    if key not in result_row and key not in self.models_to_evaluate:
                        result_row[key] = value
                
                results_data.append(result_row)
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Evaluation results CSV saved to {output_path}")
        return output_path
    
    def calculate_tag_metrics(self, results_csv_path: str) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive metrics for each tag from the results CSV."""
        logger.info("Calculating tag metrics...")
        
        # Load the results CSV
        results_df = pd.read_csv(results_csv_path)
        
        # Check if tags column exists and has data
        if 'tags' not in results_df.columns:
            logger.info("No tags column found in results CSV")
            results_df['tags'] = ''
        elif results_df['tags'].isna().all() or (results_df['tags'] == '').all():
            logger.info("Tags column exists but is empty, attempting to extract from original data...")
            # Try to match queries with original data to get tags
            if hasattr(self, 'df') and self.df is not None:
                original_tags = {}
                for _, row in self.df.iterrows():
                    if 'tags' in row and pd.notna(row['tags']):
                        original_tags[row['query']] = row['tags']
                
                # Add tags to results
                results_df['tags'] = results_df['query'].map(original_tags).fillna('')
            else:
                logger.info("No original data available for tag extraction")
                results_df['tags'] = ''
        else:
            logger.info(f"Found {len(results_df[results_df['tags'].notna() & (results_df['tags'] != '')])} rows with tags data")
        
        # Create a list to store all tag-query combinations (analyze all models)
        tag_data = []
        
        # Analyze all models
        model_rows = results_df
        
        for _, row in model_rows.iterrows():
            tags = self.extract_tags_from_list(row['tags'])
            for tag in tags:
                tag_data.append({
                    'tag': tag,
                    'query': row['query'],
                    'rank': row['rank'],
                    'temporal_relevance': row['temporal_relevance'],
                    'data_consistency': row['data_consistency'],
                    'depth': row['depth'],
                    'relevance': row['relevance']
                })
        
        if not tag_data:
            logger.info("No tag data found for analysis")
            return {}
        
        tag_df = pd.DataFrame(tag_data)
        
        # Group by tag and calculate metrics
        tag_metrics = {}
        
        for tag in tag_df['tag'].unique():
            tag_subset = tag_df[tag_df['tag'] == tag]
            
            # Calculate rank distribution
            rank_counts = tag_subset['rank'].value_counts().sort_index()
            num_models = len(self.models_to_evaluate)
            
            # Dynamically create rank counts for all possible ranks
            rank_count_dict = {}
            for rank in range(1, num_models + 1):
                rank_count_dict[f'rank_{rank}_count'] = rank_counts.get(rank, 0)
            
            # Calculate averages
            avg_rank = tag_subset['rank'].mean()
            avg_temporal_relevance = tag_subset['temporal_relevance'].mean()
            avg_data_consistency = tag_subset['data_consistency'].mean()
            avg_depth = tag_subset['depth'].mean()
            avg_relevance = tag_subset['relevance'].mean()
            
            tag_metrics[tag] = {
                'total_queries': len(tag_subset),
                'avg_rank': round(avg_rank, 9),
                **rank_count_dict,
                'avg_temporal_relevance': round(avg_temporal_relevance, 9),
                'avg_data_consistency': round(avg_data_consistency, 9),
                'avg_depth': round(avg_depth, 9),
                'avg_relevance': round(avg_relevance, 9),
                'queries': tag_subset['query'].tolist(),
                'ranks': tag_subset['rank'].tolist()
            }
        
        logger.info(f"Calculated metrics for {len(tag_metrics)} tags")
        return tag_metrics
    
    def calculate_per_model_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate aggregate statistics for each model.
        
        Returns:
            Dictionary with statistics for each model
        """
        if not self.evaluations:
            raise ValueError("No evaluations available. Run evaluate_all_queries first.")
        
        metric_types = ['temporal_relevance', 'data_consistency', 'depth', 'relevance']
        model_stats = {}
        
        for model_name in self.models_to_evaluate:
            model_stats[model_name] = {}
            
            for metric_type in metric_types:
                # Collect all scores for this model and metric
                all_scores = []
                for eval in self.evaluations:
                    if model_name in eval.model_scores:
                        scores = eval.model_scores[model_name]
                        if metric_type == 'temporal_relevance':
                            score = scores.temporal_relevance
                        elif metric_type == 'data_consistency':
                            score = scores.data_consistency
                        elif metric_type == 'depth':
                            score = scores.depth
                        else:  # relevance
                            score = scores.relevance
                        
                        if isinstance(score, (int, float)) and not pd.isna(score):
                            all_scores.append(float(score))
                
                if all_scores:
                    model_stats[model_name][metric_type] = {
                        'mean': float(np.mean(all_scores)),
                        'std': float(np.std(all_scores)),
                        'variance': float(np.var(all_scores)),
                        'min': float(np.min(all_scores)),
                        'max': float(np.max(all_scores)),
                        'median': float(np.median(all_scores)),
                        'q25': float(np.percentile(all_scores, 25)),
                        'q75': float(np.percentile(all_scores, 75)),
                        'count': len(all_scores)
                    }
                else:
                    model_stats[model_name][metric_type] = {
                        'mean': 0.0, 'std': 0.0, 'variance': 0.0,
                        'min': 0, 'max': 0, 'median': 0.0,
                        'q25': 0.0, 'q75': 0.0, 'count': 0
                    }
            
            # Calculate win count and average rank for this model
            wins = 0
            ranks = []
            for eval in self.evaluations:
                model_rank = next((rank for model, rank in eval.rankings if model == model_name), len(self.models_to_evaluate))
                ranks.append(model_rank)
                if model_rank == 1:
                    wins += 1
            
            model_stats[model_name]['overall'] = {
                'win_count': wins,
                'avg_rank': float(np.mean(ranks)) if ranks else 0.0,
                'total_queries': len(self.evaluations)
            }
        
        return model_stats

    async def perform_error_analysis_per_model(self, judge_llm: ChatJudgeLLM = None, batch_size: int = 50) -> Dict[str, Any]:
        """
        Perform per-model error taxonomy: for each model and metric, cluster low-score (<=8) patterns.
        Returns dict with 'per_model', 'total_queries', 'models_evaluated'.
        """
        if not self.evaluations:
            raise ValueError("No evaluations to analyze. Run evaluate_all_queries first.")
        if judge_llm is None:
            judge_llm = self._create_judge_llm()
        metrics = ['temporal_relevance', 'data_consistency', 'depth', 'relevance']
        metric_reasoning_map = {
            'temporal_relevance': 'temporal_reasoning',
            'data_consistency': 'data_consistency_reasoning',
            'depth': 'depth_reasoning',
            'relevance': 'relevance_reasoning'
        }
        model_metric_data = {m: {met: [] for met in metrics} for m in self.models_to_evaluate}
        for eval in self.evaluations:
            for model_name, scores in eval.model_scores.items():
                for metric in metrics:
                    score = getattr(scores, metric)
                    reasoning = getattr(scores, metric_reasoning_map[metric])
                    model_metric_data[model_name][metric].append({
                        'query': eval.query, 'score': score, 'reasoning': reasoning
                    })
        total_queries = len(self.evaluations)
        error_analysis = {
            'per_model': {},
            'total_queries': total_queries,
            'models_evaluated': self.models_to_evaluate.copy()
        }
        for model_name in self.models_to_evaluate:
            logger.info(f"Analyzing errors for model: {model_name}")
            error_analysis['per_model'][model_name] = {}
            for metric in metrics:
                data = model_metric_data[model_name][metric]
                low_score_data = [d for d in data if d['score'] <= 8]
                if not low_score_data:
                    error_analysis['per_model'][model_name][metric] = {
                        'error_clusters': [], 'total_low_scores': 0, 'message': 'No low scores found for this metric'
                    }
                    continue
                all_batch_clusters = await self._process_error_analysis_batches(
                    judge_llm, model_name, metric, low_score_data, batch_size
                )
                final_clusters, summary = await self._synthesize_error_clusters(
                    judge_llm, model_name, metric, all_batch_clusters, len(low_score_data)
                )
                error_analysis['per_model'][model_name][metric] = {
                    'error_clusters': final_clusters,
                    'total_low_scores': len(low_score_data),
                    'summary': summary
                }
                logger.info(f"  {metric}: {len(final_clusters)} error clusters from {len(low_score_data)} low scores")
        return error_analysis

    async def _process_error_analysis_batches(
        self, judge_llm: ChatJudgeLLM, model_name: str, metric: str,
        low_score_data: List[Dict], batch_size: int
    ) -> List[Dict]:
        """Process error analysis in batches (per-model only)."""
        all_batch_clusters = []
        num_batches = (len(low_score_data) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(low_score_data))
            batch_data = low_score_data[start_idx:end_idx]
            judge_llm.clear_conversation_history()
            prompt = self._create_error_analysis_batch_prompt(model_name, metric, batch_data, batch_idx + 1, num_batches)
            for attempt in range(3):
                try:
                    result = await judge_llm.a_generate(prompt)
                    analysis_data = parse_llm_json_response(result)
                    clusters = analysis_data.get('error_clusters', [])
                    all_batch_clusters.extend(clusters)
                    break
                except Exception as e:
                    logger.info(f"Error analyzing {model_name}/{metric} batch {batch_idx + 1} (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"Failed error analysis for {model_name}/{metric} batch {batch_idx + 1} after 3 attempts")
        return all_batch_clusters

    async def _synthesize_error_clusters(
        self, judge_llm: ChatJudgeLLM, model_name: str, metric: str,
        all_batch_clusters: List[Dict], total_low_scores: int
    ) -> Tuple[List[Dict], str]:
        """Synthesize batch clusters into final per-model error clusters."""
        if not all_batch_clusters:
            return [], "No error patterns found."
        if len(all_batch_clusters) <= 5:
            return all_batch_clusters, f"Found {len(all_batch_clusters)} error patterns from {total_low_scores} low-scoring entries."
        judge_llm.clear_conversation_history()
        prompt = self._create_error_synthesis_prompt(model_name, metric, all_batch_clusters, total_low_scores)
        for attempt in range(3):
            try:
                result = await judge_llm.a_generate(prompt)
                synthesis_data = parse_llm_json_response(result)
                final_clusters = synthesis_data.get('error_clusters', [])
                summary = synthesis_data.get('summary', '')
                return final_clusters, summary
            except Exception as e:
                logger.info(f"Error synthesizing {model_name}/{metric} clusters (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Failed to synthesize error clusters for {model_name}/{metric} after 3 attempts")
        return all_batch_clusters[:10], "Synthesis incomplete."

    def _create_error_analysis_batch_prompt(self, model_name: str, metric: str, batch_data: List[Dict], batch_num: int, total_batches: int) -> str:
        """Create prompt for a batch of per-model error analysis."""
        examples_text = "\n\n---\n\n".join([
            f"Query: {d['query']}\nScore: {d['score']}\nReasoning: {d['reasoning']}" for d in batch_data
        ])
        metric_display = metric.replace('_', ' ').title()
        return f"""Analyze the following low-scoring evaluations for model "{model_name}" on the "{metric_display}" metric.
This is batch {batch_num} of {total_batches}.

Identify COMMON ERROR PATTERNS. Group similar issues into distinct error clusters. For each cluster provide examples and occurrence count in this batch.

EVALUATION DATA ({len(batch_data)} entries):
{examples_text}

Respond with ONLY valid JSON:
{{
    "error_clusters": [
        {{
            "error_type": "<Name of the error pattern>",
            "description": "<Description of the error pattern>",
            "occurrence_count": <count in this batch>,
            "severity": "<high/medium/low>",
            "example_queries": ["<query1>", "<query2>"],
            "example_issues": ["<issue1>", "<issue2>"]
        }}
    ]
}}"""

    def _create_error_synthesis_prompt(self, model_name: str, metric: str, all_clusters: List[Dict], total_low_scores: int) -> str:
        """Create prompt for synthesizing per-model error clusters."""
        clusters_text = json.dumps(all_clusters, indent=2)
        metric_display = metric.replace('_', ' ').title()
        return f"""You analyzed {total_low_scores} low-scoring evaluations for model "{model_name}" on "{metric_display}" in multiple batches.

Merge similar error patterns, aggregate occurrence counts, consolidate into the main patterns. Provide an overall summary.

RAW ERROR PATTERNS FROM ALL BATCHES:
{clusters_text}

Respond with ONLY valid JSON:
{{
    "error_clusters": [
        {{
            "error_type": "<Consolidated error pattern name>",
            "description": "<Comprehensive description>",
            "occurrence_count": <aggregated count>,
            "severity": "<high/medium/low>",
            "example_queries": ["<query1>", "<query2>", "<query3>"],
            "example_issues": ["<issue1>", "<issue2>", "<issue3>"]
        }}
    ],
    "summary": "<Summary of main issues for {model_name} on {metric_display}>"
}}"""

    def extract_tags_from_string(self, tags_str):
        """Extract tags from string representation of list or direct text format."""
        try:
            if pd.isna(tags_str) or tags_str == '':
                return []
            
            # If already a list, return as is
            if isinstance(tags_str, list):
                return tags_str
            
            # If it's a string, try multiple parsing strategies
            if isinstance(tags_str, str):
                # Strategy 1: Try to parse as literal list (e.g., "['tag1', 'tag2']")
                try:
                    parsed = ast.literal_eval(tags_str)
                    if isinstance(parsed, list):
                        return [tag.strip() for tag in parsed if tag.strip()]
                except (ValueError, SyntaxError):
                    pass
                
                # Strategy 2: Handle direct text format - split by newlines if present
                # Remove leading/trailing whitespace
                tags_str = tags_str.strip()
                
                # If contains newlines, split by newlines
                if '\n' in tags_str:
                    tags = [tag.strip() for tag in tags_str.split('\n') if tag.strip()]
                    # Filter out empty strings and the word "tags" if it appears as a header
                    tags = [tag for tag in tags if tag and tag.lower() != 'tags']
                    return tags if tags else []
                
                # Strategy 3: Treat entire string as a single tag (direct format)
                return [tags_str]
            
            return []
        except (AttributeError, TypeError, ValueError):
            return []
    
    def calculate_tag_wise_rankings(self) -> pd.DataFrame:
        """Calculate tag-wise average rankings and metrics for all models.
        
        Returns:
            DataFrame with tag-wise rankings containing columns:
            tag, model, avg_rank, rank_1_count, rank_2_count, ..., rank_N_count
            (where N is the number of models), avg_temporal_relevance, 
            avg_data_consistency, avg_depth, avg_relevance, total_queries
        """
        if not self.evaluations:
            raise ValueError("No evaluations to analyze. Run evaluate_all_queries first.")
        
        # Get number of models to determine how many rank columns we need
        num_models = len(self.models_to_evaluate)
        
        # Create a mapping from query to original row data for preserving tags
        query_to_original_data = {}
        if hasattr(self, 'df') and self.df is not None:
            for _, row in self.df.iterrows():
                query_to_original_data[row['query']] = row
        
        # Build results data similar to create_evaluation_results_csv
        results_data = []
        for eval in self.evaluations:
            original_data = query_to_original_data.get(eval.query, {})
            
            for model_name, model_scores in eval.model_scores.items():
                model_rank = next((rank for model, rank in eval.rankings if model == model_name), len(self.models_to_evaluate))
                
                result_row = {
                    'query': eval.query,
                    'model': model_name,
                    'rank': model_rank,
                    'temporal_relevance': model_scores.temporal_relevance,
                    'data_consistency': model_scores.data_consistency,
                    'depth': model_scores.depth,
                    'relevance': model_scores.relevance,
                    'tags': original_data.get('tags', ''),
                }
                results_data.append(result_row)
        
        # Create DataFrame from results
        df = pd.DataFrame(results_data)
        
        # Extract tags for each row
        df['extracted_tags'] = df['tags'].apply(self.extract_tags_from_string)
        
        # Get all unique tags
        all_tags = set()
        for tags_list in df['extracted_tags']:
            all_tags.update(tags_list)
        
        all_tags = sorted(list(all_tags))
        
        # Get unique models
        models = df['model'].unique()
        
        # Calculate tag-wise metrics for each model
        tag_wise_results = []
        
        for tag in all_tags:
            # Filter data for this tag
            tag_data = df[df['extracted_tags'].apply(lambda x: tag in x if x else False)]
            
            if len(tag_data) == 0:
                continue
            
            for model in models:
                model_tag_data = tag_data[tag_data['model'] == model]
                
                if len(model_tag_data) == 0:
                    continue
                
                # Calculate metrics for this model-tag combination
                avg_rank = model_tag_data['rank'].mean()
                
                # Dynamically calculate rank counts for all possible ranks
                rank_counts = {}
                for rank_num in range(1, num_models + 1):
                    rank_counts[f'rank_{rank_num}_count'] = len(model_tag_data[model_tag_data['rank'] == rank_num])
                
                avg_temporal_relevance = model_tag_data['temporal_relevance'].mean()
                avg_data_consistency = model_tag_data['data_consistency'].mean()
                avg_depth = model_tag_data['depth'].mean()
                avg_relevance = model_tag_data['relevance'].mean()
                
                result_row = {
                    'tag': tag,
                    'model': model,
                    'avg_rank': round(avg_rank, 3),
                    **rank_counts,  # Unpack all rank count columns
                    'avg_temporal_relevance': round(avg_temporal_relevance, 3),
                    'avg_data_consistency': round(avg_data_consistency, 3),
                    'avg_depth': round(avg_depth, 3),
                    'avg_relevance': round(avg_relevance, 3),
                    'total_queries': len(model_tag_data)
                }
                tag_wise_results.append(result_row)
        
        # Create results DataFrame
        results_df = pd.DataFrame(tag_wise_results)
        
        # Sort by tag and then by average rank
        if len(results_df) > 0:
            results_df = results_df.sort_values(['tag', 'avg_rank'])
        
        return results_df
    
    def save_statistics_xlsx(self, output_path: str = None, error_analysis: Dict[str, Any] = None) -> str:
        """
        Save all statistics to a single XLSX file with multiple sheets.

        Args:
            output_path: Path to output XLSX file (None for auto-generated)
            error_analysis: Optional per-model error taxonomy from perform_error_analysis_per_model().

        Returns:
            Path to the saved XLSX file
        """
        if not self.evaluations:
            raise ValueError("No evaluations to save. Run evaluate_all_queries first.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/output/evaluation_statistics_{timestamp}.xlsx"
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Calculate statistics
        model_stats = self.calculate_per_model_statistics()
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Evaluation Results (previously saved as CSV)
            # Create a mapping from query to original row data for preserving all fields
            query_to_original_data = {}
            if hasattr(self, 'df') and self.df is not None:
                for _, row in self.df.iterrows():
                    query_to_original_data[row['query']] = row
            
            # Create results data - one row per model per query
            results_data = []
            for eval in self.evaluations:
                # Get original data for this query to preserve all fields
                original_data = query_to_original_data.get(eval.query, {})
                
                # Create a row for each model
                for model_name, model_scores in eval.model_scores.items():
                    # Get this model's rank
                    model_rank = next((rank for model, rank in eval.rankings if model == model_name), len(self.models_to_evaluate))
                    
                    # Base row with common fields
                    result_row = {
                        'query': eval.query,
                        'model': model_name,
                        'rank': model_rank,
                        'temporal_relevance': model_scores.temporal_relevance,
                        'temporal_relevance_reasoning': model_scores.temporal_reasoning,
                        'data_consistency': model_scores.data_consistency,
                        'data_consistency_reasoning': model_scores.data_consistency_reasoning,
                        'depth': model_scores.depth,
                        'depth_reasoning': model_scores.depth_reasoning,
                        'relevance': model_scores.relevance,
                        'relevance_reasoning': model_scores.relevance_reasoning,
                        'ranking_reasoning': eval.overall_analysis,  # Same reasoning for all models in this query
                        'tags': original_data.get('tags', ''),  # Preserve original tags
                    }
                    
                    # Add all other fields from original data that might exist
                    for key, value in original_data.items():
                        if key not in result_row and key not in self.models_to_evaluate:
                            result_row[key] = value
                    
                    results_data.append(result_row)
            
            # Create DataFrame and add as first sheet
            results_df = pd.DataFrame(results_data)
            results_df.to_excel(writer, sheet_name='Evaluation Results', index=False)
            
            # Sheet 2: Per-Model Statistics
            stats_data = []
            metric_names = {
                'temporal_relevance': 'Temporal Relevance',
                'data_consistency': 'Data Consistency',
                'depth': 'Depth',
                'relevance': 'Relevance'
            }
            
            for model_name, stats in model_stats.items():
                for metric_type, metric_name in metric_names.items():
                    if metric_type in stats:
                        m = stats[metric_type]
                        stats_data.append({
                            'Model': model_name,
                            'Metric': metric_name,
                            'Mean': round(m['mean'], 2),
                            'Std': round(m['std'], 2),
                            'Variance': round(m['variance'], 2),
                            'Min': int(m['min']),
                            'Max': int(m['max']),
                            'Median': round(m['median'], 2),
                            'Q25': round(m['q25'], 2),
                            'Q75': round(m['q75'], 2),
                            'Count': int(m['count'])
                        })
            
            df_stats = pd.DataFrame(stats_data)
            df_stats.to_excel(writer, sheet_name='Per-Model Statistics', index=False)
            
            # Sheet 3: Tag-wise Rankings
            try:
                tag_wise_df = self.calculate_tag_wise_rankings()
                if len(tag_wise_df) > 0:
                    tag_wise_df.to_excel(writer, sheet_name='Tag-wise Rankings', index=False)
                    logger.info("Tag-wise rankings added to XLSX")
                else:
                    logger.info("No tag-wise rankings data available")
            except Exception as e:
                logger.warning(f"Could not generate tag-wise rankings: {e}")

            # Sheet 4: Per-Model Error Analysis (always present)
            per_model_data = []
            if error_analysis and error_analysis.get('per_model'):
                total_queries = error_analysis.get('total_queries', 0)
                for model_name, model_data in error_analysis['per_model'].items():
                    for metric, metric_data in model_data.items():
                        metric_display = metric.replace('_', ' ').title()
                        clusters = metric_data.get('error_clusters', [])
                        total_low = metric_data.get('total_low_scores', 0)
                        for cluster in clusters:
                            per_model_data.append({
                                'Model': model_name,
                                'Metric': metric_display,
                                'Error Type': cluster.get('error_type', 'Unknown'),
                                'Description': cluster.get('description', ''),
                                'Occurrence Count': cluster.get('occurrence_count', 0),
                                'Total Low Scores': total_low,
                                'Total Queries': total_queries,
                                'Severity': cluster.get('severity', 'medium'),
                                'Example Queries': '; '.join(cluster.get('example_queries', [])),
                                'Example Issues': '; '.join(cluster.get('example_issues', []))
                            })
            columns = ['Model', 'Metric', 'Error Type', 'Description', 'Occurrence Count', 'Total Low Scores', 'Total Queries', 'Severity', 'Example Queries', 'Example Issues']
            df_per_model = pd.DataFrame(per_model_data, columns=columns) if per_model_data else pd.DataFrame(columns=columns)
            df_per_model.to_excel(writer, sheet_name='Error Analysis - Per Model', index=False)
            logger.info("Per-model error analysis sheet written to XLSX")
            
        
        logger.info(f"Statistics XLSX saved to {output_path}")
        return output_path
    
def parse_arguments():
    """Parse command-line arguments for LLM evaluation system."""
    parser = argparse.ArgumentParser(
        description='LLM Evaluation System - Runs complete evaluation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python script.py --csv_path data/input/sample_input.csv --models Sentient gpt5 grok4 pplx
        """
    )
    
    # Input arguments
    parser.add_argument('--csv_path', type=str, 
                       default='data/input/sample_input.csv',
                       help='Path to the CSV file containing evaluation data')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['Sentient', 'gpt5', 'grok4', 'pplx'],
                       help='List of models to evaluate')
    parser.add_argument('--num_workers', type=int, default=3,
                       help='Number of parallel workers for processing queries')
    parser.add_argument('--max_queries', type=int, default=None,
                       help='Maximum number of queries to evaluate (None for all)')
    
    return parser.parse_args()

async def main():
    """Run complete LLM evaluation pipeline."""
    args = parse_arguments()
    
    # Initialize evaluator
    evaluator = LLMEvaluationSystem(
        csv_path=args.csv_path,
        models_to_evaluate=args.models,
        num_workers=args.num_workers
    )
    
    # Load data
    evaluator.load_data()
    
    # Step 1: Run evaluations
    logger.info("Starting evaluation of all queries...")
    await evaluator.evaluate_all_queries(args.max_queries)

    if not evaluator.evaluations:
        logger.error(
            "No evaluations completed. Check FIREWORKS_API_KEY in .env and that your "
            "--models names match CSV column prefixes exactly (e.g. Sentient → Sentient_response)."
        )
        raise SystemExit(1)
    
    # Step 2: Calculate tag metrics
    logger.info("Calculating tag metrics...")
    csv_output_path = evaluator.create_evaluation_results_csv()
    evaluator.calculate_tag_metrics(csv_output_path)
    
    # Step 3: Generate summary report
    logger.info("Generating summary report...")
    report = evaluator.generate_summary_report()
    print(report)
    
    # Step 4: Per-model statistics
    logger.info("Calculating per-model statistics...")
    model_stats = evaluator.calculate_per_model_statistics()
    for model_name, stats in model_stats.items():
        if 'overall' in stats:
            oc = stats['overall']
            print(f"\n{model_name}:")
            print(f"  Win Count: {oc['win_count']}/{oc['total_queries']}")
            print(f"  Average Rank: {oc['avg_rank']:.2f}")
            for metric_type in ['temporal_relevance', 'data_consistency', 'depth', 'relevance']:
                if metric_type in stats:
                    m = stats[metric_type]
                    print(f"  {metric_type.replace('_', ' ').title()}: Mean={m['mean']:.2f}, Std={m['std']:.2f}")

    # Step 5: Per-model error taxonomy
    logger.info("Running per-model error analysis...")
    error_analysis = await evaluator.perform_error_analysis_per_model(batch_size=50)

    # Step 6: Save final XLSX
    logger.info("Saving evaluation statistics to XLSX...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = os.path.join('data/output', f"evaluation_statistics_{timestamp}.xlsx")
    stats_path = evaluator.save_statistics_xlsx(output_path=xlsx_path, error_analysis=error_analysis)
    print(f"XLSX file saved to: {stats_path}")
    
    # Delete temporary CSV file (XLSX contains all the same data)
    if csv_output_path and os.path.exists(csv_output_path):
        os.remove(csv_output_path)

if __name__ == "__main__":
    asyncio.run(main())