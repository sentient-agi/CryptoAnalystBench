"""
Error taxonomy evaluation: score a single model's responses on 7 binary (0/1) error dimensions.
Uses Deepseek judge. Trace context via a single get_context function.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

from src.llms.judge import ChatJudgeLLM, parse_llm_json_response
from src.llms.llm import Fireworks_LLM

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRACE_SPAN_NAME = "Cyrpto Final Response"
OUTPUT_BASE = "output"

EVAL_DATETIME = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")

TAXONOMY_FIELDS = [
    "staleness_missing_time_bounds",
    "inconsistent_claims",
    "source_reconciliation_failure",
    "shallow_synthesis",
    "missing_risk_or_mechanism_context",
    "overconfident_prediction",
    "partial_or_misframed_answer",
]


def get_context(chat_id: str, model_name: str) -> str:
    """
    Load trace for model, find span TRACE_SPAN_NAME, return its input as context string.
    Single function for all trace-based context; returns "" if missing.
    """
    if not chat_id or not str(chat_id).strip():
        return ""
    trace_path = Path(OUTPUT_BASE) / model_name / "traces" / f"{str(chat_id).strip()}.json"
    if not trace_path.exists():
        return ""
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for trace in data.get("traces", []):
        for span in trace.get("spans", []):
            if span.get("name") != TRACE_SPAN_NAME:
                continue
            inp = span.get("attributes.input.value")
            if inp is None:
                return ""
            if isinstance(inp, dict) and "messages" in inp:
                messages = inp.get("messages", [])
                if not messages:
                    return ""
                content = messages[-1].get("content", "")
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if isinstance(inp, str):
                return inp
            return json.dumps(inp, ensure_ascii=False, indent=2)
    return ""


def build_prompt(query: str, response: str, trace_context: str = "") -> str:
    """Build user prompt for 7-parameter binary scoring (same as llm_single_model_taxonomy.py)."""
    context_block = ""
    if trace_context and trace_context.strip():
        context_block = f"""
CONTEXT THAT WAS AVAILABLE TO THE MODEL (evaluate factual accuracy and relevance against this context only; do not use your own knowledge):
---
{trace_context}
---
"""
    return f"""Evaluate the response for this query.

EVALUATION DATE AND TIME: {EVAL_DATETIME}

QUERY: {query}
{context_block}
RESPONSE: {response}

Score this response on the 7 parameters using BINARY scoring (0 or 1). For each parameter:
- 0 = the issue is NOT present (response is satisfactory on this dimension)
- 1 = the issue IS present (response has this problem)

Use the criteria below (meaning, typical symptoms, how to verify) and provide brief reasoning for each binary score.

EVALUATION CRITERIA (binary 0/1):

1. Staleness / missing time bounds
Meaning: The response uses outdated/futuristic/irrelevant information or doesn't specify when the information is valid.
Typical symptoms:
- "Bitcoin ETF was approved" (but approval happened years ago)
- "Prices are rising" (no date mentioned)
- Uses old stats as current facts
- Answers "today" questions with historical data
How to verify: (1) Extract all time-sensitive claims. (2) Ask: Does it mention date/timeframe? Is data current? (3) Cross-check with latest source.

2. Inconsistent claims
Meaning: The answer contradicts itself or gives mutually incompatible statements.
Typical symptoms:
- "Model A is best" → later "Model B performs highest"
- Two different numbers for same metric
- Says "approved" and "not approved"
How to verify: (1) Extract factual claims. (2) Normalize entities/numbers. (3) Check for conflicts.

3. Source reconciliation failure
Meaning: Multiple sources exist but the model: ignores conflicts; picks one randomly; doesn't explain discrepancies.
Typical symptoms:
- Two sources say different prices → model outputs only one
- No "according to source X…" clarification
- Doesn't mention uncertainty
How to verify: (1) Compare answer vs all provided sources. (2) Check: Were conflicting sources addressed? Was justification given?

4. Shallow synthesis
Meaning: Response just copies or paraphrases pieces without reasoning or combining insights.
Typical symptoms:
- Bullet list copy-paste
- No explanation or comparison
- No reasoning chain
How to verify: (1) Does it sound detailed with proper reasoning? (2) Ask: "Did it integrate info or just repeat?" "Can it be more detailed?"

5. Missing risk or mechanism context
Meaning: Gives answer but ignores risks, trade-offs, or underlying mechanism.
Typical symptoms:
- "Invest in X" without risk
- "Use this method" without limitations
- No explanation of how it works
How to verify: (1) Ask: Are risks mentioned? Is mechanism explained? (2) Compare to domain expectations.

6. Overconfident prediction
Meaning: Makes strong claims about uncertain future events without probability or hedging.
Typical symptoms:
- "Will definitely rise"
- "Guaranteed"
- No uncertainty qualifiers
- Any prediction claim with full confidence
How to verify: (1) Detect future/prediction language. (2) Check presence of uncertainty.

7. Partial or misframed answer
Meaning: Doesn't fully answer question or answers wrong interpretation.
Typical symptoms:
- Only answers first part of multi-part question
- Responds to different question
- Missing required fields
How to verify: (1) Decompose question into sub-parts. (2) Check coverage for each.

ADDITIONAL INSTRUCTIONS:
- Base each 0/1 on the type of query. If the query does not require time bounds, give parameter 1 a 0. If the query is narrow, do not score 1 for "shallow" or "missing risk" unless the ask implied more.
- Evaluate factual accuracy and relevance against the CONTEXT provided above only; do not use your own knowledge.

REASONING: For each parameter, provide a brief justification (why 0 or 1).

IMPORTANT: Respond with ONLY valid JSON in English. Do not include any text before or after the JSON. Do not use markdown formatting. All responses must be in English only.

Required JSON format:
{{
    "staleness_missing_time_bounds": {{ "score": <0 or 1>, "reasoning": "<brief explanation>" }},
    "inconsistent_claims": {{ "score": <0 or 1>, "reasoning": "<brief explanation>" }},
    "source_reconciliation_failure": {{ "score": <0 or 1>, "reasoning": "<brief explanation>" }},
    "shallow_synthesis": {{ "score": <0 or 1>, "reasoning": "<brief explanation>" }},
    "missing_risk_or_mechanism_context": {{ "score": <0 or 1>, "reasoning": "<brief explanation>" }},
    "overconfident_prediction": {{ "score": <0 or 1>, "reasoning": "<brief explanation>" }},
    "partial_or_misframed_answer": {{ "score": <0 or 1>, "reasoning": "<brief explanation>" }}
}}"""


class ErrorTaxonomyEvaluator:
    """Evaluate a single model on 7 binary error taxonomy parameters."""

    def __init__(self, csv_path: str, model_column: str, api_key: str = None):
        self.csv_path = csv_path
        self.model_column = model_column
        self.model_name = (
            model_column.replace("_response", "") if model_column.endswith("_response") else model_column
        )
        self.chat_id_col = None  # set in load_data if column exists
        self.df = None
        self.evaluations: List[Dict[str, Any]] = []
        llm = Fireworks_LLM(
            model_id="accounts/fireworks/models/deepseek-v3p1",
            api_key=api_key,
            temperature=0.1,
            max_tokens=4000,
        )
        self.judge = ChatJudgeLLM(llm, "You are an expert LLM evaluator for crypto/blockchain.")

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        if "query" not in self.df.columns or self.model_column not in self.df.columns:
            raise ValueError(f"CSV must have 'query' and '{self.model_column}'")
        candidate = f"{self.model_name}_chat_id"
        if candidate in self.df.columns:
            self.chat_id_col = candidate
        self.df = self.df[self.df[self.model_column].notna() & (self.df[self.model_column] != "")]
        logger.info(f"Loaded {len(self.df)} rows from {self.csv_path}")
        return self.df

    async def evaluate_one(self, row: pd.Series) -> Dict[str, Any]:
        query = row["query"]
        response = row[self.model_column]
        tags_val = row.get("tags")
        tags_str = "" if pd.isna(tags_val) or tags_val == "" else str(tags_val).strip()
        trace_context = ""
        if self.chat_id_col and pd.notna(row.get(self.chat_id_col)):
            cid = str(row[self.chat_id_col]).strip()
            if cid:
                trace_context = get_context(cid, self.model_name)
        prompt = build_prompt(query, response, trace_context)
        self.judge.clear_conversation_history()
        for attempt in range(3):
            try:
                result = await self.judge.a_generate(prompt)
                data = parse_llm_json_response(result)
                out = {"query": query, "tags": tags_str}
                for f in TAXONOMY_FIELDS:
                    if f not in data or "score" not in data[f] or "reasoning" not in data[f]:
                        raise ValueError(f"Missing field: {f}")
                    s = data[f]["score"]
                    if s not in (0, 1, 0.0, 1.0):
                        raise ValueError(f"Invalid score for {f}: {s}")
                    out[f] = int(s)
                    out[f"{f}_reasoning"] = data[f]["reasoning"]
                return out
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/3 failed: {e}")
                await asyncio.sleep(1)
        err = "Error after retries"
        out = {"query": query, "tags": tags_str}
        for f in TAXONOMY_FIELDS:
            out[f] = 1
            out[f"{f}_reasoning"] = err
        return out

    async def evaluate_all(self, max_queries: int = None, max_workers: int = 3) -> None:
        if self.df is None:
            self.load_data()
        subset = self.df.head(max_queries) if max_queries else self.df
        sem = asyncio.Semaphore(max_workers)

        async def run(idx: int, row: pd.Series):
            async with sem:
                r = await self.evaluate_one(row)
                logger.info(f"Done {idx + 1}/{len(subset)}")
                return r

        tasks = [run(i, row) for i, (_, row) in enumerate(subset.iterrows())]
        self.evaluations = await asyncio.gather(*tasks)

    def save_results(self, output_path: str = None) -> str:
        if not self.evaluations:
            raise ValueError("No evaluations. Run evaluate_all first.")
        output_path = output_path or f"data/output/error_taxonomy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.evaluations)
        with pd.ExcelWriter(output_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Evaluation Results", index=False)
            summary = pd.DataFrame({
                "Metric": TAXONOMY_FIELDS,
                "Proportion present": [df[c].mean() for c in TAXONOMY_FIELDS],
                "Count present": [df[c].sum() for c in TAXONOMY_FIELDS],
            })
            summary.to_excel(w, sheet_name="Summary", index=False)
        df.to_csv(Path(output_path).with_suffix(".csv"), index=False, encoding="utf-8")
        logger.info(f"Results saved to {output_path}")
        return output_path


async def main():
    parser = argparse.ArgumentParser(description="Error taxonomy evaluation (7 binary params)")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--model_column", type=str, required=True, help="Response column name (e.g. sentient_response)")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    ev = ErrorTaxonomyEvaluator(args.csv_path, args.model_column)
    ev.load_data()
    await ev.evaluate_all(max_queries=args.max_queries, max_workers=args.max_workers)
    path = ev.save_results(args.output)
    print(f"Done. Results: {path}")


if __name__ == "__main__":
    asyncio.run(main())
