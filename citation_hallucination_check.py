"""
Combined citation and hallucination check: process trace JSONs and generate XLSX with two sheets.

1. Read input CSV file (same as script.py): columns {model_name}_chat_id identify models.
2. For each row and model, load trace from output/<model>/traces/<chat_id>.json using chat_id from CSV.
3. Find span with name == "Cyrpto Final Response" and get context/response for verification.
4. Run both citation-check and hallucination-check evals.
5. Generate final XLSX file with two sheets:
   - Sheet 1: Citation results
   - Sheet 2: Hallucination results
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.llms.judge import extract_json_from_response
from src.prompts.mapping import EVAL_PROMPT_MAPPING

load_dotenv()

SPAN_NAME = "Cyrpto Final Response"
MODEL = "accounts/fireworks/models/deepseek-v3p1"
TEMPERATURE = 0.1
BASE_URL = "https://api.fireworks.ai/inference/v1"

CITATION_OUTPUT_COLUMNS = [
    "model_name",
    "session_id",
    "trace_id",
    "context.span_id",
    "span_name",
    "span_kind",
    "parent_id",
    "response",
    "eval_result",
    "chain",
    "total_claims_identified",
    "claims_with_citations",
    "correctly_cited_direct",
    "correctly_cited_derivable",
    "incorrectly_cited",
    "missing_citations",
    "no_citation_needed",
    "citation_precision_percentage",
    "citation_completeness_percentage",
    "is_valid",
    "formula_check",
    "missing_citations_details",
    "label",
    "justification",
]

HALLUCINATION_OUTPUT_COLUMNS = [
    "model_name",
    "session_id",
    "trace_id",
    "context.span_id",
    "span_name",
    "span_kind",
    "parent_id",
    "response",
    "eval_result",
    "chain",
    "total_claims_identified",
    "verified_exact",
    "verified_derivable",
    "fabricated",
    "verification_rate_percentage",
    "label",
    "justification",
]


def _parse_citation_eval_result(eval_result: str) -> dict:
    """
    Split eval_result into chain (text before JSON block) and parse citation JSON.
    Return dict with citation-specific keys.
    """
    empty = {
        "chain": "",
        "total_claims_identified": "",
        "claims_with_citations": "",
        "correctly_cited_direct": "",
        "correctly_cited_derivable": "",
        "incorrectly_cited": "",
        "missing_citations": "",
        "no_citation_needed": "",
        "citation_precision_percentage": "",
        "citation_completeness_percentage": "",
        "is_valid": "",
        "formula_check": "",
        "missing_citations_details": "",
        "label": "",
        "justification": "",
    }
    if not (eval_result and isinstance(eval_result, str)):
        return empty

    s = eval_result.strip()
    try:
        parsed_json, json_start = extract_json_from_response(
            s, required_keys=("summary", "label")
        )
        chain = s[:json_start].strip() if json_start >= 0 else ""
    except ValueError:
        return empty

    summary = parsed_json.get("summary") or {}
    metrics = parsed_json.get("metrics") or {}
    verification = parsed_json.get("verification") or {}
    missing_details = parsed_json.get("missing_citations_details")
    return {
        "chain": chain,
        "total_claims_identified": summary.get("total_claims_identified", ""),
        "claims_with_citations": summary.get("claims_with_citations", ""),
        "correctly_cited_direct": summary.get("correctly_cited_direct", ""),
        "correctly_cited_derivable": summary.get("correctly_cited_derivable", ""),
        "incorrectly_cited": summary.get("incorrectly_cited", ""),
        "missing_citations": summary.get("missing_citations", ""),
        "no_citation_needed": summary.get("no_citation_needed", ""),
        "citation_precision_percentage": metrics.get("citation_precision_percentage", ""),
        "citation_completeness_percentage": metrics.get("citation_completeness_percentage", ""),
        "is_valid": verification.get("is_valid", ""),
        "formula_check": verification.get("formula_check", ""),
        "missing_citations_details": json.dumps(missing_details, ensure_ascii=False) if missing_details is not None else "",
        "label": parsed_json.get("label", ""),
        "justification": parsed_json.get("justification", ""),
    }


def _parse_hallucination_eval_result(eval_result: str) -> dict:
    """
    Split eval_result into chain (text before JSON block) and parse hallucination JSON.
    Return dict with hallucination-specific keys.
    """
    empty = {
        "chain": "",
        "total_claims_identified": "",
        "verified_exact": "",
        "verified_derivable": "",
        "fabricated": "",
        "verification_rate_percentage": "",
        "label": "",
        "justification": "",
    }
    if not (eval_result and isinstance(eval_result, str)):
        return empty

    s = eval_result.strip()
    try:
        parsed_json, json_start = extract_json_from_response(
            s, required_keys=("summary", "label")
        )
        chain = s[:json_start].strip() if json_start >= 0 else ""
    except ValueError:
        return empty

    summary = parsed_json.get("summary") or {}
    metrics = parsed_json.get("metrics") or {}
    return {
        "chain": chain,
        "total_claims_identified": summary.get("total_claims_identified", ""),
        "verified_exact": summary.get("verified_exact", ""),
        "verified_derivable": summary.get("verified_derivable", ""),
        "fabricated": summary.get("fabricated", ""),
        "verification_rate_percentage": metrics.get("verification_rate_percentage", ""),
        "label": parsed_json.get("label", ""),
        "justification": parsed_json.get("justification", ""),
    }


def _build_citation_row(row: dict, parsed: dict, model_name: str = "") -> dict:
    """Build citation output row: base fields + eval_result (full) + parsed columns."""
    span_id = row.get("context.span_id", row.get("span_id", ""))
    base = {
        "model_name": model_name,
        "session_id": row.get("session_id", ""),
        "trace_id": row.get("trace_id", ""),
        "context.span_id": span_id,
        "span_name": row.get("span_name", ""),
        "span_kind": row.get("span_kind", ""),
        "parent_id": row.get("parent_id", ""),
        "response": row.get("response", ""),
        "eval_result": row.get("eval_result", ""),
    }
    return {c: base.get(c, parsed.get(c, "")) for c in CITATION_OUTPUT_COLUMNS}


def _build_hallucination_row(row: dict, parsed: dict, model_name: str = "") -> dict:
    """Build hallucination output row: base fields + eval_result (full) + parsed columns."""
    span_id = row.get("context.span_id", row.get("span_id", ""))
    base = {
        "model_name": model_name,
        "session_id": row.get("session_id", ""),
        "trace_id": row.get("trace_id", ""),
        "context.span_id": span_id,
        "span_name": row.get("span_name", ""),
        "span_kind": row.get("span_kind", ""),
        "parent_id": row.get("parent_id", ""),
        "response": row.get("response", ""),
        "eval_result": row.get("eval_result", ""),
    }
    return {c: base.get(c, parsed.get(c, "")) for c in HALLUCINATION_OUTPUT_COLUMNS}


def get_context(span: dict) -> str:
    """
    Get comprehensive context details from trace for verification.
    
    Extracts context from the span's input value, including all relevant information
    needed for citation and hallucination verification.
    
    Args:
        span: Span dictionary containing attributes and metadata.
        
    Returns:
        Context string extracted from span's input value (messages or direct content).
    """
    inp = span.get("attributes.input.value")
    if inp is None:
        return ""
    
    # Handle messages format (most common case)
    if isinstance(inp, dict) and "messages" in inp:
        messages = inp["messages"]
        if not messages:
            return ""
        # Get the last message content
        msg = messages[-1]
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    
    # Handle direct string content
    if isinstance(inp, str):
        return inp
    
    # Handle other formats (dict, list, etc.) - convert to JSON string
    return json.dumps(inp, ensure_ascii=False, indent=2)


def _output_to_response(value) -> str:
    """<response> = attributes.output.value["choices"][0]["message"]["content"]."""
    if value is None:
        return ""
    if isinstance(value, dict) and "choices" in value:
        choices = value.get("choices", [])
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content is not None:
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _span_to_flat_row(session_id: str, trace_id: str, span: dict) -> dict:
    """Build a flat row with all span/input fields plus context, response, eval_result."""
    out = span.get("attributes.output.value")
    context = get_context(span)
    response = _output_to_response(out)
    row = {
        "session_id": session_id,
        "trace_id": trace_id,
        "context.span_id": span.get("context.span_id", ""),
        "span_name": span.get("name", ""),
        "span_kind": span.get("span_kind", ""),
        "parent_id": span.get("parent_id", ""),
        "context": context,
        "response": response,
        "eval_result": None,
    }
    for k, v in span.items():
        if k in row:
            continue
        if isinstance(v, (dict, list)):
            row[k] = json.dumps(v, ensure_ascii=False)
        else:
            row[k] = v
    return row


def get_models_and_chat_id_columns(csv_path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Read CSV and discover model -> chat_id column mapping (same as script.py).
    Columns named {model_name}_chat_id define models; model name is the prefix before _chat_id.

    Returns:
        (dataframe, model_to_chat_id_column) e.g. {"Sentient": "Sentient_chat_id", ...}
    """
    df = pd.read_csv(csv_path)
    model_to_chat_id_col = {}
    for col in df.columns:
        if col.endswith("_chat_id"):
            model_name = col[: -len("_chat_id")]
            if model_name:
                model_to_chat_id_col[model_name] = col
    return df, model_to_chat_id_col


def load_trace_row(output_root: str, model_name: str, chat_id: str) -> dict | None:
    """
    Load trace from output_root/<model_name>/traces/<chat_id>.json, find span "Cyrpto Final Response",
    and return a flat row dict with context and response (same pattern as script.py get_context).

    Returns:
        Row dict with session_id, trace_id, context.span_id, context, response, etc., or None if missing.
    """
    trace_path = Path(output_root) / model_name / "traces" / f"{chat_id}.json"
    if not trace_path.exists():
        return None
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    session_id = data.get("session_id", str(chat_id))
    for trace in data.get("traces", []):
        trace_id = trace.get("trace_id", "")
        for span in trace.get("spans", []):
            if span.get("name") != SPAN_NAME:
                continue
            return _span_to_flat_row(session_id, trace_id, span)
    return None


def _map_variables_to_values(variables: list[str], context: str, response: str) -> dict:
    """Map variable names from mapping to actual values."""
    var_kwargs = {}
    for var in variables:
        var_name = var.strip("<>")
        if var_name == "context":
            var_kwargs[var_name] = context
        elif var_name == "response":
            var_kwargs[var_name] = response
    return var_kwargs


def run_eval(client: OpenAI, user_prompt: str, system_prompt: str, variables: list[str], **kwargs) -> str:
    """
    Fill variables in user_prompt and call LLM; return raw eval text.
    
    Args:
        client: OpenAI client instance.
        user_prompt: User prompt template with variable placeholders.
        system_prompt: System prompt.
        variables: List of variable names (e.g., ["<response>", "<context>"]).
        **kwargs: Values for variables (e.g., response="...", context="...").
        
    Returns:
        Raw eval text from LLM.
    """
    filled = user_prompt
    for var in variables:
        var_name = var.strip("<>")
        if var_name in kwargs:
            filled = filled.replace(var, str(kwargs[var_name]))
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filled},
        ],
        temperature=TEMPERATURE,
        max_tokens=32768,
    )
    return (resp.choices[0].message.content or "").strip()


def process_traces_and_generate_xlsx(
    csv_path: str,
    output_xlsx: str,
    output_root: str = "output",
    models_filter: list[str] | None = None,
) -> None:
    """
    Process traces from CSV, run both citation and hallucination checks, generate XLSX with two sheets.

    Args:
        csv_path: Path to input CSV (used to get model -> session_ids mapping).
        output_xlsx: Path to output XLSX file.
        output_root: Root directory (default output); expects <output_root>/<model>/traces/.
        models_filter: If set, only process these model names.
    """
    df, model_to_chat_id_col = get_models_and_chat_id_columns(csv_path)
    if not model_to_chat_id_col:
        print("No _chat_id columns found in CSV.")
        return
    if models_filter is not None:
        model_to_chat_id_col = {m: col for m, col in model_to_chat_id_col.items() if m in models_filter}
        if not model_to_chat_id_col:
            print("No models match filter.")
            return

    citation_config = EVAL_PROMPT_MAPPING["citation_check"]
    citation_system_prompt = citation_config["system_prompt"]
    citation_user_prompt = citation_config["user_prompt"]
    citation_variables = citation_config["variables"]

    hallucination_config = EVAL_PROMPT_MAPPING["hallucination_check"]
    hallucination_system_prompt = hallucination_config["system_prompt"]
    hallucination_user_prompt = hallucination_config["user_prompt"]
    hallucination_variables = hallucination_config["variables"]

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY required")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    citation_rows = []
    hallucination_rows = []

    for model_name, chat_id_col in model_to_chat_id_col.items():
        for _, row in df.iterrows():
            chat_id = row.get(chat_id_col)
            if pd.isna(chat_id) or not str(chat_id).strip():
                continue
            chat_id = str(chat_id).strip()
            trace_row = load_trace_row(output_root, model_name, chat_id)
            if trace_row is None:
                continue

            # Run citation check
            citation_row = trace_row.copy()
            var_kwargs = _map_variables_to_values(
                citation_variables, citation_row["context"], citation_row["response"]
            )
            citation_row["eval_result"] = run_eval(
                client,
                citation_user_prompt,
                citation_system_prompt,
                citation_variables,
                **var_kwargs,
            )
            parsed_citation = _parse_citation_eval_result(citation_row.get("eval_result") or "")
            final_citation_row = _build_citation_row(citation_row, parsed_citation, model_name)
            citation_rows.append(final_citation_row)

            # Run hallucination check
            hallucination_row = trace_row.copy()
            var_kwargs = _map_variables_to_values(
                hallucination_variables, hallucination_row["context"], hallucination_row["response"]
            )
            hallucination_row["eval_result"] = run_eval(
                client,
                hallucination_user_prompt,
                hallucination_system_prompt,
                hallucination_variables,
                **var_kwargs,
            )
            parsed_hallucination = _parse_hallucination_eval_result(hallucination_row.get("eval_result") or "")
            final_hallucination_row = _build_hallucination_row(hallucination_row, parsed_hallucination, model_name)
            hallucination_rows.append(final_hallucination_row)

        print(f"Processed model: {model_name}")

    # Create DataFrames
    citation_df = pd.DataFrame(citation_rows)
    hallucination_df = pd.DataFrame(hallucination_rows)

    # Write to XLSX with two sheets
    output_path = Path(output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        citation_df.to_excel(writer, sheet_name="Citation", index=False)
        hallucination_df.to_excel(writer, sheet_name="Hallucination", index=False)

    print(f"Done. Output: {output_xlsx}")
    print(f"  Citation rows: {len(citation_rows)}")
    print(f"  Hallucination rows: {len(hallucination_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run combined citation and hallucination checks on traces from CSV and generate XLSX with two sheets."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        metavar="PATH",
        help="Input CSV with {model_name}_chat_id columns. Reads traces from output/<model>/traces/<chat_id>.json.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        required=True,
        metavar="PATH",
        help="Output XLSX file path (will contain two sheets: Citation and Hallucination).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Output root directory (default: output). Expects <output_root>/<model>/traces/.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to process. Default: all from CSV.",
    )
    args = parser.parse_args()

    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    process_traces_and_generate_xlsx(
        csv_path=args.csv,
        output_xlsx=args.output_xlsx,
        output_root=args.output_root,
        models_filter=models_filter,
    )


if __name__ == "__main__":
    main()
