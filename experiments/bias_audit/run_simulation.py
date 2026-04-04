"""
Module 2: LLM Simulation Runner

Sends synthetic personas + personality survey questions to Gemini and collects responses.
Supports checkpointing (resumes from where it left off) and configurable concurrency.

Usage:
    python run_simulation.py --model gemini-2.5-flash --n_workers 50
    python run_simulation.py --model gemini-3.0-flash --n_workers 30 --temperature 0.0
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path so we can import llm_helper
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "text_simulation"))

from dotenv import load_dotenv

from utils import (
    SYSTEM_MESSAGE,
    build_survey_prompt,
    compute_trait_scores,
    load_demographics_config,
    load_traits_config,
    parse_llm_response,
)

load_dotenv(PROJECT_ROOT / ".env")

# ============================================================================
# Logging setup — logs raw LLM outputs to outputs.log for debugging
# ============================================================================
_LOG_PATH = Path(__file__).parent / "results" / "outputs.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bias_audit")


# ============================================================================
# Gemini Direct Call (standalone, no dependency on llm_helper.py)
# ============================================================================
# We implement our own async Gemini caller to avoid tight coupling with the
# paper's llm_helper.py (which has Gemini SDK imports that may break if
# the SDK version changes). The logic is equivalent.
# ============================================================================

async def call_gemini(
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Call Gemini API and return the full response text.
    Uses google.generativeai SDK (0.x) with warnings suppressed.
    Reads from candidates[0].content.parts to avoid truncation by response.text shortcut.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import google.generativeai as genai
            from google.generativeai import types
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Set it in .env or pass --api_key"
        )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)

    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(
                f"{SYSTEM_MESSAGE}\n\n{prompt}",
                generation_config=types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            ),
        )

        # Extract full text from candidates to avoid truncation by response.text shortcut
        full_text = ""
        try:
            if response.candidates:
                parts = response.candidates[0].content.parts
                full_text = "".join(p.text for p in parts if hasattr(p, "text"))
        except Exception:
            pass  # Fall back to response.text

        if not full_text:
            try:
                full_text = response.text or ""
            except Exception:
                full_text = ""

        usage = {}
        try:
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }
        except Exception:
            pass

        if full_text:
            return {"response_text": full_text, "usage": usage, "error": None}
        else:
            return {"response_text": "", "usage": usage, "error": "Empty response from Gemini"}

    except Exception as e:
        return {"response_text": "", "usage": {}, "error": str(e)}


async def call_openai(
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Call OpenAI API and return the response text.
    Fallback provider if Gemini is unavailable.
    """
    try:
        import openai
        import httpx
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set.")

    try:
        async with httpx.AsyncClient(timeout=120.0) as http_client:
            client = openai.AsyncOpenAI(api_key=key, http_client=http_client)
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return {
                "response_text": response.choices[0].message.content,
                "usage": usage,
                "error": None,
            }
    except Exception as e:
        return {"response_text": "", "usage": {}, "error": str(e)}


# ============================================================================
# Main Simulation Logic
# ============================================================================

async def process_single_persona(
    persona_id: str,
    prompt: str,
    question_map: List[Tuple[str, int, str]],
    traits_config: dict,
    model_name: str,
    provider: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Dict:
    """Process a single persona: send prompt, parse response, compute scores."""
    async with semaphore:
        expected_count = len(question_map)
        last_error = None

        for attempt in range(max_retries):
            # Choose provider
            if provider == "gemini":
                result = await call_gemini(prompt, model_name, temperature)
            elif provider == "openai":
                result = await call_openai(prompt, model_name, temperature)
            else:
                return {"persona_id": persona_id, "error": f"Unknown provider: {provider}"}

            # --- Always log the raw response ---
            logger.info(
                "PERSONA=%s | ATTEMPT=%d | TOKENS=%s | ERROR=%s\n--- RAW RESPONSE ---\n%s\n--- END ---",
                persona_id,
                attempt + 1,
                result.get("usage", {}).get("total_tokens", "N/A"),
                result.get("error"),
                result.get("response_text", ""),
            )

            if result["error"]:
                last_error = result["error"]
                # Check for model-not-found to enable fallback
                if "not found" in result["error"].lower() or "404" in result["error"]:
                    return {
                        "persona_id": persona_id,
                        "error": f"MODEL_NOT_FOUND: {result['error']}",
                    }
                await asyncio.sleep(2 ** attempt)
                continue

            # Parse the response
            parsed = parse_llm_response(result["response_text"], expected_count)
            if parsed is None:
                last_error = f"Failed to parse response: {result['response_text'][:200]}"
                logger.warning(
                    "PERSONA=%s | ATTEMPT=%d | PARSE FAILED (got %d chars, expected ~%d Q answers)\nFULL TEXT: %s",
                    persona_id,
                    attempt + 1,
                    len(result["response_text"]),
                    expected_count,
                    result["response_text"],
                )
                await asyncio.sleep(2 ** attempt)
                continue

            logger.info("PERSONA=%s | ATTEMPT=%d | PARSE OK (%d/%d questions)", persona_id, attempt + 1, len(parsed), expected_count)

            # Compute trait scores
            scores = compute_trait_scores(parsed, question_map, traits_config)

            return {
                "persona_id": persona_id,
                "raw_responses": parsed,
                "trait_scores": scores,
                "usage": result["usage"],
                "error": None,
            }

        return {"persona_id": persona_id, "error": f"Failed after {max_retries} retries: {last_error}"}


async def run_simulation(
    personas_json_path: str,
    demographics_config: dict,
    traits_config: dict,
    model_name: str,
    provider: str = "gemini",
    n_workers: int = 50,
    temperature: float = 0.0,
    max_retries: int = 3,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run the full simulation for all personas.

    Supports checkpointing: skips personas that already have results.
    """
    # Load personas
    with open(personas_json_path, 'r') as f:
        persona_texts = json.load(f)

    # Set up output
    out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
    raw_dir = out_dir / "raw_responses"
    scores_dir = out_dir / "scores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Build question map (same for all personas)
    dummy_prompt, question_map = build_survey_prompt("", traits_config)

    # Check for existing results (checkpointing)
    existing = set()
    scores_file = scores_dir / "trait_scores.jsonl"
    if scores_file.exists():
        with open(scores_file, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing.add(rec["persona_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Filter to personas that still need processing
    to_process = {
        pid: text for pid, text in persona_texts.items()
        if pid not in existing
    }

    if not to_process:
        print("✅ All personas already processed (checkpointing). Nothing to do.")
        return

    if existing:
        print(f"📋 Resuming: {len(existing)} already done, {len(to_process)} remaining.")
    else:
        print(f"🚀 Processing {len(to_process)} personas with {model_name} ({provider})...")

    # Build prompts
    prompts = {}
    for pid, persona_text in to_process.items():
        prompt, _ = build_survey_prompt(persona_text, traits_config)
        prompts[pid] = prompt

    # Run async
    semaphore = asyncio.Semaphore(n_workers)
    tasks = [
        process_single_persona(
            pid, prompt, question_map, traits_config,
            model_name, provider, temperature, semaphore, max_retries
        )
        for pid, prompt in prompts.items()
    ]

    # Process with progress tracking
    success_count = 0
    fail_count = 0
    model_not_found = False

    # Open files for appending (checkpointing)
    with open(scores_file, 'a') as scores_f:
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro

            pid = result["persona_id"]

            if result.get("error"):
                if "MODEL_NOT_FOUND" in str(result["error"]):
                    model_not_found = True
                    print(f"\n❌ Model '{model_name}' not found. Aborting.")
                    print("   Try a different model with --model flag.")
                    # Cancel remaining tasks
                    for task in tasks:
                        if hasattr(task, 'cancel'):
                            task.cancel()
                    break

                fail_count += 1
                print(f"  ❌ {pid}: {result['error'][:100]}")
            else:
                success_count += 1

                # Save raw response
                raw_path = raw_dir / f"{pid}.json"
                with open(raw_path, 'w') as f:
                    json.dump({
                        "persona_id": pid,
                        "raw_responses": {str(k): v for k, v in result["raw_responses"].items()},
                        "usage": result["usage"],
                    }, f, indent=2)

                # Append trait scores to JSONL
                score_record = {
                    "persona_id": pid,
                    **result["trait_scores"]
                }
                scores_f.write(json.dumps(score_record) + "\n")
                scores_f.flush()

            # Progress
            total_done = success_count + fail_count
            if total_done % 50 == 0 or total_done == len(to_process):
                print(f"  📊 Progress: {total_done}/{len(to_process)} "
                      f"(✅ {success_count} | ❌ {fail_count})")

    if model_not_found:
        return

    print(f"\n{'=' * 50}")
    print(f"✅ Simulation complete!")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"   Results: {scores_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM simulation for bias audit."
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Model name (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--provider", type=str, default="gemini",
        choices=["gemini", "openai"],
        help="LLM provider (default: gemini)"
    )
    parser.add_argument(
        "--n_workers", type=int, default=50,
        help="Number of concurrent requests (default: 50)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Max retries per persona (default: 3)"
    )
    parser.add_argument(
        "--personas_json", type=str, default=None,
        help="Path to personas.json (default: results/personas.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--demographics_config", type=str, default=None,
        help="Path to demographics YAML config"
    )
    parser.add_argument(
        "--traits_config", type=str, default=None,
        help="Path to personality traits YAML config"
    )
    args = parser.parse_args()

    # Load configs
    demographics_config = load_demographics_config(args.demographics_config)
    traits_config = load_traits_config(args.traits_config)

    # Paths
    base_dir = Path(__file__).parent
    personas_json = args.personas_json or str(base_dir / "results" / "personas.json")
    output_dir = args.output_dir or str(base_dir / "results")

    if not Path(personas_json).exists():
        print(f"❌ Personas file not found: {personas_json}")
        print("   Run generate_personas.py first.")
        sys.exit(1)

    # Run
    asyncio.run(run_simulation(
        personas_json_path=personas_json,
        demographics_config=demographics_config,
        traits_config=traits_config,
        model_name=args.model,
        provider=args.provider,
        n_workers=args.n_workers,
        temperature=args.temperature,
        max_retries=args.max_retries,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    main()
