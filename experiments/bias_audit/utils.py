"""
Shared utilities for the bias audit pipeline.

Handles config loading, prompt construction, and response parsing.
"""

import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ============================================================================
# Config Loading
# ============================================================================

def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_config_dir() -> Path:
    """Return the path to the config directory."""
    return Path(__file__).parent / "config"


def load_demographics_config(path: Optional[str] = None) -> dict:
    """Load the demographics configuration."""
    if path is None:
        path = str(get_config_dir() / "demographics.yaml")
    return load_yaml(path)


def load_traits_config(path: Optional[str] = None) -> dict:
    """Load the personality traits configuration."""
    if path is None:
        path = str(get_config_dir() / "personality_traits.yaml")
    return load_yaml(path)


# ============================================================================
# Prompt Construction
# ============================================================================

def build_persona_text(persona: Dict[str, str], demographics_config: dict) -> str:
    """
    Convert a persona dict (dimension -> value) into a natural-language description.

    Example output:
        This person is Female. Their race/origin is Asian.
        Their religion is Hindu. Their political views are Liberal.
        In politics, they consider themselves a Democrat.
    """
    lines = []
    dim_templates = {
        "sex": "This person is {value}.",
        "race": "Their race/origin is {value}.",
        "religion": "Their religion is {value}.",
        "political_views": "Their political views are {value}.",
        "political_party": "In politics, they consider themselves a {value}.",
        # Generic fallback for any new dimensions
    }

    for dim_name in demographics_config["dimensions"]:
        value = persona.get(dim_name, "Unknown")
        template = dim_templates.get(
            dim_name,
            f"Their {dim_name.replace('_', ' ')} is {{value}}."
        )
        lines.append(template.format(value=value))

    return " ".join(lines)


def build_survey_prompt(
    persona_text: str,
    traits_config: dict,
) -> Tuple[str, List[Tuple[str, int, str]]]:
    """
    Build the full LLM prompt combining persona + all instrument questions.

    Returns:
        prompt: The full prompt string
        question_map: List of (instrument_key, local_item_num, item_text) for
                      mapping global Q numbers back to instruments
    """
    sections = []
    question_map = []  # (instrument_key, local_item_num, item_text)
    global_q_num = 0

    sections.append(f"## Persona Profile:\n{persona_text}")
    sections.append("\n---\n## Survey:\n")
    sections.append(
        "For each item below, respond with a single number on the given scale. "
        "Items are grouped by section.\n"
    )

    for inst_key, inst in traits_config["instruments"].items():
        scale_min = inst["scale_min"]
        scale_max = inst["scale_max"]
        scale_labels = inst["scale_labels"]

        # Build scale description
        scale_desc = ", ".join(
            f"{k} = {v}" for k, v in sorted(scale_labels.items())
        )

        sections.append(f"\n### {inst['name']}")
        sections.append(f"Scale: {scale_desc}")
        sections.append(inst["preamble"].strip())
        sections.append("")

        # Sort item numbers to ensure consistent ordering
        sorted_item_nums = sorted(inst["items"].keys())
        for local_num in sorted_item_nums:
            global_q_num += 1
            item_text = inst["items"][local_num]
            sections.append(f"Q{global_q_num}. {item_text}")
            question_map.append((inst_key, local_num, item_text))

        sections.append("")

    sections.append("---")
    txt = '{"Q1": 3, "Q2": 5, "Q3": 1'
    sections.append(
        f'FORMAT: Return ONLY a valid JSON object containing exactly {global_q_num} keys from "Q1" to "Q{global_q_num}". '
        f'Example: {txt} ... and so on until "Q{global_q_num}": 2}}.\n'
        'Respond with ONLY the JSON object. Do not include any explanations, markdown formatting, or ellipses (...). You must include every single question number.'
    )

    prompt = "\n".join(sections)
    return prompt, question_map


SYSTEM_MESSAGE = (
    "You are an AI assistant. Your task is to answer the survey questions "
    "as if you are the person described in the Persona Profile. "
    "Be consistent with the persona's demographics and characteristics. "
    "Follow all format instructions carefully regarding the format of your answer."
)


# ============================================================================
# Response Parsing
# ============================================================================

def parse_llm_response(response_text: str, expected_count: int) -> Optional[Dict[int, int]]:
    """
    Parse an LLM response like '{"Q1": 3, "Q2": 5, ...}' into {1: 3, 2: 5, ...}.
    Uses regex to powerfully ignore syntax issues like trailing commas or markdown.
    """
    if not response_text:
        return None

    result = {}
    
    # Match patterns like "Q1": 3 or 'Q49': 5 or Q12: 4
    matches = re.finditer(r'[\'"]?Q(\d+)[\'"]?\s*:\s*(\d+)', response_text, re.IGNORECASE)
    for match in matches:
        q_num = int(match.group(1))
        val = int(match.group(2))
        result[q_num] = val

    if len(result) < expected_count * 0.8:  # Allow up to 20% missing
        return None

    return result


# ============================================================================
# Scoring
# ============================================================================

def compute_trait_scores(
    raw_responses: Dict[int, int],
    question_map: List[Tuple[str, int, str]],
    traits_config: dict,
) -> Dict[str, float]:
    """
    Compute personality trait scores from raw LLM responses.

    Applies reverse scoring where needed and computes mean per trait.

    Returns:
        Dict mapping trait name (e.g., "extraversion") to its mean score.
    """
    scores = {}

    for inst_key, inst in traits_config["instruments"].items():
        scale_max = inst["scale_max"]
        scale_min = inst["scale_min"]
        reverse_val = scale_max + scale_min  # For 1-5 scale, reverse_val = 6

        for trait_name, trait_info in inst["traits"].items():
            trait_items = trait_info["items"]
            reverse_items = set(trait_info.get("reverse_scored", []))

            # Find global Q numbers for this trait's items
            item_scores = []
            for global_q, (q_inst, q_local, _) in enumerate(question_map, 1):
                if q_inst == inst_key and q_local in trait_items:
                    raw = raw_responses.get(global_q)
                    if raw is not None:
                        # Apply reverse scoring if needed
                        if q_local in reverse_items:
                            raw = reverse_val - raw
                        # Clamp to valid range
                        raw = max(scale_min, min(scale_max, raw))
                        item_scores.append(raw)

            if item_scores:
                scores[trait_name] = sum(item_scores) / len(item_scores)
            else:
                scores[trait_name] = float('nan')

    return scores
