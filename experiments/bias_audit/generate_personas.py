"""
Module 1: Synthetic Persona Generator

Generates N personas by uniformly sampling one value per demographic dimension.
Outputs both a CSV (for regression) and a JSON (for LLM prompting).

Usage:
    python generate_personas.py --n 1000 --seed 42
    python generate_personas.py --n 1000 --seed 42 --demographics_config config/custom.yaml
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

from utils import load_demographics_config, build_persona_text


def generate_personas(
    n: int,
    demographics_config: dict,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generate N synthetic personas by uniformly sampling each dimension.

    Args:
        n: Number of personas to generate
        demographics_config: Loaded demographics YAML config
        seed: Random seed for reproducibility

    Returns:
        List of persona dicts, each mapping dimension_name -> sampled_value
    """
    rng = random.Random(seed)
    dimensions = demographics_config["dimensions"]
    personas = []

    for i in range(n):
        persona = {"persona_id": f"synth_{i:04d}"}
        for dim_name, dim_info in dimensions.items():
            persona[dim_name] = rng.choice(dim_info["values"])
        personas.append(persona)

    return personas


def save_personas_csv(personas: List[Dict[str, str]], output_path: Path) -> None:
    """Save personas as a CSV file (one row per persona, columns = dimensions)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not personas:
        print("Warning: No personas to save.")
        return

    fieldnames = list(personas[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(personas)

    print(f"✅ Saved {len(personas)} personas to {output_path}")


def save_personas_json(
    personas: List[Dict[str, str]],
    demographics_config: dict,
    output_path: Path,
) -> None:
    """Save personas as a JSON file mapping persona_id -> persona_text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    persona_texts = {}
    for persona in personas:
        pid = persona["persona_id"]
        persona_texts[pid] = build_persona_text(persona, demographics_config)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(persona_texts, f, indent=2)

    print(f"✅ Saved {len(persona_texts)} persona texts to {output_path}")


def print_distribution_summary(
    personas: List[Dict[str, str]],
    demographics_config: dict,
) -> None:
    """Print a summary of the distribution of values across dimensions."""
    dimensions = demographics_config["dimensions"]
    print(f"\n📊 Distribution summary (N={len(personas)}):")
    print("-" * 50)

    for dim_name, dim_info in dimensions.items():
        counts = {}
        for persona in personas:
            val = persona.get(dim_name, "MISSING")
            counts[val] = counts.get(val, 0) + 1

        expected = len(personas) / len(dim_info["values"])
        print(f"\n  {dim_name} (expected ~{expected:.0f} each):")
        for val in dim_info["values"]:
            count = counts.get(val, 0)
            pct = count / len(personas) * 100
            print(f"    {val}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic personas for bias audit."
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Number of personas to generate (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--demographics_config", type=str, default=None,
        help="Path to demographics YAML config (default: config/demographics.yaml)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: results/)"
    )
    args = parser.parse_args()

    # Load config
    config = load_demographics_config(args.demographics_config)

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"

    # Generate personas
    print(f"🎲 Generating {args.n} synthetic personas (seed={args.seed})...")
    personas = generate_personas(args.n, config, seed=args.seed)

    # Save outputs
    save_personas_csv(personas, output_dir / "personas.csv")
    save_personas_json(personas, config, output_dir / "personas.json")

    # Print distribution
    print_distribution_summary(personas, config)

    print(f"\n✅ Done! Generated {args.n} personas.")


if __name__ == "__main__":
    main()
