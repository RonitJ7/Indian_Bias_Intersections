"""
Top-level orchestrator for the bias audit experiment.

Chains together persona generation, LLM simulation, and regression analysis.
Each stage can also be run independently.

Usage:
    # Full pipeline
    python run_experiment.py --n_personas 1000 --seed 42 --model gemini-2.5-flash

    # Individual stages
    python run_experiment.py --stage generate --n_personas 1000 --seed 42
    python run_experiment.py --stage simulate --model gemini-2.5-flash --n_workers 50
    python run_experiment.py --stage analyze --interactions "sex:race,race:religion"
"""

import argparse
import asyncio
import sys
from pathlib import Path

from generate_personas import generate_personas, save_personas_csv, save_personas_json, print_distribution_summary
from utils import load_demographics_config, load_traits_config


def stage_generate(args):
    """Stage 1: Generate synthetic personas."""
    print("=" * 60)
    print("STAGE 1: Generate Synthetic Personas")
    print("=" * 60)

    config = load_demographics_config(args.demographics_config)
    output_dir = Path(args.output_dir)

    personas = generate_personas(args.n_personas, config, seed=args.seed)
    save_personas_csv(personas, output_dir / "personas.csv")
    save_personas_json(personas, config, output_dir / "personas.json")
    print_distribution_summary(personas, config)

    print(f"\n✅ Stage 1 complete: {args.n_personas} personas generated.\n")


def stage_simulate(args):
    """Stage 2: Run LLM simulation."""
    print("=" * 60)
    print("STAGE 2: Run LLM Simulation")
    print("=" * 60)

    # Import here to avoid loading heavy deps when not needed
    from run_simulation import run_simulation

    demographics_config = load_demographics_config(args.demographics_config)
    traits_config = load_traits_config(args.traits_config)

    personas_json = str(Path(args.output_dir) / "personas.json")
    if not Path(personas_json).exists():
        print(f"❌ No personas found at {personas_json}. Run stage 'generate' first.")
        sys.exit(1)

    # Try primary model, fall back if not found
    model = args.model
    provider = args.provider
    fallback_model = args.fallback_model

    print(f"   Model: {model} (fallback: {fallback_model})")
    print(f"   Provider: {provider}")
    print(f"   Workers: {args.n_workers}")
    print(f"   Temperature: {args.temperature}")
    print()

    asyncio.run(run_simulation(
        personas_json_path=personas_json,
        demographics_config=demographics_config,
        traits_config=traits_config,
        model_name=model,
        provider=provider,
        n_workers=args.n_workers,
        temperature=args.temperature,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
    ))

    # Check if model-not-found occurred and try fallback
    scores_file = Path(args.output_dir) / "scores" / "trait_scores.jsonl"
    if not scores_file.exists() or scores_file.stat().st_size == 0:
        if fallback_model and fallback_model != model:
            print(f"\n🔄 Primary model may have failed. Trying fallback: {fallback_model}")
            asyncio.run(run_simulation(
                personas_json_path=personas_json,
                demographics_config=demographics_config,
                traits_config=traits_config,
                model_name=fallback_model,
                provider=provider,
                n_workers=args.n_workers,
                temperature=args.temperature,
                max_retries=args.max_retries,
                output_dir=args.output_dir,
            ))

    print(f"\n✅ Stage 2 complete.\n")


def stage_analyze(args):
    """Stage 3: Run regression analysis."""
    print("=" * 60)
    print("STAGE 3: Regression Analysis")
    print("=" * 60)

    from analyze_results import run_analysis

    run_analysis(
        results_dir=args.output_dir,
        interactions_str=args.interactions,
    )

    print(f"\n✅ Stage 3 complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bias Audit Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with Gemini
  python run_experiment.py --n_personas 1000 --model gemini-2.5-flash

  # Just generate personas
  python run_experiment.py --stage generate --n_personas 1000

  # Just run simulation (personas must exist)
  python run_experiment.py --stage simulate --model gemini-2.5-flash

  # Just run analysis (scores must exist)
  python run_experiment.py --stage analyze --interactions "sex:race,race:religion"

  # Smoke test (5 personas)
  python run_experiment.py --n_personas 5 --stage all
        """
    )

    # Stage control
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "generate", "simulate", "analyze"],
        help="Which stage to run (default: all)"
    )

    # Generation args
    parser.add_argument("--n_personas", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Simulation args
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--fallback_model", type=str, default="gemini-3.0-flash",
                       help="Fallback model if primary is unavailable")
    parser.add_argument("--provider", type=str, default="gemini",
                       choices=["gemini", "openai"])
    parser.add_argument("--n_workers", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)

    # Analysis args
    parser.add_argument("--interactions", type=str, default="sex:race,race:religion")

    # Config paths
    parser.add_argument("--demographics_config", type=str, default=None)
    parser.add_argument("--traits_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / "results")

    print()
    print("🔬 BIAS AUDIT EXPERIMENT")
    print("=" * 60)
    print(f"   Stage:      {args.stage}")
    print(f"   N personas: {args.n_personas}")
    print(f"   Model:      {args.model} (fallback: {args.fallback_model})")
    print(f"   Provider:   {args.provider}")
    print(f"   Seed:       {args.seed}")
    print(f"   Output:     {args.output_dir}")
    print(f"   Interactions: {args.interactions}")
    print("=" * 60)
    print()

    stages = {
        "generate": stage_generate,
        "simulate": stage_simulate,
        "analyze": stage_analyze,
    }

    if args.stage == "all":
        for stage_name in ["generate", "simulate", "analyze"]:
            stages[stage_name](args)
    else:
        stages[args.stage](args)

    print("🎉 Experiment pipeline complete!")


if __name__ == "__main__":
    main()
