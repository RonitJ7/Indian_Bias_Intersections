"""
Top-level orchestrator for the bias audit experiment.

Chains together persona generation, LLM simulation, and regression analysis.
Each stage can also be run independently.

Configuration is loaded from config/experiment.yaml first. Any CLI flags
you pass will OVERRIDE the values in that file.

Usage:
    # Full pipeline using config/experiment.yaml
    python run_experiment.py

    # Override specific values on the fly
    python run_experiment.py --n_personas 5 --stage generate
    python run_experiment.py --stage simulate --model gemini-2.0-flash
    python run_experiment.py --stage analyze
"""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from generate_personas import generate_personas, save_personas_csv, save_personas_json, print_distribution_summary
from utils import load_demographics_config, load_traits_config


# ============================================================================
# Config Loading
# ============================================================================

_BIAS_AUDIT_DIR = Path(__file__).parent
_DEFAULT_EXPERIMENT_CONFIG = _BIAS_AUDIT_DIR / "config" / "experiment.yaml"


def load_experiment_config(path: Path) -> dict:
    """Load experiment.yaml, returning an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_args(parsed_cli, experiment_cfg: dict):
    """
    Merge experiment.yaml values with CLI args.
    CLI args always win; experiment.yaml provides the defaults.
    """
    bias_audit_dir = _BIAS_AUDIT_DIR

    def _resolve(cli_val, cfg_key, fallback):
        return cli_val if cli_val is not None else experiment_cfg.get(cfg_key, fallback)

    args = argparse.Namespace()

    # Stage
    args.stage = parsed_cli.stage or experiment_cfg.get("stage", "all")

    # Generation
    args.n_personas = _resolve(parsed_cli.n_personas, "n_personas", 1000)
    args.seed = _resolve(parsed_cli.seed, "seed", 42)

    # Simulation
    args.model = _resolve(parsed_cli.model, "model", "gemini-2.5-flash")
    args.fallback_model = _resolve(parsed_cli.fallback_model, "fallback_model", "gemini-2.0-flash")
    args.provider = _resolve(parsed_cli.provider, "provider", "gemini")
    args.n_workers = _resolve(parsed_cli.n_workers, "n_workers", 50)
    args.temperature = _resolve(parsed_cli.temperature, "temperature", 0.0)
    args.max_retries = _resolve(parsed_cli.max_retries, "max_retries", 3)

    # Analysis — reference categories and regression output dir come from experiment.yaml
    args.reference_categories = experiment_cfg.get("reference_categories", None)
    args.regression_dir = experiment_cfg.get("regression_dir", "regression")

    # Paths — resolve relative paths against the bias_audit directory
    raw_output = _resolve(parsed_cli.output_dir, "output_dir", "results")
    args.output_dir = str(
        (bias_audit_dir / raw_output).resolve()
        if not Path(raw_output).is_absolute()
        else Path(raw_output)
    )

    raw_demo = _resolve(parsed_cli.demographics_config, "demographics_config",
                        "config/demographics.yaml")
    args.demographics_config = str(
        (bias_audit_dir / raw_demo).resolve()
        if not Path(raw_demo).is_absolute()
        else Path(raw_demo)
    )

    raw_traits = _resolve(parsed_cli.traits_config, "traits_config",
                          "config/personality_traits.yaml")
    args.traits_config = str(
        (bias_audit_dir / raw_traits).resolve()
        if not Path(raw_traits).is_absolute()
        else Path(raw_traits)
    )

    return args


# ============================================================================
# Stages
# ============================================================================

def stage_generate(args, demographics_config):
    """Stage 1: Generate synthetic personas."""
    print("=" * 60)
    print("STAGE 1: Generate Synthetic Personas")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    personas = generate_personas(args.n_personas, demographics_config, seed=args.seed)
    save_personas_csv(personas, output_dir / "personas.csv")
    save_personas_json(personas, demographics_config, output_dir / "personas.json")
    print_distribution_summary(personas, demographics_config)

    print(f"\n✅ Stage 1 complete: {args.n_personas} personas generated.\n")


def stage_simulate(args, demographics_config, traits_config):
    """Stage 2: Run LLM simulation."""
    print("=" * 60)
    print("STAGE 2: Run LLM Simulation")
    print("=" * 60)

    from run_simulation import run_simulation

    personas_json = str(Path(args.output_dir) / "personas.json")
    if not Path(personas_json).exists():
        print(f"❌ No personas found at {personas_json}. Run stage 'generate' first.")
        sys.exit(1)

    model = args.model
    fallback_model = args.fallback_model
    provider = args.provider

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

    # Fallback if primary produced nothing
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


def stage_analyze(args, demographics_config):
    """Stage 3: Run regression analysis."""
    print("=" * 60)
    print("STAGE 3: Regression Analysis")
    print("=" * 60)

    from analyze_results import run_analysis

    # Build interactions string from demographics.yaml interactions block
    raw_interactions = demographics_config.get("interactions", [])
    interactions_str = ",".join(f"{a}:{b}" for a, b in raw_interactions)

    run_analysis(
        results_dir=args.output_dir,
        interactions_str=interactions_str,
        reference_categories=args.reference_categories,
        regression_dir=args.regression_dir,
    )

    print(f"\n✅ Stage 3 complete.\n")


# ============================================================================
# Main
# ============================================================================

def main():
    # Load experiment config first
    experiment_cfg = load_experiment_config(_DEFAULT_EXPERIMENT_CONFIG)

    parser = argparse.ArgumentParser(
        description="Bias Audit Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline using config/experiment.yaml
  python run_experiment.py

  # Override specific values
  python run_experiment.py --n_personas 5 --stage generate
  python run_experiment.py --stage simulate --model gemini-2.0-flash
  python run_experiment.py --stage analyze

  # Point to a different experiment config entirely
  python run_experiment.py --experiment_config my_other_config.yaml
        """
    )

    # All args are optional — defaults come from experiment.yaml
    parser.add_argument("--stage", type=str, default=None,
                        choices=["all", "generate", "simulate", "analyze"])
    parser.add_argument("--n_personas", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--fallback_model", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None,
                        choices=["gemini", "openai"])
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_retries", type=int, default=None)
    parser.add_argument("--demographics_config", type=str, default=None)
    parser.add_argument("--traits_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_config", type=str, default=None,
                        help="Path to a custom experiment.yaml (default: config/experiment.yaml)")

    parsed_cli = parser.parse_args()

    # Allow pointing to a different experiment config at runtime
    if parsed_cli.experiment_config:
        experiment_cfg = load_experiment_config(Path(parsed_cli.experiment_config))

    args = build_args(parsed_cli, experiment_cfg)

    # Load shared configs once (reused across stages)
    demographics_config = load_demographics_config(args.demographics_config)
    traits_config = load_traits_config(args.traits_config)

    # Banner
    interactions_display = ",".join(
        f"{a}:{b}" for a, b in demographics_config.get("interactions", [])
    )
    print()
    print("🔬 BIAS AUDIT EXPERIMENT")
    print("=" * 60)
    print(f"   Stage:      {args.stage}")
    print(f"   N personas: {args.n_personas}")
    print(f"   Model:      {args.model} (fallback: {args.fallback_model})")
    print(f"   Provider:   {args.provider}")
    print(f"   Seed:       {args.seed}")
    print(f"   Output:     {args.output_dir}")
    print(f"   Interactions: {interactions_display or '(none)'}")
    print("=" * 60)
    print()

    stage_map = {
        "generate": lambda: stage_generate(args, demographics_config),
        "simulate": lambda: stage_simulate(args, demographics_config, traits_config),
        "analyze":  lambda: stage_analyze(args, demographics_config),
    }

    if args.stage == "all":
        for name in ["generate", "simulate", "analyze"]:
            stage_map[name]()
    else:
        stage_map[args.stage]()

    print("🎉 Experiment pipeline complete!")


if __name__ == "__main__":
    main()
