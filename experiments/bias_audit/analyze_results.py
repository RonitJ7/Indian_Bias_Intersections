"""
Module 3: Analysis Engine

Loads trait scores and persona demographics, runs OLS regression with interaction
terms, and outputs coefficient tables and visualizations.

Usage:
    python analyze_results.py --interactions "sex:race,race:religion"
    python analyze_results.py --interactions "sex:race" --reference_categories '{"sex":"Male","race":"White"}'
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import load_demographics_config, load_traits_config


# ============================================================================
# Data Loading
# ============================================================================

def load_data(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load persona demographics and trait scores, merge them.

    Returns:
        personas_df: DataFrame with demographic columns
        scores_df: DataFrame with trait score columns
    """
    # Load personas CSV
    personas_path = results_dir / "personas.csv"
    if not personas_path.exists():
        raise FileNotFoundError(f"Personas CSV not found: {personas_path}")
    personas_df = pd.read_csv(personas_path)

    # Load trait scores JSONL
    scores_path = results_dir / "scores" / "trait_scores.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError(f"Trait scores not found: {scores_path}")

    records = []
    with open(scores_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    scores_df = pd.DataFrame(records)

    # Merge on persona_id
    merged = personas_df.merge(scores_df, on="persona_id", how="inner")
    print(f"📊 Loaded {len(merged)} personas with trait scores.")

    # Separate back into demographics and scores
    trait_cols = [c for c in scores_df.columns if c != "persona_id"]
    demo_cols = [c for c in personas_df.columns if c != "persona_id"]

    return merged, demo_cols, trait_cols


# ============================================================================
# Feature Engineering
# ============================================================================

def one_hot_encode(
    df: pd.DataFrame,
    demo_cols: List[str],
    reference_categories: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode demographic columns, dropping the reference category.

    Args:
        df: DataFrame with demographic columns
        demo_cols: List of demographic column names
        reference_categories: Dict mapping dimension -> reference value to drop.
                            If None, drops the first alphabetically.

    Returns:
        X: DataFrame with one-hot encoded features
        feature_names: List of feature column names
    """
    if reference_categories is None:
        reference_categories = {}

    encoded_parts = []
    for col in demo_cols:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)

        # Determine which to drop
        ref_val = reference_categories.get(col)
        if ref_val:
            drop_col = f"{col}_{ref_val}"
        else:
            # Drop first alphabetically
            drop_col = sorted(dummies.columns)[0]

        if drop_col in dummies.columns:
            dummies = dummies.drop(columns=[drop_col])
        else:
            warnings.warn(f"Reference category '{drop_col}' not found. Dropping first column.")
            dummies = dummies.iloc[:, 1:]

        encoded_parts.append(dummies)

    X = pd.concat(encoded_parts, axis=1)
    return X, list(X.columns)


def add_interaction_terms(
    X: pd.DataFrame,
    interactions: List[Tuple[str, str]],
    demo_cols: List[str],
) -> pd.DataFrame:
    """
    Add interaction terms between pairs of demographic dimensions.

    Args:
        X: One-hot encoded feature DataFrame
        interactions: List of (dim1, dim2) pairs, e.g., [("sex", "race")]
        demo_cols: Original demographic column names

    Returns:
        X with interaction columns appended
    """
    for dim1, dim2 in interactions:
        # Find columns belonging to each dimension
        cols1 = [c for c in X.columns if c.startswith(f"{dim1}_")]
        cols2 = [c for c in X.columns if c.startswith(f"{dim2}_")]

        for c1 in cols1:
            for c2 in cols2:
                interaction_name = f"{c1}__x__{c2}"
                X[interaction_name] = X[c1] * X[c2]

    return X


def parse_interactions(interactions_str: str) -> List[Tuple[str, str]]:
    """Parse interaction string like 'sex:race,race:religion' into list of tuples."""
    if not interactions_str:
        return []
    pairs = []
    for pair in interactions_str.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


# ============================================================================
# Regression
# ============================================================================

def run_regression(
    X: pd.DataFrame,
    y: pd.Series,
    trait_name: str,
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Fit OLS regression: y ~ X (with constant).

    Returns:
        statsmodels OLS results object, or None if fitting fails.
    """
    # Drop rows with NaN in y
    mask = ~y.isna()
    X_clean = X[mask].copy()
    y_clean = y[mask].copy()

    if len(y_clean) < X_clean.shape[1] + 10:
        print(f"  ⚠️ {trait_name}: Too few observations ({len(y_clean)}) for {X_clean.shape[1]} features")
        return None

    # Add constant
    X_with_const = sm.add_constant(X_clean, has_constant='skip')

    try:
        model = sm.OLS(y_clean, X_with_const)
        results = model.fit()
        return results
    except Exception as e:
        print(f"  ⚠️ {trait_name}: Regression failed: {e}")
        return None


def extract_coefficients(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    trait_name: str,
) -> pd.DataFrame:
    """Extract coefficient table from OLS results."""
    coef_df = pd.DataFrame({
        "trait": trait_name,
        "feature": results.params.index,
        "coefficient": results.params.values,
        "std_error": results.bse.values,
        "t_statistic": results.tvalues.values,
        "p_value": results.pvalues.values,
        "ci_lower": results.conf_int()[0].values,
        "ci_upper": results.conf_int()[1].values,
    })

    # Exclude constant
    coef_df = coef_df[coef_df["feature"] != "const"].copy()

    # Add significance markers
    coef_df["significance"] = ""
    coef_df.loc[coef_df["p_value"] < 0.05, "significance"] = "*"
    coef_df.loc[coef_df["p_value"] < 0.01, "significance"] = "**"
    coef_df.loc[coef_df["p_value"] < 0.001, "significance"] = "***"

    return coef_df


# ============================================================================
# Visualization
# ============================================================================

def plot_coefficient_heatmap(
    all_coefficients: pd.DataFrame,
    output_path: Path,
    title: str = "LLM Bias: Regression Coefficients by Demographic Feature",
) -> None:
    """
    Create a heatmap of regression coefficients (features × traits).

    Only shows main effects (no interaction terms) for readability.
    """
    # Filter to main effects only (no __x__ interactions)
    main_effects = all_coefficients[~all_coefficients["feature"].str.contains("__x__")]

    # Pivot to matrix
    pivot = main_effects.pivot_table(
        index="feature", columns="trait", values="coefficient", aggfunc="first"
    )

    if pivot.empty:
        print("  ⚠️ No coefficients to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2.5), max(8, len(pivot) * 0.4)))

    # Create heatmap
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-max(abs(pivot.values.min()), abs(pivot.values.max())),
                   vmax=max(abs(pivot.values.min()), abs(pivot.values.max())))

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Get significance
                mask = (
                    (main_effects["feature"] == pivot.index[i]) &
                    (main_effects["trait"] == pivot.columns[j])
                )
                sig = main_effects.loc[mask, "significance"].values
                sig_str = sig[0] if len(sig) > 0 else ""
                text_color = "white" if abs(val) > pivot.values.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}{sig_str}", ha="center", va="center",
                       fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Coefficient (β)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Heatmap saved to {output_path}")


def plot_forest(
    all_coefficients: pd.DataFrame,
    trait_name: str,
    output_path: Path,
) -> None:
    """Create a forest plot for a single trait showing coefficients with CIs."""
    trait_df = all_coefficients[all_coefficients["trait"] == trait_name].copy()
    trait_df = trait_df.sort_values("coefficient")

    if trait_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(trait_df) * 0.35)))

    y_pos = range(len(trait_df))
    colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in trait_df["p_value"]]

    ax.barh(y_pos, trait_df["coefficient"], color=colors, alpha=0.7, height=0.6)
    ax.errorbar(
        trait_df["coefficient"], y_pos,
        xerr=[
            trait_df["coefficient"] - trait_df["ci_lower"],
            trait_df["ci_upper"] - trait_df["coefficient"]
        ],
        fmt="none", ecolor="black", elinewidth=1, capsize=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(trait_df["feature"], fontsize=9)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Coefficient (β)", fontsize=12)
    ax.set_title(f"Regression Coefficients: {trait_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="p < 0.05"),
        Patch(facecolor="#95a5a6", alpha=0.7, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_intersectional_heatmap(
    all_coefficients: pd.DataFrame,
    output_path: Path,
    p_threshold: float = 0.01,
    title: str = "LLM Intersectional Bias: Significant Interaction Coefficients (p < 0.01)",
) -> None:
    """
    Create a heatmap of statistically significant interaction-term coefficients.

    Only shows features containing '__x__' (interaction terms) where p < p_threshold.
    Rows are interaction pairs, columns are personality traits.
    """
    # Filter to interaction terms only with significance threshold
    interactions_df = all_coefficients[
        all_coefficients["feature"].str.contains("__x__") &
        (all_coefficients["p_value"] < p_threshold)
    ].copy()

    if interactions_df.empty:
        print(f"  ⚠️ No significant interaction terms found at p < {p_threshold}. Skipping intersectional heatmap.")
        return

    # Pivot: rows = interaction features, columns = traits
    pivot = interactions_df.pivot_table(
        index="feature", columns="trait", values="coefficient", aggfunc="first"
    )

    # Sort rows by max absolute coefficient across traits for readability
    pivot = pivot.reindex(
        pivot.abs().max(axis=1).sort_values(ascending=False).index
    )

    n_rows = len(pivot)
    n_cols = len(pivot.columns)
    fig_width = max(10, n_cols * 2.5)
    fig_height = max(6, n_rows * 0.55)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Symmetric color scale
    abs_max = float(np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(
        pivot.values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-abs_max,
        vmax=abs_max,
    )

    # Axis labels — clean up column names
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in pivot.columns],
        rotation=45, ha="right", fontsize=11,
    )
    ax.set_yticks(range(n_rows))
    # Replace __x__ with x for readability
    clean_labels = [f.replace("__x__", " x ").replace("_", " ") for f in pivot.index]
    ax.set_yticklabels(clean_labels, fontsize=8)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            feature_name = pivot.index[i]
            trait_name = pivot.columns[j]
            row_mask = (
                (interactions_df["feature"] == feature_name) &
                (interactions_df["trait"] == trait_name)
            )
            sig_arr = interactions_df.loc[row_mask, "significance"].values
            sig_str = sig_arr[0] if len(sig_arr) > 0 else ""
            text_color = "white" if abs(val) > abs_max * 0.6 else "black"
            ax.text(
                j, i, f"{val:+.3f}{sig_str}",
                ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coefficient (beta)", fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.set_xlabel("Personality Trait", fontsize=11)
    ax.set_ylabel("Interaction Term (Race x Religion  /  Sex x Race)", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Intersectional heatmap saved to {output_path}")

# ============================================================================
# Main
# ============================================================================

def run_analysis(
    results_dir: str,
    interactions_str: str = "",
    reference_categories: Optional[Dict[str, str]] = None,
) -> None:
    """Run the full analysis pipeline."""
    results_path = Path(results_dir)
    regression_dir = results_path / "regression_different_defaults"
    regression_dir.mkdir(parents=True, exist_ok=True)

    # Default reference categories
    if reference_categories is None:
        reference_categories = {
            "sex": "Female",
            "race": "Other",
            "religion": "Other",
            "political_views": "Liberal",
            "political_party": "Something else",
        }

    # Load data
    merged_df, demo_cols, trait_cols = load_data(results_path)

    # One-hot encode
    X, feature_names = one_hot_encode(merged_df, demo_cols, reference_categories)
    print(f"   Features: {len(feature_names)} main effects")

    # Add interaction terms
    interactions = parse_interactions(interactions_str)
    if interactions:
        X = add_interaction_terms(X, interactions, demo_cols)
        n_interactions = len(X.columns) - len(feature_names)
        print(f"   + {n_interactions} interaction terms")
        print(f"   Total features: {len(X.columns)}")

    # Run regression for each trait
    all_coefficients = []
    model_summaries = []

    print(f"\n🔬 Running OLS regression for {len(trait_cols)} traits...\n")

    for trait in trait_cols:
        y = merged_df[trait].astype(float)
        results = run_regression(X, y, trait)

        if results is not None:
            coef_df = extract_coefficients(results, trait)
            all_coefficients.append(coef_df)

            # Print summary
            n_sig = (coef_df["p_value"] < 0.05).sum()
            print(f"  ✅ {trait}: R²={results.rsquared:.4f}, "
                  f"Adj.R²={results.rsquared_adj:.4f}, "
                  f"{n_sig}/{len(coef_df)} significant features (p<0.05)")

            model_summaries.append({
                "trait": trait,
                "r_squared": results.rsquared,
                "adj_r_squared": results.rsquared_adj,
                "f_statistic": results.fvalue,
                "f_pvalue": results.f_pvalue,
                "n_observations": int(results.nobs),
                "n_features": X.shape[1],
                "n_significant": n_sig,
            })

            # Save full statsmodels summary
            with open(regression_dir / f"{trait}_ols_summary.txt", 'w') as f:
                f.write(str(results.summary()))

    if not all_coefficients:
        print("❌ No successful regressions. Check your data.")
        return

    # Combine all coefficients
    all_coef_df = pd.concat(all_coefficients, ignore_index=True)

    # Save coefficient table
    coef_path = regression_dir / "coefficients.csv"
    all_coef_df.to_csv(coef_path, index=False)
    print(f"\n📋 Coefficient table saved to {coef_path}")

    # Save model summaries
    summary_path = regression_dir / "model_summaries.csv"
    pd.DataFrame(model_summaries).to_csv(summary_path, index=False)
    print(f"📋 Model summaries saved to {summary_path}")

    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")

    # Heatmap of all coefficients
    plot_coefficient_heatmap(
        all_coef_df,
        regression_dir / "coefficient_heatmap.png",
    )

    # Forest plots per trait
    for trait in trait_cols:
        plot_forest(
            all_coef_df, trait,
            regression_dir / f"forest_{trait}.png",
        )
    print(f"  📊 Forest plots saved to {regression_dir}/")

    # Intersectional heatmap — only interaction terms with p < 0.01
    plot_intersectional_heatmap(
        all_coef_df,
        regression_dir / "intersectional_heatmap.png",
        p_threshold=0.01,
    )

    # Print the most significant biases
    print(f"\n{'=' * 60}")
    print("🔍 TOP SIGNIFICANT BIASES (p < 0.05):")
    print(f"{'=' * 60}")

    sig = all_coef_df[all_coef_df["p_value"] < 0.05].sort_values("p_value")
    if sig.empty:
        print("  No statistically significant biases found.")
    else:
        for _, row in sig.head(20).iterrows():
            direction = "↑" if row["coefficient"] > 0 else "↓"
            print(f"  {direction} {row['trait']:25s} | {row['feature']:35s} | "
                  f"β={row['coefficient']:+.4f} | p={row['p_value']:.4f} {row['significance']}")

    print(f"\n✅ Analysis complete! All results in {regression_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze bias audit results with OLS regression."
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Results directory (default: results_new_defaults/)"
    )
    parser.add_argument(
        "--interactions", type=str, default="sex:race,race:religion",
        help="Interaction terms as 'dim1:dim2,dim3:dim4' (default: 'sex:race,race:religion')"
    )
    parser.add_argument(
        "--reference_categories", type=str, default=None,
        help="JSON string of reference categories, e.g., '{\"sex\":\"Male\",\"race\":\"White\"}'"
    )
    args = parser.parse_args()

    results_dir = args.results_dir or str(Path(__file__).parent / "results")

    ref_cats = None
    if args.reference_categories:
        ref_cats = json.loads(args.reference_categories)

    run_analysis(
        results_dir=results_dir,
        interactions_str=args.interactions,
        reference_categories=ref_cats,
    )


if __name__ == "__main__":
    main()
