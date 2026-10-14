import argparse
from pathlib import Path
import pandas as pd
from src.diagnostic_analysis import run_diagnostic_analysis

def main():
    p = argparse.ArgumentParser(description="Run sepsis diagnostic analysis")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", default="analysis_output", help="Output folder")
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    r = run_diagnostic_analysis(pd.read_csv(a.input))
    names = ["quality_report", "descriptive_summary", "age_summary", "age_tests",
             "sex_table", "sex_percent", "sex_tests", "episode_table",
             "episode_rates", "episode_tests", "odds_ratios"]
    for name in names:
        r[name].to_csv(out / f"{name}.csv")
    pd.Series(r["metrics"], name="value").to_csv(out / "model_metrics.csv")
    (out / "model_summary.txt").write_text(r["model"].summary().as_text(), encoding="utf-8")
    for name, fig in r["figures"].items():
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    print(f"Saved results to {out.resolve()}")
    print(pd.Series(r["metrics"]).to_string())

if __name__ == "__main__":
    main()
