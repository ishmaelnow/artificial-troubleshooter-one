# scripts/run_pipeline.py
from pathlib import Path
import sys
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # current python

def run(step_name: str, script: Path):
    print(f"\n=== {step_name}: {script} ===")
    subprocess.run([PY, str(script)], check=True)

def main():
    # Ensure data exists (run generator if needed)
    raw_csv = PROJECT_ROOT / "data" / "raw" / "system_metrics.csv"
    if not raw_csv.exists():
        print("Raw dataset not found — generating sample data first...")
        run("Generate sample data", PROJECT_ROOT / "scripts" / "download_sample_data.py")

    # Step 3: issue detection
    run("Step 3 — Issue detection", PROJECT_ROOT / "scripts" / "run_issue_detection.py")

    # Step 4: z-score explanations (transparency)
    run("Step 4 — Explain anomalies (z-score)", PROJECT_ROOT / "scripts" / "explain_anomalies_zscore.py")

    # Step 5: root cause classification (use predicted causes)
    run("Step 5 — Root cause classification", PROJECT_ROOT / "scripts" / "run_root_cause.py")

    # Step 6: recommendations (map predicted causes to fixes)
    run("Step 6 — Recommend solutions", PROJECT_ROOT / "scripts" / "recommend_solutions.py")

    # Step 7: end-to-end check on a hold-out window
    run("Step 7 — Test agent on hold-out", PROJECT_ROOT / "scripts" / "test_agent.py")

    print("\n✅ Pipeline finished. Artifacts are in data/processed/")

if __name__ == "__main__":
    main()
