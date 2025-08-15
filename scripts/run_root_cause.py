# scripts/run_root_cause.py
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from agent.root_cause import (
    FEATURES, RootCauseAnalyzer, RootCauseConfig, attach_root_cause_labels
)

def main() -> None:
    # Use Step 4 output (class-style z-score explanations)
    src_path = PROJECT_ROOT / "data" / "processed" / "anomaly_explanations_zscore.csv"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing {src_path}. Run scripts/explain_anomalies_zscore.py first.")

    df = pd.read_csv(src_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Keep only truly anomalous rows for cause prediction
    # (Our pipeline encodes anomalies as 1; if your variant uses -1, adjust here.)
    anomalies = df[df["anomaly"] == 1].copy()

    # Attach root_cause labels from 'anomalous_columns'
    anomalies = attach_root_cause_labels(anomalies)

    # Time-based split: first 80% train (historical), last 20% test
    split_idx = int(len(anomalies) * 0.8) if len(anomalies) > 0 else 0
    train, test = anomalies.iloc[:split_idx], anomalies.iloc[split_idx:]

    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("Not enough anomalous rows to train/test. Generate more data or lower detection threshold.")

    # Train decision tree (trees don't need scaling)
    clf = RootCauseAnalyzer(RootCauseConfig(max_depth=4, min_samples_leaf=5, random_state=42))
    clf.fit(train[FEATURES], train["root_cause"])

    # Evaluate on test set
    preds = clf.predict(test[FEATURES])
    print("Classes seen in training:", sorted(train["root_cause"].unique().tolist()))
    print("\nClassification report (test):")
    print(classification_report(test["root_cause"], preds, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(test["root_cause"], preds, labels=sorted(train["root_cause"].unique().tolist())))

    print("\nFeature importances:")
    print(clf.feature_importances())

    # Preview a few predictions
    preview_cols = ["timestamp"] + FEATURES + ["root_cause"]
    preview = test[preview_cols].copy()
    preview["predicted_root_cause"] = preds
    print("\nSample predictions:")
    print(preview.head(10).to_string(index=False))

    # Save predictions
    out_path = PROJECT_ROOT / "data" / "processed" / "root_cause_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(out_path, index=False)
    print(f"\nSaved predictions to: {out_path}")

if __name__ == "__main__":
    main()
