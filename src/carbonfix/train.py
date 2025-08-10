import argparse, yaml, os, json, time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)
    data_path = Path(cfg["data"]["csv"])
    target = cfg["data"]["target"]
    features = cfg["data"]["features"]

    df = pd.read_csv(data_path)
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(**cfg.get("model", {}))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    out_dir = Path(cfg.get("output_dir", "experiments/runs")) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    metrics = {"mae": float(mae), "r2": float(r2)}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {out_dir}")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
