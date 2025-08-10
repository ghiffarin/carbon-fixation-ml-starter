# Carbon Fixation ML Starter

Clean, simple scaffold to build ML projects around microalgae carbon fixation.

## What's inside
- `data/` raw & processed placeholders (git-ignored except for `.gitkeep`)
- `notebooks/` exploratory notebooks
- `src/carbonfix/` Python package for reusable code
- `experiments/` tracks runs and configs
- `models/` saved model artifacts (git-ignored)
- `reports/figures/` outputs for papers/blogs
- `.github/workflows/ci.yml` basic tests & lint on push

## Quickstart
```bash
# create and activate env (choose one)
python -m venv .venv && source .venv/bin/activate      # mac/linux
# or
python -m venv .venv && .\.venv\Scripts\activate     # windows

pip install -r requirements.txt

# run tests & lint
pytest -q
ruff check .
mypy src
```

## First experiment idea (P1)
Train a baseline regressor to predict daily CO₂ fixation (or biomass productivity) from ops + weather features.

Steps:
1) Drop a CSV in `data/raw/` (e.g., ATP3 subset).
2) Run `notebooks/01_eda.ipynb` to profile data.
3) Configure `experiments/baseline.yaml`.
4) `python -m carbonfix.train --config experiments/baseline.yaml`

## Model card (template)
- **Task**: Predict daily CO₂ fixation [kg CO₂/m³/day]
- **Data**: source, time range, sites
- **Metrics**: MAE, RMSE, MAPE, by-site breakdown
- **Caveats**: domain shift, sensor noise, missing weather
- **Intended use**: planning & ops; not safety-critical control
