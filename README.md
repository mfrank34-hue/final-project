# Charlotte next-day weather models

This repository uses daily Charlotte weather observations to study two questions:

1. Can a linear regression predict tomorrow's maximum temperature (°F) from recent high temperature, low temperature, rainfall, and the calendar date?
2. Can a decision tree estimate whether tomorrow will have measurable rain? A **rain** decision is made when its estimated probability is **greater than 15%**.

The modeling code is in [`weather_models.py`](weather_models.py). The held-out results are in [`model_results.md`](model_results.md). This phase contains the two requested models; the course rubric requires a third trained model for the final report.

## Data and preparation

[`loaddata.csv`](loaddata.csv) is the group's compiled daily weather dataset for Charlotte, North Carolina. The project notes identify [Weather Underground monthly history](https://www.wunderground.com/history/monthly/us/nc/charlotte/date/2021-1) as the original data source. Each row is a calendar day, from January 1, 2021 through October 26, 2026. The CSV has 2,156 rows, including 31 exact duplicate dates from August 2023. The script removes those duplicates, converts temperature and precipitation fields to numbers, and only uses complete seven-day windows followed by the next calendar day. The source CSV stays unchanged.

The supplied CSV includes high and low temperatures, dew point, humidity, maximum wind, pressure, and precipitation. This first model pair uses high temperature, low temperature, and precipitation plus the known calendar date. Wind is excluded in line with the group's brainstorming notes. The CSV does not contain a recorded probability of rain: **the 15% figure is a cutoff applied to the model's predicted probability**, while the observed rain label is next-day precipitation greater than 0 inches. This label may include trace amounts and does not separate rain from other precipitation.

## Modeling and validation

For a target date, the regression uses the preceding three days and the decision tree uses the preceding seven days. `lag_0` is the most recent day; `lag_6` is six days earlier. Sine and cosine of the target calendar day give both models a seasonal cue that is known before the forecast. The regression also uses a second seasonal harmonic. No target-day weather value is used as an input.

- **Linear regression:** ordinary least squares, fitted after scaling input columns using training data only. A three-day feature window was selected using rolling validation in 2023–2025, then checked once on the 2026 holdout. Baseline: tomorrow's high equals today's high. Evaluation: MAE, RMSE, and R².
- **Decision tree:** a binary CART tree using Gini impurity, maximum depth 4, and at least 30 training examples per leaf. Leaf rain frequencies are its probability estimates. Baselines: the training-period rain rate and whether it rained today. Evaluation: accuracy, precision, recall, F1, Brier score, and confusion matrix. The 15% cutoff is fixed by the project requirement.

Targets through December 31, 2025 form the training set; January 1 through October 26, 2026 are held out for testing. This chronological split checks later weather without using 2026 labels to fit or select the models. The tree settings were set before viewing test scores. The regression's 2026 improvement is small, so it should be treated as tentative. These results are retrospective and are **not a current weather forecast**. In particular, the tree probabilities have not been calibrated, and its 15% cutoff causes many false positives on the held-out data.

## Reproduce the results

Use Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python weather_models.py --output model_results.md
```

Run the script from the repository root. By default it reads the adjacent `loaddata.csv`. Use `--data path/to/file.csv` to select another file with the same columns. Omit `--output` to print the report instead. The code implements both models directly with NumPy so it does not depend on scikit-learn.

## Scope and transparency

The final report still needs a third trained model, exploratory visualizations, peer-reviewed background sources, analysis of measurement and social effects, and group contribution details. Model results should be described as predictions from historical associations, not causal effects or operational forecasts. The group should check the source records and assess whether errors differ by season or weather conditions before interpreting possible uses.

