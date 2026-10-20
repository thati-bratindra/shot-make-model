# Shot Make Probability Model

A model that estimates the probability an NBA shot goes in, from where it was taken, where the defenders were, the shot clock and who was involved.

It works in two stages. A gradient boosted model (xgboost) first learns how hard each shot is from the situation alone, with no player or team identity. Its out-of-fold predictions are then used to rate how much each shooter, closest defender and team beats that expectation, with shrinkage for small samples. A second gradient boosted model uses the shot features plus those ratings, and is blended with a spline logistic regression on the log-odds scale. The full walkthrough, with charts, is in [`writeup.html`](writeup.html) (source: [`writeup.Rmd`](writeup.Rmd)).

## Data

The data in `data/` is **synthetic**. [`simulate_data.R`](simulate_data.R) invents 30 teams, a pool of players and 300,000 shots (200,000 for training over two seasons, 100,000 to predict from a third season) with the same columns as player-tracking shot data: shot location, distance, shot type, game state, dribbles, shot clock, shooter speed, the closest defender's distance at release and over the last second, and up to four contesters. Each player has hidden shooting and defending skills, and every make or miss is drawn from a probability built from those pieces, so the results show how well the model recovers a known process.

## Run it

Requires R with `readr`, `dplyr`, `stringr`, `xgboost` (1.7 or newer; the code handles the 2.x API changes), and `ggplot2`, `knitr` and `rmarkdown` for the writeup.

```r
install.packages(c("readr", "dplyr", "stringr", "xgboost", "ggplot2", "knitr", "rmarkdown"))
```

```bash
Rscript simulate_data.R     # writes data/training.csv.gz and data/testing.csv.gz (already included)
Rscript shot_model.R        # trains, cross-validates, writes predictions.csv and model_artifacts.rds
Rscript -e 'rmarkdown::render("writeup.Rmd")'   # rebuilds writeup.html from model_artifacts.rds
```

The `.csv.gz` files can be read directly without unzipping, e.g. `readr::read_csv("data/training.csv.gz")` in R or `polars.read_csv("data/training.csv.gz")` in Python. `shot_model.R` runs in a couple of minutes on a laptop; set `SAMPLE_N` at the top to debug on a subset.

## Results on the synthetic data

| Model | Out-of-fold log-loss |
|---|---|
| League average for every shot | 0.6912 |
| Stage 1 GBM (no identity) | 0.6406 |
| Stage 2 GBM (with skill ratings) | 0.6372 |
| Spline logistic regression | 0.6388 |
| Final blend (75% GBM, 25% spline) | 0.6370 |

The predictions are well calibrated: across 20 equal-sized bins, the average predicted chance and the actual make rate agree to within 1.3 percentage points.

## Files

| File | What it is |
|---|---|
| `shot_model.R` | Feature engineering, both model stages, skill ratings, blend, calibration check and outputs |
| `simulate_data.R` | Generates the synthetic training and testing data |
| `writeup.Rmd` / `writeup.html` | Plain-language walkthrough of the method and results |
| `predictions.csv` | `shot_id`, `make_prob` for every testing shot |
| `model_artifacts.rds` | Scores, importance, SHAP sample and calibration tables the writeup renders from |
