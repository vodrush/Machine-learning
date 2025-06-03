# Football Match Outcome Prediction

This repository contains a small machine learning demo that predicts whether the
home team will win a football match. It uses actual results from the 2022 FIFA
World Cup and trains a logistic regression model on the ranking difference
between the two teams.

## Dataset

A small dataset of real match results from the 2022 FIFA World Cup group stage
is provided in `data/football_matches.csv`. It includes the following columns:

- `home_team`
- `away_team`
- `home_team_rank`
- `away_team_rank`
- `home_goals`
- `away_goals`
- `home_win` (1 if the home team won, otherwise 0)

Only the ranking difference is used as a feature for training in this demo.

## Running the example

1. Install dependencies (requires Python 3):
   ```bash
   pip install pandas scikit-learn
   ```
2. Run the script:
```bash
python3 main.py
```
   The script trains a logistic regression model, prints the accuracy on a test
   split and then predicts the outcome of the Spain vs France match.

This example is intentionally lightweight—it demonstrates the workflow of data
loading, training and making predictions with scikit-learn. The dataset is tiny
and the model is not tuned for production use.
