import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "data/football_matches.csv"


def load_data(path: str = DATA_PATH):
    """Load match results and return dataframe, features and labels."""
    data = pd.read_csv(path)
    data["rank_diff"] = data["home_team_rank"] - data["away_team_rank"]
    X = data[["rank_diff"]]
    y = data["home_win"]
    return data, X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Train logistic regression and print accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2f}")
    return model


def get_team_rank(data: pd.DataFrame, team: str) -> int:
    """Return the known ranking for a team from the dataset."""
    ranks = pd.concat(
        [
            data.loc[data["home_team"] == team, "home_team_rank"],
            data.loc[data["away_team"] == team, "away_team_rank"],
        ]
    )
    return int(ranks.iloc[0]) if not ranks.empty else None


def predict_match(model: LogisticRegression, data: pd.DataFrame, home_team: str, away_team: str) -> int:
    """Predict the outcome of a match given the team names."""
    home_rank = get_team_rank(data, home_team)
    away_rank = get_team_rank(data, away_team)
    if home_rank is None or away_rank is None:
        raise ValueError("Team ranking not found in dataset")
    rank_diff = home_rank - away_rank
    proba = model.predict_proba([[rank_diff]])[0, 1]
    pred = int(proba >= 0.5)
    result = "home win" if pred else "home draw/loss"
    print(
        f"Prediction for {home_team} vs {away_team}: {result} (prob={proba:.2f})"
    )
    return pred

def main():
    data, X, y = load_data()
    model = train_model(X, y)
    # Predict the Spain vs France match with Spain at home
    predict_match(model, data, "Spain", "France")

if __name__ == '__main__':
    main()
