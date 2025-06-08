import pandas as pd
import mlflow
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from prefect import flow, task
from typing import List


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("03-orchestration")


@task
def read_dataframe(year, month) -> pd.DataFrame():
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(filename)
    print("number of rows before wrangling", df.shape[0])

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print("number of rows after wrangling", df.shape[0])

    return df


@task
def features_selection_and_encoding(df, encoder) -> List:
    categorical = ['PULocationID', 'DOLocationID']
    #numerical = ["trip_distance"]
    features = df[categorical]
    df_dicts = features.to_dict(orient="records")

    try:
        check_is_fitted(encoder)
        X = encoder.transform(df_dicts)
    except NotFittedError:
        X = encoder.fit_transform(df_dicts)

    target = "duration"
    y = df[target]
    return [X, y, encoder]

@task
def train_model(X, y, encoder) -> None:
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X, y)
        print("model intercept", lr.intercept_)
        mlflow.log_metric("intercept_i", lr.intercept_)
        mlflow.sklearn.log_model(encoder, "Dict Vectorizer")
        mlflow.sklearn.log_model(lr, "linear regression")

@flow
def run_training(year, month):
    df = read_dataframe(year, month)
    dv = DictVectorizer()
    X, y, dv = features_selection_and_encoding(df, dv)
    train_model(X, y, dv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year')
    parser.add_argument('--month', type=int, default=1, help='month')
    args = parser.parse_args()

    run_training(args.year, args.month)


