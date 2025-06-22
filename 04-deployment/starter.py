import sys
import pickle
import pandas as pd
#from IPython import get_ipython


output_file = 'data/output_file.parquet'



with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):

    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def predict(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def answer_q1(y_pred):
    y_pred_series = pd.Series(y_pred)
    print("y prediction Standard Deviation : ", float(y_pred_series.describe().loc['std'].round(3)))


def answer_q2(df, y_pred, output_file, year, month):
    #Q2. Preparing the output

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    print("mean predicted duration: ", float(df_result['predicted_duration'].mean().round(3)))
    
    """
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print("New parquet file created: ", output_file)"""

    #get_ipython().system('ls -lh $output_file')



def answer_q3():
    #get_ipython().system('jupyter nbconvert --to script starter.ipynb')
    pass


def run(year, month, output_file):
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = read_data(filename)
    y_pred = predict(df)
    answer_q2(df, y_pred, "data/output_file2.parquet", year, month)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python starter.py <year> <month>")
        sys.exit(1)

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    if not (1 <= month <= 12):
        print("Month must be between 1 and 12")
        sys.exit(1)

    run(year, month, output_file)