import pandas as pd
import numpy as np
import click
import sys
from pickle import load
from configparser import ConfigParser
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression


def read_file(path: str) -> pd.DataFrame:
    """Function to create DataFrame from input .csv file path"""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print('Error: Invalid file path')
        sys.exit(1)
    else:
        return df


def load_pickles() -> tuple[PowerTransformer, LogisticRegression]:
    """Function to load dumped .pkl model and scaler"""
    try:
        scaler = load(open('pickles/scaler.pkl', 'rb'))
        model = load(open('pickles/model.pkl', 'rb'))
    except FileNotFoundError:
        print('Error: Dumped model and/or scaler not found')
        sys.exit(1)
    else:
        return scaler, model


def validate_df(df: pd.DataFrame, config):
    """Helper function to validate columns compatibility"""
    columns = config['DATAFRAME']['columns'].split(',')
    dtypes = config['DATAFRAME']['dtypes'].split(',')

    if df.columns.tolist() != columns:
        raise ValueError('Incompatible input .csv file columns, check config columns')
    if [str(x) for x in df.dtypes.values.tolist()] != dtypes:
        raise ValueError('Incompatible input .csv file columns dtypes, check config dtypes')


def parse_df(df: pd.DataFrame, scaler: PowerTransformer, config) -> tuple[pd.DataFrame, pd.Series]:
    """Parsing dataframe and preparing for model input"""
    ohc_columns = config['DATAFRAME']['ohc_columns'].split(',')
    missing_columns = config['DATAFRAME']['missing_columns'].split(',')

    # removing first column, casting Job type, encoding Risk column
    df = df.iloc[:, 1:]
    df['Job'] = df['Job'].astype('str')
    df['Risk'] = np.where(df['Risk'] == 'bad', 1, 0)

    # adding new feature - Rate
    rate = df['Credit amount'] / df['Duration']
    df.insert(loc=9, column='Rate', value=rate)

    # imputing potential NAs based on config
    for column in missing_columns:
        df[column] = df[column].fillna('NAN')

    # splitting dataframe into X and y
    X = pd.get_dummies(df.iloc[:, :-1])
    X = X.reindex(columns=ohc_columns, fill_value=0)
    y = df['Risk']

    # scaling
    X.iloc[:, :4] = scaler.transform(X.iloc[:, :4])

    return X, y


def make_prediction(X: pd.DataFrame, y: pd.Series, model: LogisticRegression, config) -> pd.DataFrame:
    """Make predictions and create df in comparison to actual values"""
    threshold = float(config['MODEL']['threshold'])
    result_df = pd.DataFrame(columns=['prediction', 'actual'])
    prediction = [1 if i >= threshold else 0 for i in model.predict_proba(X)[:, -1]]

    result_df['prediction'] = prediction
    result_df['actual'] = y
    return result_df


@click.command()
@click.argument('path', default='data/sample_data.csv')
def main(path):
    """PATH is a path to the input .csv file"""
    config = ConfigParser()
    config.read('config.ini')

    df = read_file(path=path)
    scaler, model = load_pickles()

    validate_df(df, config)
    X, y = parse_df(df, scaler, config)
    results = make_prediction(X, y, model, config)
    print(results)

    accuracy = sum(1 for x, y in zip(results['prediction'], results['actual']) if x == y) / len(results)
    print(f'Accuracy: {round(accuracy, 2)}')


if __name__ == '__main__':
    main()
