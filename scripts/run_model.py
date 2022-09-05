import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from pickle import load

THRESHOLD = 0.41745963362163924

def missing(df):
    perc_missing = (df.isnull().sum() * 100 / len(df)).round(2)
    missing_value_df = pd.DataFrame({'num_missing': df.isnull().sum(),
                                     'perc_missing': perc_missing})
    missing_value_df = missing_value_df[missing_value_df.num_missing != 0]
    return missing_value_df.sort_values(by='perc_missing', ascending=False)


def parse_df(path, p):
    df_orig = pd.read_csv('../loans.csv')
    df_orig = df_orig.iloc[:, 1:]
    df_orig.Job = df_orig.Job.astype('str')
    
    df_orig['Risk'] = np.where(df_orig['Risk'] == 'bad', 1, 0)
    
    rate = df_orig['Credit amount'] / df_orig['Duration']
    df_orig.insert(loc=9, column='Rate', value=rate)
    
    for feature in missing(df_orig).index:
        df_orig[feature] = df_orig[feature].fillna('NAN')
    
    dummies_frame = pd.get_dummies(df_orig.iloc[:, :-1])
    
    df = pd.read_csv(p.file_path)
    df = df.iloc[:, 1:]
    df.Job = df.Job.astype('str')
    
    df['Risk'] = np.where(df['Risk'] == 'bad', 1, 0)
    
    rate = df['Credit amount'] / df['Duration']
    df.insert(loc=9, column='Rate', value=rate)
    
    for feature in missing(df).index:
        df[feature] = df[feature].fillna('NAN')
    
    X = pd.get_dummies(df.iloc[:, :-1])
    X = X.reindex(columns = dummies_frame.columns, fill_value=0)
    y = df['Risk']
    
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=Path)

    p = parser.parse_args()
    X, y = parse_df(p.file_path, p)
    
    scaler = load(open('../scaler.pkl', 'rb'))
    model = load(open('../model.pkl', 'rb'))
    
    X.iloc[:, :4] = scaler.transform(X.iloc[:, :4])
    prediction = [1 if i >= THRESHOLD else 0 for i in model.predict_proba(X)[:, -1]]
    
    print('PREDICTED:')
    print(prediction)
    print('TRUE:')
    print(y.tolist())


if __name__ == "__main__":
    main()