import pandas as pd


def split_data(df):
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_size = 0.2
    split_idx = int(len(df_shuffled) * (1 - test_size))

    train_data = df_shuffled[:split_idx]
    test_data = df_shuffled[split_idx:]

    X_train = train_data.drop(['Primary energy consumption per capita (kWh/person)'], axis=1)
    y_train = train_data[['Primary energy consumption per capita (kWh/person)']]

    X_test = test_data.drop(['Primary energy consumption per capita (kWh/person)'], axis=1)
    y_test = test_data[['Primary energy consumption per capita (kWh/person)']]

    return X_train, X_test, y_train, y_test