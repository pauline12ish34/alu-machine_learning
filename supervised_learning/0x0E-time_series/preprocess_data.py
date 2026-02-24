#!/usr/bin/env python3
"""[summary]
"""


def preprocessing():
    """[summary]

    Returns:
        [type]: [description]
    """
    filename = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    raw_data = pd.read_csv(filename)
    df = raw_data.dropna()
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], unit='s')
    df.reset_index(inplace=True, drop=True)
    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(inplace=True, drop=True)
    df = df[0::60]
    date_time = pd.to_datetime(
        df.pop('Timestamp'))
    data_cols = ['Open',
                 'High',
                 'Low',
                 'Close',
                 'Volume_(BTC)',
                 'Volume_(Currency)',
                 'Weighted_Price']
    print(df.describe().transpose())
    plot_features = df[data_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)
    plot_features = df[data_cols][:720]
    plot_features.index = date_time[:720]
    _ = plot_features.plot(subplots=True)
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    num_features = df.shape[1]
    print('Number of features:', num_features)
    print(train_df.describe().transpose())
    plot_features = train_df[data_cols]
    plot_features.index = date_time[0:int(n * 0.7)]
    _ = plot_features.plot(subplots=True)
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)
    print('MEAN:\n', train_mean, '\n', sep='')
    print('STD:\n', train_std, '\n', sep='')
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    df_std = (df - train_mean) / train_std
    print(df_std.head())
    return train_df, val_df, test_df
