# https://keras.io/examples/timeseries/timeseries_weather_forecasting/
# https://www.researchgate.net/publication/343250071_Recognizing_Emotions_Evoked_by_Music_using_CNN-LSTM_Networks_on_EEG_signals
import pandas as pd
from utils import get_dfs, extract_classes
from sklearn import preprocessing
from tensorflow import keras
from model import run_model
from datetime import timedelta


step = 1000

past = 50000
future = 10000
learning_rate = 0.0001
batch_size = 300000
epochs = 10

def select_data_around_event(df_filtered, events, before=0, after=5):
    df_output = None
    for idx in events.loc[events.label=='start'].index:
        start = pd.to_datetime(idx) + timedelta(seconds=before)
        end = pd.to_datetime(idx) + timedelta(seconds=after)
        if df_output is None:
            df_output = df_filtered.loc[(start<=pd.to_datetime(df_filtered.index)) & (pd.to_datetime(df_filtered.index)<=end)]
        else:
            df_output = pd.concat([df_output, df_filtered.loc[(start<=pd.to_datetime(df_filtered.index)) & (pd.to_datetime(df_filtered.index)<=end)]])
        df_output = df_output.drop_duplicates().copy()
    return df_output


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


def prepare_ts_for_training(ts_df, event_df):
    df_lag = ts_df.copy()
    for i in range(1, 100):
        df_lag = df_lag.merge(ts_df.shift(i), how='inner', left_index=True, right_index=True, suffixes=('',f'_{i:02d}'))
    ts_df = df_lag.dropna().copy()
    del df_lag

    res = None
    classes = extract_classes(event_df)
    tmp_event_df = event_df.copy()
    for c in range(len(classes)):
        start_events = event_df.loc[(event_df.data == classes[c])]
        ref_start_date = start_events.index[0]
        start = event_df.loc[(event_df.data == classes[c]) & (event_df.label == 'start')].index[0]
        end = event_df.loc[(event_df.data == classes[c]) & (event_df.label == 'stop')].index

        if len(end) == 0:
            end = ts_df.index[-1]
        else:
            end = end[0]

        if c < len(classes) - 2:
            tmp_event_df = tmp_event_df.drop(tmp_event_df.index[0])

        current = ts_df[start:end]
        current['class'] = classes[c].replace('"', '')
        if res is None:
            res = current
        else:
            res = pd.concat([res, current])
    # res['date'] = res.index
    return res



def load_ds():
    feature_keys = final_df.columns
    print(feature_keys)
    selected_features = [feature_keys[i] for i in range(len(feature_keys))]
    features = final_df[selected_features]

    le = preprocessing.LabelEncoder()
    features['class'] = le.fit_transform(features['class'])
    
    # features.index = final_df['class']
    # display(features.head())
    
    features = normalize(features.values, train_split)
    features = pd.DataFrame(features)
    features[features.columns[-1]] = le.fit_transform(final_df['class'])
    # display(features.head())

    train_data = features.loc[0 : train_split - 1]
    val_data = features.loc[train_split:]


    start = past + future
    end = start + train_split

    x_train = train_data[[i for i in range(len(feature_keys) - 1)]].values
    y_train = features.iloc[start:end][features.columns[-1]]

    sequence_length = int(past / step)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    sequence_length = int(past / step)


    x_end = len(val_data) - past - future
    label_start = train_split + past + future
    x_val = val_data.iloc[:x_end][[i for i in range(len(feature_keys) - 1)]].values
    y_val = features.iloc[label_start:][features.columns[-1]]
    # display(x_val.shape)
    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,)
    return dataset_train, dataset_val



if __name__ == "__main__":

    hdf_file_path = [
        "../data/20221029-171117.hdf5",
        "../data/20221029-192231.hdf5",
        "../data/20221029-200201.hdf5",
        "../data/20221029-202757.hdf5",
        "../data/othmane_assis_EEG_20221029-231521.hdf5"
    ]

    result_filtered = None
    result_events = None

    for path in hdf_file_path:
        _, df_events, df_filtered, _ = get_dfs(path)

        df_filtered = select_data_around_event(df_filtered, df_events, before=-2, after=12)
        if result_filtered is None:
            result_filtered = df_filtered
        else:
            result_filtered = pd.concat([result_filtered, df_filtered])

        if result_events is None:
            result_events = df_events
        else:
            result_events = pd.concat([result_events, df_events])
    # Clean events to have only 3 classes
    result_events.replace('"repos"', 'neutral', inplace=True)
    result_events.replace(['"calme"', '"lent"'], 'positive', inplace=True)
    result_events.replace(['"rapide"', '"agite"'], 'negative', inplace=True)
    result_events = result_events[result_events.data != '"fin"']
    result_events = result_events[result_events.data != '"calme"']
    result_events = result_events[result_events.data != '"agite"']
    result_events = result_events[result_events.data != '"interuption"']



    final_df = prepare_ts_for_training(df_filtered, df_events)



    split_fraction = 0.715
    train_split = int(split_fraction * int(final_df.shape[0]))

    start = past + future
    end = start + train_split


    dataset_train, dataset_val = load_ds()


    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)



    m = run_model(dataset_train, dataset_val, inputs, num_classes=3, epochs=epochs)



