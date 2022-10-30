import pandas as pd

def get_dfs(hdf_file_path):
    paths = ['/raw', '/events', '/filtered', '/bands']
    result = []

    for path in paths:
        ts_df = pd.read_hdf(hdf_file_path, path)
        result.append(ts_df)
        if path == '/events' and 'stop' not in ts_df.label.values:
            ts_df.loc[result[0].index[-1]] = ['stop', ts_df.data.values[-1]]
            
    return (r for r in result)


def extract_classes(df_events):
    classes = []
    for index, row in df_events.iterrows():
        if row['label'] == 'start':
            classes.append(row['data'])
    return classes