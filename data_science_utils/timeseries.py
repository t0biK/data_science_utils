import pandas as pd
from tqdm import tqdm as tqdm
import multiprocessing
from functools import reduce
import numpy as np
from tsfresh.feature_extraction.settings import MinimalFCParameters


def get_sub_timeseries(df_x, index, window_start, window_end, identifier):
    """
    Helper method which extracts a sub dataframe of a pandas dataframe. The sub dataframe is defined by a
    index aswell as the relative start end end index of the window. An identifier column is added with a
    constant value which is given as parameter. This ensures multiple sub dataframes can be distinguished
    from each other.

    Example:
    -----[|-----------|-----------|----------|]------------|-------------|---------------->
      index-window_end                 index-window_start              index

    Because iloc is not supported in dask dataframes it is assumed the index equals the level of the row
    (like reset_index does).
    :param df_x: the pandas dataframe the sub dataframe should be extracted
    :param index: absolute index of the dataframe the window is extracted
    :param window_start: relative start of the subwindow
    :param window_end: relative end of the subwindow
    :param identifier: a unique constant identifier to distinguish later the sub dataframe
    :return: the extracted sub dataframe
    """
    sub_df_x = df_x.iloc[index - window_end:index - window_start].copy()
    sub_df_x['window_id'] = identifier
    return sub_df_x


def make_windows(arr, win_size, step_size, start_window_id=1):
    """
    arr: any 2D array whose columns are distinct variables and
      rows are data records at some timestamp t
    win_size: size of data window (given in data points)
    step_size: size of window step (given in data point)

    Note that step_size is related to window overlap (overlap = win_size - step_size), in
    case you think in overlaps.
    """
    w_list = list()
    n_records = arr.shape[0]
    remainder = (n_records - win_size) % step_size
    num_windows = 1 + int((n_records - win_size - remainder) / step_size)
    for k in tqdm(range(num_windows)):
        a = arr[k * step_size:win_size - 1 + k * step_size + 1]
        b = np.append(a, [[k + start_window_id]] * a.shape[0], axis=1)
        w_list.append(b)
    return np.concatenate(np.array(w_list), axis=0)


def get_rolling_timeseries(df_x, start_index, lag, window_size):
    """
    Extracts all possible sub windows of the given dataframe. It is assumed that the index of the dataframe is
    the default one (reset_index()).
    Example:
    -----[|[[...-------|-----------|----------|]]]...--------|-------------|-------------|------->
         |<---    window_end-window_start  --->|           index
         |<---                 window_end               --->|
                                             |<---window--->|
                                                 _start
                                                            |<---        lag        -->|
    :param df_x: pandas dataframe where the sub windows are extracted.
    :param start_index: the first index the sub windows should be extracted.
                        This is necessary because the method can be applied multiple times with different windows.
                        To merge the extracted features later on the window id must match.
    :param lag: the distance between the current row and the target row (y). necessary to limit the number of windows
                at the end of the dataframe where an extraction of sub windows would be possible but no target row
                is available.
    :param window_start: relative distance between the current row and the start of the sub windows
    :param window_end: relative distance between the current row and the start of the sub windows
    :param npartitions: (dask parameter) the number of partitions used for the dataframe (used for parallelization).
                        According to stackoverflow the number should be a multiple of the number of processors
                        (default = 1xcpu_count)
    :return: a pandas dataframe containing all sub windows each with a unique window id (ascending numbers from 0 to #windows)
    """
    df_x = df_x.copy()
    # first entry is specified entry
    df_x.drop(df_x.head(start_index).index, inplace=True)

    df_x.drop(df_x.tail(lag).index, inplace=True)
    subwindows = make_windows(df_x.values, window_size, step_size=1, start_window_id=start_index+lag)
    sub_df_x_comp = pd.DataFrame(subwindows, columns=np.append(df_x.columns.values, 'window_id'))
    sub_df_x_comp = sub_df_x_comp.astype(df_x.dtypes)
    return sub_df_x_comp


def extract_sub_window(df_x, y, window, start_index, lag, fc_parameters=MinimalFCParameters(), n_jobs=-1):
    from tsfresh import extract_relevant_features
    window_start, window_end = window
    sub_df_x = get_rolling_timeseries(df_x, start_index, lag, window_end-window_start)
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    print('Remove non target values...')
    y = y.iloc[start_index + lag:]
    # y = y[y.index.isin(sub_df_x.window_id)]
    print('Extracting features...')
    features = extract_relevant_features(sub_df_x, y, column_id="window_id", column_sort="timestamp", column_value=None,
                                         default_fc_parameters=fc_parameters, n_jobs=n_jobs)
    # features = pd.concat([extracted_features], axis=1)
    features = features.add_suffix(f"_{window_start}_{window_end}")
    return features


def extract_sub_windows(df_x, df_y, window_array, lag, fc_parameters=MinimalFCParameters(), n_jobs=-1):
    # df_x = df_x.reset_index('timestamp')
    df_x['timestamp'] = list(range(len(df_x)))

    split_func = lambda x: list(map(int, x.split("-")))
    windows = np.array(list(map(split_func, window_array)))
    max_end = max(windows[:, 1])

    y = df_y.iloc[max_end + lag:]
    y = y.reset_index(drop=True)
    y.index.name = 'window_id'
    features = [
        extract_sub_window(df_x.copy(), y.copy(), window, max_end - (window[1] - window[0]), lag, fc_parameters, n_jobs)
        for window in windows]
    features = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'),
                      features)
