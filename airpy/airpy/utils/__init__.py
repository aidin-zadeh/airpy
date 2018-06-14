from pandas import DataFrame
from pandas import concat
from datetime import datetime


def parse_datetime(x):
    return datetime.strptime(x, "%Y %m %d %H")


def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
    """Return

    :param data:
    :param n_in:
    :param n_out:
    :param dropnan:
    :return:
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg


def create_difference_sequence(sequence, interval=1):
    """Create difference sequence given input 'sequence' and 'interval'.

    :param sequence:
    :param interval:
    :return:
    """

    diff = list()
    for i in range(interval, len(sequence)):
        value = sequence[i] - sequence[i - interval]
        diff.append(value)
    return pd.Series(diff)


def rescale(x, scale_range=(0,1)):
    scaler = MinMaxScaler(feature_range=scale_range)
    return scaler(x), scaler

