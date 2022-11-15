import pandas as pd

_nano_to_sec = 1e09


def custom_timestamp_parser(custom_timestamp):
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    timestamp = pd.to_datetime(custom_timestamp, format='%Y/%m/%d %H:%M:%S.%f', errors='ignore')
    datetime64 = timestamp.to_datetime64()
    return datetime64.astype('float64') / _nano_to_sec
