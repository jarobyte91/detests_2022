import pandas as pd

def target_names():
    return ['xenophobia',  'suffering', 'economic', 'migration', 'culture', 'benefits', 'health', 'security', 'dehumanisation', 'others']

def get_y():
    data_path = '/lustre06/project/6001735/detests_2022/data/train.csv'
    df = pd.read_csv(data_path)
    y = df[target_names()]
    return y

def get_texts():
    data_path = '/lustre06/project/6001735/detests_2022/data/train.csv'
    df = pd.read_csv(data_path)
    return list(df["sentence"])