
# coding: utf-8

import pandas as pd
import os
import re
from tqdm import tqdm

folder = './csv/'
bidding_id_dict = {
    '招标预告': '1',
    '招标公告': '2',
    '更正公告': '3',
    '中标公告': '4',
    '废标公告': '5',
    '其他公告': '6',
    '中标预告': '7',
    '流标公告': '8',
    '澄清公告': '9'
}


def convert(x):
    if x in bidding_id_dict:
        return bidding_id_dict[x]
    return bidding_id_dict['其他公告']


def comma_problem(x):
    return re.sub(r',', '，', str(x))


if __name__ == "__main__":
    last_col = 0
    for f in tqdm(os.listdir(folder)):
        if f.endswith('csv'):
            fp = os.path.join(folder, f)
            df = pd.read_csv(fp, header=None)
            df = df.dropna(axis=1, how='all')
            df = df.fillna('')
            for i in range(1, df.shape[1]):
                s = str(df.iloc[0, -i])
                if s.strip() != '':
                    last_col = -i
                    break
            
            if last_col < -1:
                df = df.iloc[:, last_col-3:last_col+1]
            elif last_col == -1:
                df = df.iloc[:, -4:]
            else:
                print(fp)
                continue
            if not f.startswith('45'):
                df.iloc[:, -1] = df.iloc[:, -1].apply(convert)
            df.iloc[:, -3].apply(comma_problem)  # Title
            df.iloc[:, -2].apply(comma_problem)  # Content
            df.to_csv(os.path.join('csv', f),
                      index=False, header=False, sep='|')
