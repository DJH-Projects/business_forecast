#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
from xls2csv_noformat import xls2csv
from tqdm import tqdm

count = 0
for root, dirs, files in os.walk('./xls'):
    for d in dirs:
        if not os.path.isdir(os.path.join(root,d)):
            continue
        for f in tqdm(os.listdir(os.path.join(root,d))):
            if f.endswith('xlsx'):
                count += 1
                fname = f.split('.')[0]
                fpath = os.path.join('./csv',fname+'.csv')
                try:
                    fp = open(fpath, 'w+', encoding="utf-8", newline="")
                    kwargs = {
                        'sheetid': 1,
                        'delim': ',',
                        'sheetdelimiter': '',
                        'encoding': 'utf-8',
                    }
                    
                    xls2csv(os.path.join(root,d,f), fp, **kwargs)
                except Exception as e:
                    print(fpath)
                    continue
print(count)



