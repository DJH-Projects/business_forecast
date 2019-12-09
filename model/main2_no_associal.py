# encooing = utf-8
# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

from tqdm import tqdm
import cpca
import traceback
import re
import os
import numpy as np
import sys
import shutil
import logging
import time
import pandas as pd
from datetime import datetime
from functools import wraps
from pprint import pprint
from loguru import logger
import numpy as np
import multiprocessing

import warnings
warnings.filterwarnings('ignore')

# ['ID', 'Title', 'Content', 'Type']
input_format = ['ID', 'Title', 'Content', 'Type']

bidding_id_dict = {
    '1': '招标预告',
    '2': '招标公告',
    '3': '更正公告',
    '4': '中标公告',
    '5': '废标公告',
    '6': '其他公告',
    '7': '中标预告',
    '8': '流标公告',
    '9': '澄清公告',
    'nan':''
}


# ["Test_ID", "Test_Title", "Test_Content", "Test_type", "Similar_ID"]
# ["Test_ID", "Test_Title", "Test_Content", "Test_type", "Similar_ID","Similar_Title","Similar_Content","Similar_Type"]
output_format = ["Test_ID", "Test_Title",
                 "Test_Content", "Test_type", "Similar_ID", "Similar_Title", "Similar_Content", "Similar_Type"]

# 结果目录
output_folder = '../output_associal_6v120_2-2'

# 全部输入目录
input_folder = '../raw_data/input'
seprator = '|'

# 无关联结果目录
result_folder = '../output_no_associal'


def get_datafrme_from_files(file_paths, types=[]):
    '''
    将多个文件的内容读入一个df
    '''
    if types == []:
        return None
    df = pd.DataFrame(columns=input_format)
    for fp in file_paths:
        try:
            t = pd.read_csv(fp, error_bad_lines=False, sep=seprator)  # 注意参数情况
        except Exception as e:
            logger.info('read_csv cause format error in {}'.format(fp))
            continue
        t.columns = input_format
        df = pd.concat([df, t], axis=0)
    if df.shape == (0, 4):
        return None
    df = df[df[input_format[-1]].isin(types)]
    return df.dropna(axis=0, how='any')


def get_ids_from_xls(folder):
    ids = []
    for f in os.listdir(folder):
        df = pd.read_excel(os.path.join(folder, f))
        ids.extend(df['Test_ID'].tolist())
    return list(set(ids))

def map_type(num):
    return bidding_id_dict[str(num)]

def main(f):
    logger.info('processing {} ...'.format(f))
    input_df = get_datafrme_from_files(
        [os.path.join(input_folder, f)], [1, 3, 4, 5, 6, 7, 8, 9])
    logger.info(input_df.shape)
    input_df = input_df[input_df['ID'].isin(output_ids)]
    input_df.columns = ["Test_ID", "Test_Title",
                        "Test_Content", "Test_type"]
    for col in ["Similar_ID", "Similar_Title", "Similar_Content", "Similar_Type"]:
        input_df[col] = ''

    fp = os.path.join(result_folder, f.split('.')[0] + '.xls')
    input_df['Test_type'] = df['Test_type'].map(map_type)
    input_df['Similar_Type'] = df['Similar_Type'].map(map_type)
    input_df.to_excel(fp, index=False, encoding='utf-8')
    logger.info('saving as  {} ...'.format(fp))


if __name__ == "__main__":
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    output_ids = get_ids_from_xls(output_folder)
    pool = multiprocessing.Pool(os.cpu_count()*2)
    pool.map(main, os.listdir(input_folder))
