# encoding = utf-8
import traceback
import cpca
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
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

import warnings
warnings.filterwarnings('ignore')

# 控制tensorflow输出信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
##################################################################################


#候选阈值0.7-0.9,需要judge_cpca
#阈值0.9
threshold = 0.9
candidate_threshold = 0.7

def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        logger.info('[Function: {name} start...]'.format(
            name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        logger.info('[Function: {name} finished, spent time: {time:.2f}s]'.format(
            name=function.__name__, time=t1 - t0))
        return result
    return function_timer
##################################################################################


@func_timer
def get_new_annotated_data():
    file_path = '../annotated_data.csv'
    new_file_path = '../new_annotated_data.csv'
    new_data = []

    yes_thresholds = []
    no_thresholds = []

    df = pd.read_csv(file_path, header=None)

    print(df.shape)

    for row in df.values[1:]:
        senta = row[0].strip()
        sentb = row[1].strip()
        senta = re.sub(pattern, '', senta)
        sentb = re.sub(pattern, '', sentb)
        veca = embed(senta)["outputs"]
        vecb = embed(sentb)["outputs"]
        score = np.inner(veca, vecb)[0]
        if str(row[-1]) == '1':
            yes_thresholds.append(score)
        else:
            if score > candidate_threshold and judge_cpca(senta, sentb):
                row[-1] = '1'
            no_thresholds.append(score)
        new_data.append(row.tolist())
    if len(yes_thresholds) == 0 or len(no_thresholds) == 0:
        print("thresholds == []")
        return
    
    df = pd.DataFrame(new_data)
    df.columns = ['sentence1','sentence2','label']
    df.to_csv(new_file_path,index=False)

    print('yes_thresholds = avg:{} max:{} min:{}'.format(
        sum(yes_thresholds)/len(yes_thresholds), max(yes_thresholds), min(yes_thresholds)))
    print('no_thresholds = avg:{} max:{} min:{}'.format(
        sum(no_thresholds)/len(no_thresholds), max(no_thresholds), min(no_thresholds)))


def judge_cpca(senta, sentb):
    '''
    判断省市区信息是否一致
    '''
    df = cpca.transform([senta, sentb],open_warning=False,cut=False, lookahead=3)
    for i, j in zip(df.loc[0][:-1], df.loc[1][:-1]):
        if i != j:
            return False
    return True


@func_timer
def main():
    file_path = '../new_annotated_data.csv'

    yes_thresholds = []
    no_thresholds = []

    df = pd.read_csv(file_path, header=None)

    print(df.shape)

    for row in df.values[1:]:
        senta = row[0].strip()
        sentb = row[1].strip()
        senta = re.sub(pattern, '', senta)
        sentb = re.sub(pattern, '', sentb)
        veca = embed(senta)["outputs"]
        vecb = embed(sentb)["outputs"]
        score = np.inner(veca, vecb)[0]
        if str(row[-1]) == '1':
            yes_thresholds.append(score)
        else:
            no_thresholds.append(score)

    if len(yes_thresholds) == 0 or len(no_thresholds) == 0:
        print("thresholds == []")
        return

    print('yes_thresholds = avg:{} max:{} min:{}'.format(
        sum(yes_thresholds)/len(yes_thresholds), max(yes_thresholds), min(yes_thresholds)))
    print('no_thresholds = avg:{} max:{} min:{}'.format(
        sum(no_thresholds)/len(no_thresholds), max(no_thresholds), min(no_thresholds)))




if __name__ == "__main__":
    pattern = r'、|《|》|～|`|！|@|#|￥|%|…|&|（|）|；|;|×|—|-|=|\(|\)|>|<|\\|/|_|。|，|"|”|【|】|\[|\]|{|}|'
    pattern += r'正常|等|关于|设计|竞争性系统|项目|公告|.标|工标|单一|工程|合同|来源|失败|公告|流标|废标|终止|暂停|中止|更改|变更|更正|补遗|补充|澄清|延期|交易|结果|公示|成交|中选|中标|比选|比价|竞标|竞价|限制价|控制价|控价|限价|询价|询比|预公告|预公示|预审|抽签|选人|采购|邀请|需求|招标|磋商|谈判|竞争性磋商'
    embed = hub.load("./universal-sentence-encoder-multilingual_2")
    # get_new_annotated_data()
    main()
