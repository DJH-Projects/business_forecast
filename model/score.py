# encooing = utf-8

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import traceback
import os
import numpy as np
import sys
import logging
import pandas as pd
from loguru import logger
import re
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import cpca
import warnings
warnings.filterwarnings('ignore')

##################################################################################
# 候选匹配策略
##################################################################################
def extract_other_address(sent='河北省邯郸市峰矿县义津镇北羊台村二街道'):
    '''
    县（区），镇（乡），村，街道
    抽取省市之外的地址特征，给候选进行二次判断
    '''
    rst = []
    p1 = r'[\u4e00-\u9fa5]{2}[县区]'
    p2 = r'[\u4e00-\u9fa5]{2}[镇乡]'
    p3 = r'[\u4e00-\u9fa5]{2}村'
    p4 = r'[\u4e00-\u9fa5]{2}街道'
    
    for p in [p1,p2,p3,p4]:
        g = re.search(p,sent)
        if g:
            rst.append(g.group(0))
        else:
            rst.append('')
    
    return rst


def extract_project_id(sent):
    '''
    遍历全部输入，得到有项目编号的例子，进而测试项目编号正则表达式
    '''
    prefix_lst =['项目编号','招标编号','采购编号','编号']
    for prefix in prefix_lst:
        if prefix in sent:
            pattern = r'{}[:：]?([\[\]0-9a-zA-Z\-_\u4e00-\u9fa5]+[0-9a-zA-Z]+)'.format(prefix)
            i = sent.index(prefix)
            sent = re.sub(r'\(|\)|\s','',sent[i:i+50])
            g = re.search(pattern,sent)
            if g:
                item = g.group(1)
                if 4 < len(item) < 30:
                    return item.strip()
    # 特殊形式（GXZC2019-G1-27190-KWZB）
    g = re.search(r'（([A-Za-z]+[0-9a-zA-Z\-]+)）',sent)
    if g:
        item = g.group(1)
        if 4 < len(item) < 30:
            return item.strip()
    return ''

def extract_company_name(sent):
    '''
    公司机构的名字
    '''
    pattern = r'[\u4e00-\u9fa5]+?((医|保健|研究)院|中心|局|厂|所|公司|集团|学(校|院)|(大|中|小)学|委员会|基地)|(招投?标|交易|采购)中心|事务所|研究院'
    g = re.search(pattern,sent)
    if g:
        item = g.group(0)
        if 4<=len(item)<=30:
            return item
    return ''

def extract_project_name(sent):
    '''
    项目名字
    '''
    pattern = r'[\u4e00-\u9fa5]{5,65}(项目|工程)'
    g = re.search(pattern,sent)
    if g:
        item = g.group(0)
        if 5<=len(item)<=65:
            return item
    return ''

def judge_other_address(senta, sentb):
    '''
    比较两个地址部分是否一样
    '''
    # logger.info('judge_other_address...')
    itema = extract_other_address(senta)
    itemb = extract_other_address(sentb)
    count = 4
    for i,j in zip(itema,itemb):
        if i == '' and j == '':
            count -= 1
            continue
        if i==j:
            count += 1
        else:
            count -= 1
    return count>=0
    
def judge_project_id(senta, sentb):
    '''
    title+content
    判断项目编号是否同时存在并且一致,严格相等
    '''
    # logger.info('judge_project_id...')
    itema = extract_project_id(senta)
    itemb = extract_project_id(sentb)
    if itema!='' and itemb!='':
        return itema == itemb
    return None

def judge_company(senta, sentb):
    '''
    title+content
    判断公司名字是否同时存在并且一致,严格相等
    '''
    # logger.info('judge_company...')
    itema = extract_company_name(senta)
    itemb = extract_company_name(sentb)
    if itema!='' and itemb!='':
        return itema == itemb
    return None


def judge_project_name(senta, sentb):
    '''
    title+content
    判断公司名字是否同时存在并且一致,严格相等
    '''
    # logger.info('judge_project_name...')
    itema = extract_project_name(senta)
    itemb = extract_project_name(sentb)
    if itema!='' and itemb!='':
        return itema == itemb
    return None

def judge_cpca(senta, sentb):
    '''
    判断省市区信息是否一致
    '''
    # logger.info('judge_cpca...')
    df = cpca.transform([senta, sentb],open_warning=False,cut=False, lookahead=3)
    for i, j in zip(df.loc[0][:-1], df.loc[1][:-1]):
        if i != j:
            return False
    return True
    
##################################################################################

'''
threshold = 0.9
candidate_threshold = 0.75

0.9260628465804066
0.9529555695631063
0.9260628465804066
0.9335073342165597

threshold = 0.9
candidate_threshold = 0.8
0.88909426987061
0.9417406295542843
0.88909426987061
0.9039560955231397

threshold = 0.9
candidate_threshold = 0.8

0.8853974121996303
0.9366245667926448
0.8853974121996303
0.9004302696256167
'''


if __name__ == "__main__":
    pattern = r'、|《|》|～|`|！|@|#|￥|%|…|&|（|）|；|;|×|—|-|=|\(|\)|>|<|\\|/|_|。|，|"|”|【|】|\[|\]|{|}|'
    pattern += r'正常|等|关于|设计|竞争性系统|项目|公告|.标|工标|单一|工程|合同|来源|失败|公告|流标|废标|终止|暂停|中止|更改|变更|更正|补遗|补充|澄清|延期|交易|结果|公示|成交|中选|中标|比选|比价|竞标|竞价|限制价|控制价|控价|限价|询价|询比|预公告|预公示|预审|抽签|选人|采购|邀请|需求|招标|磋商|谈判|竞争性磋商'

    threshold = 0.9
    candidate_threshold = 0.85

    df = pd.read_csv('../annotated_data.csv',sep='|')
    embed = hub.load("./universal-sentence-encoder-multilingual_2")

    y_true = df['label'].tolist()
    y_pred = []

    for row in df.values:
        if abs(len(row[0])-len(row[1])) > 20:
            y_pred.append(0)
            continue
        senta = row[0].strip()
        sentb = row[1].strip()
        senta = re.sub(pattern, '', senta)
        sentb = re.sub(pattern, '', sentb)
        veca = embed(senta)["outputs"]
        vecb = embed(sentb)["outputs"]
        score = np.inner(veca, vecb)[0]

        ################################################################
        # 1.长度规则过滤，低分过滤
        if abs(len(senta)-len(sentb)) > 20 or float(score) < candidate_threshold:
            y_pred.append(0)
            continue

        # 3.1 项目编号
        flag = judge_project_id(senta, sentb)
        if flag == True:
            y_pred.append(1)
            continue
        if flag == False:
            y_pred.append(0)
            continue

        # 2.达到阈值
        if float(score) >= threshold:
            y_pred.append(1)
            continue

        # 3. 候选判决,应该对title + content 处理
        # senta = title + content
        # sentb = title + content
        # 3.2 项目名称
        flag = judge_project_name(senta, sentb)
        if flag is not None and flag:
            y_pred.append(1)
            continue

        # 3.3 机构名字
        flag = judge_company(senta, sentb)
        if flag is not None and flag:
            y_pred.append(1)
            continue

        # 3.4 地址信息
        if judge_cpca(senta, sentb) and judge_other_address(senta, sentb):
            y_pred.append(1)
            continue
        y_pred.append(0)
        ################################################################
    print(accuracy_score(y_true, y_pred))

    # print(precision_score(y_true, y_pred, average='micro'))
    # print(recall_score(y_true, y_pred, average='micro'))
    # print(f1_score(y_true, y_pred, average='micro'))

    # print(precision_score(y_true, y_pred, average='macro'))
    # print(recall_score(y_true, y_pred, average='macro'))
    # print(f1_score(y_true, y_pred, average='macro'))

    print(precision_score(y_true, y_pred, average='weighted'))
    print(recall_score(y_true, y_pred, average='weighted'))
    print(f1_score(y_true, y_pred, average='weighted'))
