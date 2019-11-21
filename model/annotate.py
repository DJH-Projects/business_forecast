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
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

import threading
from multiprocessing import pool

import warnings
warnings.filterwarnings('ignore')

# 控制tensorflow输出信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

##################################################################################

bidding_id_dict = {
    '招标预告':'1',
    '招标公告':'2',
    '更正公告':'3',
    '中标公告':'4',
    '废标公告':'5',
    '其他公告':'6',
    '中标预告':'7',
    '流标公告':'8',
    '澄清公告':'9'
}

##########################################################
##########################################################
'''
模型认为绝对相似的>=postive_threshold的应该是关联的
模型任务绝对不相似的<=negtive_threshold的应该是不关联的
'''

annotated_file = '../annotated_all.csv'
annotated_pos = '../annotated_pos.csv'
annotated_neg = '../annotated_neg.csv'
annotated_format = ['sentence1','sentence2','label']

positive_size = 50000
negtive_size = 20000
count_pos = 0
count_neg = 0

postive_threshold = 0.95
negtive_threshold = 0.5

mode = 'w'
if os.path.exists(annotated_pos):
    mode = 'a'
if os.path.exists(annotated_neg):
    mode = 'a'
fp_all = open(annotated_file,mode,encoding='utf-8')
fp_pos = open(annotated_pos,mode,encoding='utf-8')
fp_neg = open(annotated_neg,mode,encoding='utf-8')

fp_all.write('|'.join(annotated_format))
fp_all.write('\n')
fp_pos.write('|'.join(annotated_format))
fp_pos.write('\n')
fp_neg.write('|'.join(annotated_format))
fp_neg.write('\n')

##########################################################
##########################################################


# 纯数字遍历测试：600000*20000约6-7分钟
# 目标：1小时完成600000*20000关联 # 文件个数,约80-200个文件
HISTORY_WINDOW_SIZE = 100  
INPUT_WINDWO_SIZE = 4  
SLEEP_DELAY =  HISTORY_WINDOW_SIZE/50 # 50,100,150,200
HISTORY_BATCH_SIZE = 2  # 取决于电脑性能，本机8G内存 <=3
INPUT_BATCH_SIZE = 2  # 取决于电脑性能，本机8G内存 <=2

# 设置输入文件和历史文件所在的目录
# raw_data
# 20209v404956_speed_test_raw_data
# 20209v604673_speed_test_raw_data
# 22219v806645_speed_test_raw_data
data_folder = '../raw_data/'
input_folder = data_folder + 'input'
history_folder = data_folder + 'history'
output_folder = '../output_{}v{}_{}-{}/'.format(INPUT_WINDWO_SIZE,HISTORY_WINDOW_SIZE,INPUT_BATCH_SIZE,HISTORY_BATCH_SIZE)
error_instance_file = '../error_files.txt'

# ['ID', 'Title', 'Content', 'Type']
input_format = ['ID', 'Title', 'Content', 'Type']

# ["Test_ID", "Test_Title", "Test_Content", "Test_type", "Similar_ID"]
output_format = ["Test_ID", "Test_Title",
                 "Test_Content", "Test_type", "Similar_ID"]

# 预处理文本
pattern = r'、|《|》|～|`|！|@|#|￥|%|…|&|（|）|；|;|×|—|-|=|\(|\)|>|<|\\|/|_|。|，|"|”|【|】|\[|\]|{|}|'
pattern += r'正常|等|关于|设计|竞争性系统|项目|公告|.标|工标|单一|工程|合同|来源|失败|公告|流标|废标|终止|暂停|中止|更改|变更|更正|补遗|补充|澄清|延期|交易|结果|公示|成交|中选|中标|比选|比价|竞标|竞价|限制价|控制价|控价|限价|询价|询比|预公告|预公示|预审|抽签|选人|采购|邀请|需求|招标|磋商|谈判|竞争性磋商'

# 阈值
threshold = 0.95
candidate_threshold = 0.9

'''
(542, 3)
yes_thresholds = avg:[0.8950597] max:[1.0000002] min:[0.38766828]
no_thresholds = avg:[0.7557879] max:[0.9912609] min:[0.20290779]
'''

local_env = {
    "HISTORY_BATCH_SIZE": HISTORY_WINDOW_SIZE,
    "INPUT_BATCH_SIZE": INPUT_WINDWO_SIZE,
    "DATA_FOLDER": data_folder,
    "INPUT_FOLDER": input_folder,
    "OUTPUT_FOLDER": output_folder,
    "HISTORY_FOLDER": history_folder,
    "INPUT_FORMAT": input_format,
    "OUTPUT_FORMAT": output_format,
    "ERROR_INSTANCE_FILE": error_instance_file,
    "THRESHOLD": threshold,
    "CANDIDATE_THRESHOLD":candidate_threshold
}
##################################################################################


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


def get_datafrme_from_files(file_paths, types=[]):
    '''
    将多个文件的内容读入一个df
    '''
    if types == []:
        return None
    df = pd.DataFrame(columns=input_format)
    for fp in file_paths:
        try:
            t = pd.read_csv(fp,error_bad_lines=False,sep='|')  # 注意参数情况
        except Exception as e:
            logger.info('read_csv cause format error in {}'.format(fp))
            continue
        t.columns = input_format
        df = pd.concat([df, t], axis=0)
    if df.shape == (0, 4):
        return None
    df = df[df[input_format[-1]].isin(types)]
    return df.dropna(axis=0,how='any')
##################################################################################


def save_dataframe_to_file(df, file_name):
    '''
    保存dataframe到输出文件
    '''
    if not os.path.exists(output_folder):
        logger.info('creating "{}" folder.'.format(output_folder))
        os.mkdir(output_folder)

    logger.info("saving as {}".format(file_name))
    file_path = os.path.join(output_folder, file_name)
    df.to_csv(file_path, sep='|', index=False, encoding='utf-8')

##################################################################################


def check_folder_structure():
    '''
    检查目录结构
    '''
    folders = [data_folder, input_folder, history_folder]
    for folder in folders:
        logger.info('checking {}'.format(folder))
        if not (os.path.exists(folder) and os.path.isdir(folder)):
            logger.error('No folder "{}" '.format(folder))
            return False
    else:
        return True

##################################################################################


def get_history_files():
    '''
    历史数据默认取时间靠后的
    '''
    files = sorted(os.listdir(history_folder))[-HISTORY_WINDOW_SIZE:]
    if files == []:
        logger.info('There is no file in "{}"'.format(history_folder))
    return files

##################################################################################


def get_input_files():
    '''
    输入数据默认取时间靠前的
    '''
    files = sorted(os.listdir(input_folder))[:INPUT_WINDWO_SIZE]
    if files == []:
        logger.info('There is no file in "{}"'.format(input_folder))
    return files

##################################################################################


def finish_input_files(files):
    '''
    完成的输入文件转移到历史文件目录
    '''
    if files == []:
        return True
    count = 0
    for file in files:
        shutil.move(os.path.join(input_folder, file),
                    os.path.join(history_folder, file))
        count += 1
    return len(files) == count

##################################################################################
@func_timer
def main():
    # 基本信息
    if not check_folder_structure():
        sys.exit(1)
    logger.info('current working settings: ')
    pprint(local_env)
    print('\n')

    # 停顿
    # time.sleep(1)

    # 监控剩余处理的文件数目
    num_left_files = len(os.listdir(input_folder))

    if num_left_files == 0:
        logger.info('all input has been finished.')
        return

    epoch = 0
    global count_pos
    global count_neg
    while num_left_files != 0:
        
        if count_pos > positive_size and count_neg > negtive_size:
            logger.info('annotate successfully... ')
            fp_all.close()
            fp_pos.close()
            fp_neg.close()
            sys.exit(0)

        # 获取历史文件名字
        history_files = get_history_files()
        if history_files == []:
            sys.exit('EXIT: no history files')
        # 获取测试文件名字
        input_files = get_input_files()
        if input_files == []:
            logger.info('all input has been finished.')
            return

        # 控制打印频率
        if epoch % 1 == 0:
            logger.info('='*10)
            logger.info('There are {} left.'.format(num_left_files))

        ##########################################################
        logger.info('input_files = {}, history_files = {}'.format(
            len(input_files), len(history_files)))

        input_ = list(range(0, len(input_files), INPUT_BATCH_SIZE))

        for i,input_start in enumerate(input_):

            batch_input_files = input_files[input_start:input_start +
                                            INPUT_BATCH_SIZE]
            logger.info('input rate = {}/{}'.format(i+1,len(input_)))
            for history_start in tqdm(list(range(0, len(history_files), HISTORY_BATCH_SIZE))):
            
                if count_pos > positive_size and count_neg > negtive_size:
                    logger.info('annotate successfully... ')
                    fp_pos.close()
                    fp_neg.close()
                    fp_all.close()
                    sys.exit(0)
                
                batch_history_files = history_files[history_start:history_start +
                                                    HISTORY_BATCH_SIZE]

                thread = threading.Thread(thread_process(batch_input_files, batch_history_files))
                thread.start()
                
                time.sleep(SLEEP_DELAY)#暂停一下主线程再进行下次迭代
            
            # update input -> history
            logger.info('moving input files --> history')
            finish_input_files(batch_input_files)
            
        ##########################################################

        # update num_left_files if input folder was updated
        num_left_files = len(os.listdir(input_folder))
        epoch += 1

    logger.info('all input has been finished.')

##################################################################################
@func_timer
def thread_process(batch_input_files, batch_history_files):
    batch_input_paths = [os.path.join(input_folder, f) for f in batch_input_files]
    batch_history_paths = [os.path.join(history_folder, f) for f in batch_history_files]

    logger.info('batch_input_paths = {}, batch_history_paths = {}'.format(
        len(batch_input_paths), len(batch_history_paths)))

    # read dataframe
    df_history_bidding = get_datafrme_from_files(
        batch_history_paths, types=[2])  # 历史的招标公告
    df_input_other = get_datafrme_from_files(
        batch_input_paths, types=[1, 3, 4, 5, 6, 7, 8, 9])  # 新输入的非招标类型的公告

    # check dataframe is none
    if df_history_bidding is None or df_input_other is None:
        logger.error('dataframe is None.')
        return

    rst_df = process_one_by_one(df_input_other, df_history_bidding)

#     if rst_df is None or len(rst_df) == 0:
#         logger.error(
#             'No related found.')
#         fp_error_instance.write("{}|{}\n".format(
#             ','.join(batch_input_paths), ','.join(batch_history_paths)))
#         return

    # save output
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S.csv")
    #save_dataframe_to_file(rst_df, file_name)


##################################################################################
@func_timer
def process_one_by_one(df_input_other, df_history_bidding):
    '''
    处理历史数据和新数据的关联核心代码，返回格式化的关联结果
    df_history_bidding：历史招标公告
    df_input_other：测试的公告
    '''
    logger.info("input_shape = {}, history_shape = {}".format(
        df_input_other.shape, df_history_bidding.shape))
    # ['ID', 'Title', 'Content', 'Type']
    # ["Test_ID", "Test_Title","Test_Content", "Test_type", "Similar_ID"]

    others = df_input_other['Title'].tolist()
    historys = df_history_bidding['Title'].tolist()

    logger.info('generating embedding ...')
    try:
        input_result = embed(others)["outputs"]
        history_result = embed(historys)["outputs"]
    except Exception as e:
        logger.error('Exception')
        traceback.print_exc()
        return None

    logger.info('processing similarity_matrix ...')
    similarity_matrix = np.inner(input_result, history_result)

    del input_result
    del history_result
    global count_pos
    global count_neg

    result = []
    logger.info('reformating result dataframe ...')
    for i, row in enumerate(similarity_matrix):
        line = list(df_input_other.iloc[i])
        bidding_index = np.argmax(row)
        bidding_score = row[bidding_index]
        src_title = others[i]
        trg_title = historys[bidding_index]

        if bidding_index == i:
            continue

        # 下一步考虑预处理title（主要）和content
        if len(src_title) <= 5 and len(trg_title) <=5:
            continue

        if bidding_score <= negtive_threshold and count_neg < negtive_size:#非关联标注
            fp_all.write('{}|{}|{}\n'.format(src_title,trg_title,0))
            fp_neg.write('{}|{}|{}\n'.format(src_title,trg_title,0))
            fp_neg.flush()
            fp_all.flush()
            count_neg += 1
            continue
                        
        if count_pos >= positive_size and count_neg >= negtive_size:#标注结束
            logger.info('annotate successfully... ')
            fp_all.close()
            fp_pos.close()
            fp_neg.close()
            fp_all.close()
            sys.exit(0)
        
        # 1.长度规则过滤，低分过滤
        if abs(len(src_title)-len(trg_title)) > 20 or float(bidding_score) < candidate_threshold:
            fp_all.write('{}|{}|{}\n'.format(src_title,trg_title,0))
            fp_neg.write('{}|{}|{}\n'.format(src_title,trg_title,0))
            fp_neg.flush()
            fp_all.flush()
            count_neg += 1
            continue
        
        # 1. 项目编号
        senta = src_title + df_input_other['Content'].tolist()[i]
        sentb = trg_title + df_history_bidding['Content'].tolist()[bidding_index]
        
        flag = judge_project_id(src_title,trg_title)
        if flag == True:
            fp_all.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.flush()
            fp_all.flush()
            count_pos += 1
            continue

        if flag == False:
            continue
        
        if  bidding_score >= postive_threshold and count_pos < positive_size:
            fp_all.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.flush()
            fp_all.flush()
            count_pos += 1
            continue

        if bidding_score < candidate_threshold:
            continue

        # 3.2 项目名称
        flag = judge_project_name(senta,sentb)
        if flag is not None and flag:
            fp_all.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.flush()
            fp_all.flush()
            count_pos += 1
            continue

        # 3.3 机构名字
        flag = judge_company(senta,sentb)
        if flag is not None and flag:
            fp_all.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.flush()
            fp_all.flush()
            count_pos += 1
            continue
        # 3.4 地址信息
        if judge_cpca(senta,sentb) and judge_other_address(senta,sentb):
            fp_all.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.write('{}|{}|{}\n'.format(src_title,trg_title,1))
            fp_pos.flush()
            fp_all.flush()
            count_pos += 1
            continue
        
    del similarity_matrix

    return pd.DataFrame(result, columns=output_format + ['O'])

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

if __name__ == '__main__':
    fp_error_instance = open(error_instance_file, 'w', encoding='utf-8')
    logger.info('loading model ...')
    try:
        embed = hub.load("./universal-sentence-encoder-multilingual_2")
    except Exception as e:
        logger.error('Exception')
        traceback.print_exc()
        sys.exit(1)
    logger.info('current working path: {}\n'.format(os.getcwd()))
    try:
        poo = pool.Pool()
        main()
    except Exception as e:
        logger.error('Exception')
        traceback.print_exc()
    finally:
        fp_error_instance.close()
        fp_pos.close()
        fp_neg.close()
        fp_all.close()
