import pandas as pd
import os

import pandas as pd
import os
import re

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
    if '项目编号' not in sent:
        return ''
    pattern = r'项目编号[:：]?([\[\]0-9a-zA-Z\-_\u4e00-\u9fa5]+[0-9a-zA-Z]+)'
    i = sent.index('项目编号')
    sent = re.sub(r'\(|\)|\s','',sent[i:i+50])
    g = re.search(pattern,sent)
    if g:
        item = g.group(1)
        if 4 < len(item) < 25:
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
if __name__ == "__main__":
    data_folder = '../raw_data/'
    input_folder = data_folder + 'input'
    history_folder = data_folder + 'history'
    # ['ID', 'Title', 'Content', 'Type']
    input_format = ['ID', 'Title', 'Content', 'Type']
    # ["Test_ID", "Test_Title", "Test_Content", "Test_type", "Similar_ID"]
    output_format = ["Test_ID", "Test_Title",
                     "Test_Content", "Test_type", "Similar_ID"]

    # input_files = get_input_files()

    # input_paths = [os.path.join(input_folder, f) for f in input_files]
    # df = get_datafrme_from_files(
    #     input_paths, types=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(df.info())
