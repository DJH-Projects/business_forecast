# encoding = 'utf-8
'''
将输入文件统一拷贝到工程的输入目录下
python copy_to_data_folder.py 
'''

import os
import shutil

#################################
src_folder = '../test_folder/in'
trg_folder = './raw_data/'
extension = '.csv'
#################################


def main():

    if not os.path.exists(src_folder):
        print('{} not exists'.format(src_folder))
        return

    if not os.path.exists(trg_folder):
        print("creating {}".format(trg_folder))
        os.mkdir(trg_folder)

    count = 0
    for root, dirs, files in os.walk(src_folder):
        for name in files:
            if name.endswith(extension):
                src = os.path.join(root, name)
                trg = os.path.join(trg_folder, name)
                if os.path.exists(trg):
                    continue
                print('copying {} -> {}'.format(src, trg))
                shutil.copy(src, trg)
                count += 1
    print("final count = {}".format(count))


if __name__ == "__main__":
    main()
