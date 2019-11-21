import os

#20209v404956_speed_test_raw_data

#20209v604673_speed_test_raw_data

#22219v806645_speed_test_raw_data

_speed_test_raw_data = './20209v604673_speed_test_raw_data/'

history_folder = _speed_test_raw_data+'history'
input_folder = _speed_test_raw_data+'input'


def count_folder(folder):
    count = 0
    for f in os.listdir(folder):
        count += count_file(os.path.join(folder, f))
    return count

def count_file(file_path):
    count = 0
    for i, _ in enumerate(open(file_path,'r')):
        count += 1
    return count


if __name__ == "__main__":
    print(count_folder('output_20209v404956_speed_test_raw_data_6v120_2-3'))
#     print('history:')
#     print(count_folder(history_folder))
#     print('input:')
#     print(count_folder(input_folder))

