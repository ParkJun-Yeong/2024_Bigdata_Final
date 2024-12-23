import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


list_path = "/mnt/ssd01_250gb/juny/vfss/SAMed/lists/lists_vfss"
train_file = open(os.path.join(list_path, 'train.txt'), 'w')
valid_file = open(os.path.join(list_path, 'valid.txt'), 'w')
test_file = open(os.path.join(list_path, 'test.txt'), 'w')

all_files_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/experiment/train"
all_files = os.listdir(all_files_path)

train, test = train_test_split(all_files, test_size=0.15, shuffle=True)
test, valid = train_test_split(test, test_size=0.3, shuffle=True)


print("train valid test: ", len(train), len(valid), len(test))

for data, file in tqdm(zip([train, valid, test], [train_file, valid_file, test_file])):
    for t in data:
        print(t.replace('.npz', ''), file=file)

train_file.close()
valid_file.close()
test_file.close()