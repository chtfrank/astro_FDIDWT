
import os
import numpy as np
from sklearn.preprocessing import scale



def load_npy_data(npy_dir, split_num, delete_attris_list=None):
    split_npy_dir = os.path.join(npy_dir, 'split_'+str(split_num))
    trainX = np.load(os.path.join(split_npy_dir, 'train_data.npy'))
    validationX = np.load(os.path.join(split_npy_dir, 'validation_data.npy'))
    testX = np.load(os.path.join(split_npy_dir, 'test_data.npy'))
    train_sourcename = np.load(os.path.join(split_npy_dir, 'train_sourcename.npy'))
    validation_sourcename = np.load(os.path.join(split_npy_dir, 'validation_sourcename.npy'))
    test_sourcename = np.load(os.path.join(split_npy_dir, 'test_sourcename.npy'))
    trainY = np.load(os.path.join(split_npy_dir, 'train_labels.npy'))
    validationY = np.load(os.path.join(split_npy_dir, 'validation_labels.npy'))
    testY = np.load(os.path.join(split_npy_dir, 'test_labels.npy'))

    predict_data = np.load(os.path.join(npy_dir, 'predict_data.npy'))
    predict_sourcename = np.load(os.path.join(npy_dir, 'predict_sourcename.npy'))
    header_name = np.load(os.path.join(npy_dir, 'header_name.npy'))

    if isinstance(delete_attris_list, list):
        header_name_list = list(header_name)
        delete_attris_index = []
        for da in delete_attris_list:
            indx = header_name_list.index(da)
            delete_attris_index.append(indx)
        trainX = np.delete(trainX, delete_attris_index, axis=-1)
        validationX = np.delete(validationX, delete_attris_index, axis=-1)
        testX = np.delete(testX, delete_attris_index, axis=-1)
        predict_data = np.delete(predict_data, delete_attris_index, axis=-1)
        header_name = np.delete(header_name, delete_attris_index, axis=-1)
    # print(np.shape(trainX), np.shape(validationX), np.shape(testX),
    #       np.shape(trainY), np.shape(validationY), np.shape(testY),
    #       np.shape(train_sourcename), np.shape(validation_sourcename), np.shape(test_sourcename),
    #       np.shape(predict_data), np.shape(predict_sourcename), np.shape(header_name))

    return trainX, validationX, testX, \
           trainY, validationY, testY, \
           train_sourcename, validation_sourcename, test_sourcename, \
           predict_data, predict_sourcename, header_name




def load_FDIDWT_data(txt_path, do_scale=True):
    xdata = []
    with open(txt_path, 'r') as fp:
        for line in fp.readlines():
            words = line.strip().split(' ')
            xdata.append(list(map(float, words)))
    xdata = np.array(xdata)
    if do_scale:
        xdata = scale(xdata, axis=0)
    return xdata



def return_FDIDWT_files(data_dir):
    file_path_list = []
    for level in [1, 3]:
        level_dir = os.path.join(data_dir, 'level' + str(level))
        for outputDim in range(1, 14):
            for db in range(1, 8):
                for split in ['train', 'validation', 'test', 'un']:
                    filename = 'outputDim' + str(outputDim) + '_db' + str(db) + '_' + split + '.txt'
                    file_path = os.path.join(level_dir, filename)
                    if os.path.exists(file_path):
                        file_path_list.append(file_path)
    return file_path_list
    

def return_all_files(root_path, extension):
    all_file_list = []
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.split('.')[-1] == extension:
                file_path = os.path.join(root, f)
                all_file_list.append(file_path)
    return all_file_list
