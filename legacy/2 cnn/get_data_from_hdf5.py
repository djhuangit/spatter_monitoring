import cv2 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

def bytes_decode(io_buf):
    return cv2.imdecode(np.frombuffer(io_buf, np.uint8), -1)

def reshape_square(img_data, desired_size=300):
    old_size = img_data.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    img_data = cv2.resize(img_data, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img_data, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_img

def get_data_from_hdf5(file, percent, label):
    '''
    retrive data from spatter hdf5 files based on a defined percent,
    then decode them to become images

    ARGS:
    file: .hdf5 file path
    percent: percentage of data to be retrived
    label: label of the dataset
    '''

    #timer
    start = time.time()

    # total_data_size = 4000000
    # required_size = total_data_size * percent
    # print("amount of data to be retrived: ", required_size)

    # get number of dataset in hdf5 file
    hdf = h5py.File(file, 'r')
    ls = list(hdf.keys())
    print("retriving file:", file)
    # print(ls)
    num_dataset = len(ls)
    num_dataset_to_retrive = max(1, int(num_dataset * percent))
    print("number of dataset to be retrived:", num_dataset_to_retrive)
    random_selection = np.random.randint(low=0, high=num_dataset, size=num_dataset_to_retrive)
    print("random index", random_selection)

    data = []
    for num in random_selection:
        print("randomly selected dataset", ls[num])
        data.append(hdf[ls[num]])

    data = np.array(data).reshape(-1)
    # data = np.vstack(data)
    print("size of data:", data.shape)
    # print("size of dataset:", data[0].shape)
    # print("data:", data[0])
    data_decoded = [reshape_square(bytes_decode(x)) for x in data]
    # plt.imshow(data_decoded[0])
    # plt.show()

    time_taken = time.time() - start
    print("time taken for data retrival:", time_taken)

    return data_decoded

def main():
    start = time.time()

    # define files location and output path
    file_paths = [(r"E:\spatter backup\build_832\raspi1_right\capture_20201118_165553\20201118_165553.h5", 0),
                  (r"E:\spatter backup\build_832\raspi1_right\capture_20201120_212900\20201120_212900.h5", 0),
                  (r"E:\spatter backup\build_832\raspi2_left\capture_20201118_165622\20201118_165622.h5", 1),
                  (r"E:\spatter backup\build_832\raspi2_left\capture_20201120_232125\20201120_232125.h5", 1),
                 ] 
    
    output = r"D:\spatter\build_832_ti\40k_data_color_build_832.npy"
    
    # define percentage of data retrived
    percentage = 0.01

    data = []
    for file, label in file_paths:

        data.append(get_data_from_hdf5(file, percent=percentage, label=label))
    
    data = np.asarray(data)
    print(data[0].shape)
    data = np.asarray(data).reshape(-1, 300, 300, 3)
    print("final data size:", data.shape)

    np.save(output, data)

    time_taken = time.time() - start
    print("total time elapsed: ", time_taken)



if __name__ == "__main__":
    main()
