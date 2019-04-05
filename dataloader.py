from glob import glob
from multiprocessing import Pool
from math import ceil
from utils import *
from sklearn.model_selection import train_test_split
import time
import scipy


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.mean = None
        self.std = None
        self.noisy_train, self.noisy_test, self.gt_train, self.gt_test, self.width, self.height = self.load_data()

    def load_data(self):
        files = None
        if not self.config.num_files_to_load:
            files = sorted(glob(self.config.dataset_dir))
        else:
            files = sorted(glob(self.config.dataset_dir))[:self.config.num_files_to_load]
        print("number of total files to be loaded: ", len(files))

        noisy_list = [file for idx, file in enumerate(files) if idx % 2 == 1]
        gt_list = [file for idx, file in enumerate(files) if idx % 2 == 0]

        noisy_train_list, noisy_test_list, gt_train_list, gt_test_list = train_test_split(noisy_list, gt_list,
                                                                                          test_size=self.config.test_size,
                                                                                          random_state=1)
        print("Dataset: SSID, %d image pairs" % (len(noisy_list)))
        start_time = time.time()
        pool = Pool(processes=8)
        train_num = int(ceil(len(noisy_train_list) / 8))

        # Load training data
        noisy_loaders = [
            pool.apply_async(load_files, (
                noisy_train_list[i * train_num:i * train_num + train_num], self.config.res, self.config.test_mode))
            for i in range(8)]
        noisy_train = []
        for res in noisy_loaders:
            noisy_train.extend(res.get())

        gt_loaders = [
            pool.apply_async(load_files, (
                gt_train_list[i * train_num:i * train_num + train_num], self.config.res, self.config.test_mode))
            for i in range(8)]
        gt_train = []
        for res in gt_loaders:
            gt_train.extend(res.get())

        time2 = time.time() - start_time
        print("%d image pairs loaded for training set! setting took: %4.4fs" % (len(noisy_train), time2))

        # Load testing data
        noisy_loaders = [
            pool.apply_async(load_files, (
            noisy_test_list[i * train_num:i * train_num + train_num], self.config.res, self.config.test_mode))
            for i in range(8)]
        noisy_test = []
        for res in noisy_loaders:
            noisy_test.extend(res.get())

        gt_loaders = [
            pool.apply_async(load_files, (
            gt_test_list[i * train_num:i * train_num + train_num], self.config.res, self.config.test_mode))
            for i
            in range(8)]
        gt_test = []
        for res in gt_loaders:
            gt_test.extend(res.get())

        shape = np.shape(noisy_train)
        print (shape, len(noisy_train), len(noisy_train[0]), len(noisy_train[0][0]), np.asarray(noisy_train).shape)
        width = len(noisy_train[0])
        height = len(noisy_train[0][0])

        # standardize input images
        # self.mean = np.mean(noisy_train, axis=(1, 2), keepdims=True)
        # self.std = np.std(noisy_train, axis=(1, 2), keepdims=True)
        # noisy_train = (noisy_train - self.mean) / self.std
        # noisy_test = (noisy_test - self.mean) / self.std

        print("%d image pairs loaded for testing set! setting took: %4.4fs" % (len(noisy_test), time.time() - time2))
        return noisy_train, noisy_test, gt_train, gt_test, width, height

    def get_batch(self):
        noisy_batch = np.zeros(
            [self.config.batch_size, self.width, self.height, 3],
            dtype='float32')
        gt_batch = np.zeros(
            [self.config.batch_size, self.width, self.height, 3],
            dtype='float32')

        for i in range(self.config.batch_size):
            index = np.random.randint(len(self.noisy_train))
            noisy_patch = self.noisy_train[index]
            gt_patch = self.gt_train[index]

            # randomly flip, rotate patch (assuming that the patch shape is square)
            if self.config.augment:
                prob = np.random.rand()
                if prob > 0.5:
                    noisy_patch = np.flip(noisy_patch, axis=0)
                    gt_patch = np.flip(gt_patch, axis=0)
                prob = np.random.rand()
                if prob > 0.5:
                    noisy_patch = np.flip(noisy_patch, axis=1)
                    gt_patch = np.flip(gt_patch, axis=1)
                prob = np.random.rand()
                if prob > 0.5:
                    noisy_patch = np.rot90(noisy_patch)
                    gt_patch = np.rot90(gt_patch)
            # noisy_batch[i,:,:,:] = noisy_patch
            # gt_batch[i,:,:,:] = gt_patch
            noisy_batch[i, :, :, :] = preprocess(noisy_patch)  # pre/post processing function is defined in utils.py
            gt_batch[i, :, :, :] = preprocess(gt_patch)
        return noisy_batch, gt_batch
