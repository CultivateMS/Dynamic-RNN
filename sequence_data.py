#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
@filename:data_process.py
@create:2018.06.30
@brief:data process utils
"""
import random
import tensorflow as tf
from tensorflow.contrib import rnn


def cal_model_para(filename):
    """
    @brief:根据数据计算模型的参数
    @Args:
        filename:输入数据文件的名字
    @Returns:
        max_seq_len: 最大sequence长度
        input_size: 每个序列的维度
        num_class: 分类个数
    """
    max_seq_len = -1
    fr = open(filename)
    for i, line in enumerate(fr):
        line = line.rstrip('\n')
        data_arr = line.split('&')
        feature_data = data_arr[0].split('\t')

        if i == 0:
            #获取第一个序列的维度
            input_size = len(feature_data[0].split('#'))
            #获取分类的数目
            num_class = len(data_arr[1].split('\t'))
        #获取当前训练record的序列数
        cur_seq_len = len(feature_data)
        #获取训练数据的最大序列数
        if cur_seq_len > max_seq_len:
            max_seq_len = cur_seq_len

    if max_seq_len % 10 != 0:
        max_seq_len = ((max_seq_len / 10) + 1) * 10

    print 'According to "%s", seq_max_len is set to %d, ' \
          'input_size is set to %d, num_class is set to %d.' \
          % (filename, max_seq_len, input_size, num_class)
    return max_seq_len, input_size, num_class


# ====================
#  Sequence Dat输入a
# ====================
class SequenceData(object):
    """ 
    Generate or read sequence of data with dynamic length.
    For example:
    - Class 0: linear sequences (i.e. [0.1, 0.2, 0.3, 0.4,...])
    - Class 1: random sequences (i.e. [0.23, 0.3, 0.1, 0.87,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    def __init__(self, filename, max_seq_len):
        self.batch_id = 0
        self.data, self.labels, self.seqlen = self.load_data(filename, max_seq_len)

    def next(self, batch_size):
        """ 
        get data by batch
        e.g. n_samples = 100, batch_size = 16, batch_num = 7(6+1), last_batch_size = 4
        Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_index = min(self.batch_id + batch_size, len(self.data))

        batch_data = (self.data[self.batch_id: batch_index])
        batch_labels = (self.labels[self.batch_id: batch_index])
        batch_seqlen = (self.seqlen[self.batch_id: batch_index])

        self.batch_id = batch_index
        return batch_data, batch_labels, batch_seqlen

    def cal_max_seq_len(self, filename):
        """
        compute the max seq len of the data
        :param filename: 
        :return: 
        """
        max_seq_len = -1
        fr = open(filename)
        for line in fr:
            line = line.rstrip('\n')
            data_arr = line.split('&')
            feature_data = data_arr[0].split('\t')
            cur_seq_len = len(feature_data)
            if cur_seq_len > max_seq_len:
                max_seq_len = cur_seq_len

        if max_seq_len % 10 != 0:
            max_seq_len = ((max_seq_len / 10) + 1) * 10

        return max_seq_len

    def load_data(self, filename, max_seq_len=20):
        """
        load data return datas & lables & seqlen
        """
        fr = open(filename)
        datas = []
        labels = []
        seqlen = []
        for line in fr:
            line = line.rstrip('\n')
            data_arr = line.split('&')
            feature_data = data_arr[0].split('\t')
            cur_seq_len = len(feature_data)
            seqlen.append(cur_seq_len)

            input_size = len(feature_data[0].split('#'))
            s = [[float(i) for i in item.split('#')] for item in feature_data]
            #pad seq with zero
            s += [[0.] * input_size for i in range(max_seq_len - cur_seq_len)]
            datas.append(s)
            # 区分训练与预测
            if len(data_arr) > 1:
                label_data_list = data_arr[1].split('\t')
                labels.append([float(item) for item in label_data_list])

        return datas, labels, seqlen

    def _data_generator(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                        max_value=100):
        """
        generate seq data with different size
        """
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])

    def test(self):
        """
        测试
        :return: 
        """
        filename = 'data/test_data.txt'
        max_seq_len = self.cal_max_seq_len(filename)
        self.load_data(filename, max_seq_len)


if __name__ == '__main__':
    filename = 'data/test_data.txt'
    s_data = SequenceData(filename, 20)
    s_data.test()
    cal_model_para(filename=filename)
