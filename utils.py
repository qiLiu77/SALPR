#!/usr/bin/python
# encoding: utf-8

#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import params
from tqdm import tqdm
import numpy as np 
import cv2
import os
import random
import multiprocessing

def get_real(gt, font_size=10, image_dir='/rdata/qi.liu/code/LPR/gen_plate/gen_data/char/clean_char/image'):
    gt_np = gt.squeeze().numpy()
    all_imgs = []
    for i in range(len(gt_np)):
        imgs = []
        for j in range(font_size):
            if not gt_np[i] == 0: 
                image_path = os.path.join(image_dir, str(gt_np[i]-1), str(j)+'.png')
                # print(image_path)
                img = cv2.imread(image_path, 0)  
                img = img*1.0/255
            else:
                img = np.zeros((32,32))
            imgs.append(img)
        all_imgs.append(imgs)
    
    # print(type(all_imgs), len(all_imgs))
    all_imgs = torch.tensor(all_imgs, dtype=torch.float)
    h = all_imgs.size(2)
    w = all_imgs.size(3)
    all_imgs = all_imgs.reshape(-1, 1, h, w)
    
    # print('GT_output: ',all_imgs.size())

    return  all_imgs #(batch*10)*1*32*32

def parallel_get_real(gt, font_size=10, image_dir='/rdata/qi.liu/code/LPR/gen_plate/gen_data/char/clean_char/image'):
    manager = multiprocessing.Manager()
    # queue = manager.Queue()
    pool = multiprocessing.Pool(processes = 16)
    p = pool.apply_async(get_real, args = (gt, font_size, image_dir))
    pool.close()
    # while queue.empty():
    #     pass
    # data = queue.get()
    data = p.get()
    return data



class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = '-'+alphabet  # for `-1` index

        self.dict = {}
        for i, char in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            if len(item)<1:
                continue
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def encode_char(self, char):

        return self.dict[char]
    
    def encode_list(self, text, K=8):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.
            K : the max length of texts

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        # print(text)
        length = []
        all_result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:
            result = []
            if decode_flag:
                item = item.decode('utf-8','strict')
            # print(item)
            length.append(len(item))
            for i in range(K):
                # print(item)
                if i<len(item): 
                    char = item[i]
                    # print(char)
                    index = self.dict[char]
                    result.append(index)
                else:
                    result.append(0)
            all_result.append(result)
        return (torch.LongTensor(all_result))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
    
    def decode_list(self, t):
        texts = []
        for i in range(t.shape[0]):
            t_item = t[i,:]
            char_list = []
            for i in range(t_item.shape[0]):
                if t_item[i] == 0:
                    pass
                    # char_list.append('-')
                else:
                    char_list.append(self.alphabet[t_item[i]])
                # print(char_list, self.alphabet[44])
            # print('char_list:  ' ,''.join(char_list))
            texts.append(''.join(char_list))
        # print('texts:  ', texts)
        return texts

    def decode_sa(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(text_index):
            text = ''.join([self.alphabet[i] for i in text_index[index, :]])
            texts.append(text.strip('-'))
        return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)
    #print(v.size())


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

def to_alphabet(path):
    with open(path, 'r', encoding='utf-8') as file:
        alphabet = list(set(''.join(file.readlines())))

    return alphabet

def get_batch_label(d, i):
    
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label

'''
    qi added
    fuse all text label situation
'''
def get_batch_text_label(d, i, isPretrain=False, train=True, needChi=False):
    
    label = []
    if isPretrain == False:
        for idx in i:
            label = []
            chinese = []
            for idx in i:
                label.append(list(d.labels[idx].values())[0])
                label_chi = list(d.labels[idx].values())[0][0]
                # print(label_be,label_af)
                chinese.append(label_chi)
            if needChi == False:
                return label
            elif needChi ==True:
                return label, chinese
    else:
        if train == True:
            label_be = []
            label_af = []
            chinese = []
            for idx in i:
                # print(d.labels[idx].values())
                label_be.append(list(d.labels[idx].values())[0][0])
                label_af.append(list(d.labels[idx].values())[0][1])
                label_chi = list(d.labels[idx].values())[0][0][0]
                chinese.append(label_chi)

            # print('a   ',label_be, label_af,chinese)

            if needChi == False:
                return label_be, label_af
            elif needChi ==True:
                # print('chinese', chinese)
                return label_be, label_af, chinese
        else:
            label_be = []
            label_af = []
            chinese = []

            for idx in i:
                label_be = list(d.labels[idx].values())[0][0]
                label_af = list(d.labels[idx].values())[0][1]
                # print(label_be,label_af)
                label.append(label_be+label_af)
                label_chi = list(d.labels[idx].values())[0][0][0]
                chinese.append(label_chi)
            if needChi == False:
                return label
            elif needChi ==True:
                return label, chinese

#获取车牌颜色label
def get_batch_color_label(d, i):
    c_label = []
    for idx in i:
        c_label.append(list(d.color_labels[idx].values())[0])
    
    return torch.LongTensor(c_label)

#获取attention的GT
def get_batch_seg_label(d, i):
    
    label = []
    for idx in i:
        label.append(list(d.seg_labels[idx].values())[0])
    return torch.LongTensor(label)

def get_batch_label_once(d, i, train=True):
    
    if train == True:
        label_be = []
        label_af = []
        for idx in i:
            # print(d.labels[idx].values())
            label_be.append(list(d.labels[idx].values())[0][0])
            label_af.append(list(d.labels[idx].values())[0][1])
            # print('a   ',label_be, label_af)
        return label_be, label_af
    else:
        label=[]
        for idx in i:
            label_be = list(d.labels[idx].values())[0][0]
            label_af = list(d.labels[idx].values())[0][1]
            # print(label_be,label_af)
            label.append(label_be+label_af)
        return label

def compute_std_mean(txt_path, image_prefix, NUM=None):
    
    imgs = np.zeros([params.imgH, params.imgW, 1, 1])
    means, stds = [], []
    with open(txt_path, 'r') as file:
        contents = [c.strip().split(' ')[0] for c in file.readlines()]
        if NUM is None:
            NUM = len(contents)
        else:
            random.shuffle(contents)
        for i in tqdm(range(NUM)):
            file_name = contents[i]
            img_path = os.path.join(image_prefix, file_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            img = cv2.resize(img, (0,0), fx=params.imgW/w, fy=params.imgH/h, interpolation=cv2.INTER_CUBIC)
            img = img[:, :, np.newaxis, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(1):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()
    # print(means, stds)

    return stds, means
