# attention-seg method anfd visualize
import numpy as np
import sys, os
import time
import cv2

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
# import utils
import params_test_CCPD as params

from dataset_CCPD import CCPDDataset
import models.LPR_model as model
import Levenshtein


def decode_list(t, alphabet):
        texts = []
        for i in range(t.shape[0]):
            t_item = t[i,:]
            char_list = []
            for i in range(t_item.shape[0]):
                if t_item[i] == 0:
                    pass
                    # char_list.append('-')
                else:
                    char_list.append(alphabet[t_item[i]])
                # print(char_list, self.alphabet[44])
            # print('char_list:  ' ,''.join(char_list))
            texts.append(''.join(char_list))
        # print('texts:  ', texts)
        return texts

def val(lpr_model, val_loader, alphabet):

    print('Start val')
    lpr_model.eval()

    n_correct = 0
    infer_time = 0
    all_predict = []
    all_image_vis_atten = []

    for i_batch, (image, label) in enumerate(val_loader):
        print(i_batch)
        if i_batch >=4000:
            break
        image = image.cuda()
        start_time = time.time()
        _, preds = lpr_model(image)
        forward_time = time.time() - start_time
        infer_time += forward_time
        print('forward_time: ', forward_time*1000)

        batch_size = image.size(0)
        cost = 0
        preds_all = preds
        preds = torch.chunk(preds, preds.size(1), 1)

        label_all = torch.stack([label[0].unsqueeze(1), label[1].unsqueeze(1), label[2].unsqueeze(1), label[3].unsqueeze(1), label[4].unsqueeze(1), label[5].unsqueeze(1), label[6].unsqueeze(1)],1)
        text_label = decode_list(label_all, alphabet)

        atten_list, preds_all = preds_all.max(2)
        sim_preds = decode_list(preds_all.data, alphabet)

        for pred, target in zip(sim_preds, text_label):
            all_predict.append(pred)
            print('i_batch:', i_batch, '  target:', target, '  pred:', pred)
            if pred == target:
                n_correct += 1

    acc = float(n_correct)/len(val_dataset)
    print('totol_time: ', float(infer_time))
    print('time: ', float(infer_time)/i_batch*1000)
    print('FPS: ', 4000/float(infer_time))
    print('ACC: ', n_correct,'/',len(val_dataset), acc)
    return acc, all_predict


if __name__ == '__main__':

    alphabet = '-'+params.alphabet #'-' is for background
    nclass = len(alphabet)
    lpr_model = model.LPR_model(1, nclass, imgW=params.imgW, imgH=params.imgH,  K=params.K).cuda()

    lpr_model.load_state_dict(torch.load(params.model_path,  map_location=lambda storage, loc: storage.cuda(0)))

    val_dataset = CCPDDataset(params.image_dir,params.image_path, params.alphabet, (params.imgW, params.imgH))

    val_loader = DataLoader(val_dataset, batch_size=params.val_batchSize, shuffle=False, num_workers=params.workers)
    
    acc, preds = val(lpr_model, val_loader, alphabet)

