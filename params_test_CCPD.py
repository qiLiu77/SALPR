alphabet = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'

K = 8
std = 1
mean = 0
imgW = 96
imgH = 32
val_batchSize = 192

workers = 2
gpu = 5

model_type = 'vgg'
model_path = 'checkpoint/CCPD_model.pth'
image_dir = ''
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/test.txt'
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/ccpd_db.txt'
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/ccpd_fn.txt'
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/ccpd_rotate.txt'
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/ccpd_tilt.txt'
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/ccpd_weather.txt'
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/ccpd_challenge.txt'
image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/ccpd_nonHor.txt'
# image_path = '/data/qi.liu/datasets/open-dataset/ALPR/CCPD/CCPD2018/ccpd_dataset_ori_LPs/det_txt/all_test.txt'
# image_path = '/rdata/qi.liu/data/open-dataset/ALPR/LPR/CLPD/LPs.txt'



