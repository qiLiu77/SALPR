from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch
import params_test_CCPD as params

class CCPDDataset(Dataset):
	def __init__(self, img_root, label_path, alphabet, resize):
		super(CCPDDataset, self).__init__()
		self.img_root = img_root
		self.labels = self.get_labels(label_path)
		self.alphabet = alphabet
		self.width, self.height = resize

	def get_labels(self, label_path):
		labels = []
		with open(label_path, 'r', encoding='utf-8') as file:
			for item in file.readlines():
				item = item.strip()
				image_path = os.path.join(self.img_root,item.split(' ')[0])
				text_label = item.split(' ')[1]

				label = []
				for (i,char) in enumerate(text_label.split('_')):
					if i==0:
						if int(char) >32:
							continue
						label.append(int(char)+1)
					elif i==1:
						if int(char) > 23:
							continue
						label.append(int(char)+34)
					else:
						if int(char) > 33:
							continue
						label.append(int(char)+34)
				for i in range(8-len(label)):
					label.append(0)
				labels.append({image_path:label})

		return labels

	def __len__(self):
		return len(self.labels)

	def preprocessing(self, image):

		## already have been computed
		image = image.astype(np.float32) / 255.
		image = torch.from_numpy(image).type(torch.FloatTensor)
		image.sub_(params.mean).div_(params.std)

		return image

	def __getitem__(self, index):
		image_name = list(self.labels[index].keys())[0]
		image = cv2.imread(self.img_root+'/'+image_name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		h, w = image.shape

		image = cv2.resize(image, (0,0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)
		image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
		image = self.preprocessing(image)
		label = list(self.labels[index].values())[0] 

		return image, label
