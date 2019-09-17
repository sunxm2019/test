import torch,torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from torchvision import transforms, utils

#
# import math
#
#
# from skimage.transform import resize
# import utils
# import params
from PIL import Image


class CRNNDataset(Dataset):
	def __init__(self, image_path, label_file, alphabet, resize):
		super(CRNNDataset, self).__init__()
		self.image_path = image_path
		self.labels = self.__get_labels(label_file)
		self.alphabet = alphabet
		self.height, self.width = resize

		self.stds, self.means = self.__compute_stds_and_means()

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		img_file = list(self.labels[index].keys())[0]
		label_txt = list(self.labels[index].values())[0]

		image = cv2.imread(self.image_path+img_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		h, w = image.shape

		image = cv2.resize(image, (0, 0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)
		image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
		image = self.__preprocessing(image)

		return image, label_txt

	def __get_labels(self, label_file, code='utf-8-sig'):
		assert os.path.exists(label_file)
		with open(label_file, 'rb') as file:
			lines = file.readlines()

		labels = []
		for line in lines:
			line = line.decode(code, 'ignore')
			key = line.split('\t')[0]
			value = line.split('\t')[-1][:-1]
			labels.append({key:value.strip()})

		return labels

	def __compute_stds_and_means(self):
		imgs = np.zeros([self.height, self.width, 1, 1])
		img_file_name_list = [list(line.keys())[0] for line in self.labels]

		for i in range(len(img_file_name_list)):
			img = cv2.imread(self.image_path + img_file_name_list[i])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			h, w = img.shape[:2]

			img = cv2.resize(img, (0, 0), fx=self.width / w, fy=self.height / h, interpolation=cv2.INTER_CUBIC)
			img = img[:, :, np.newaxis, np.newaxis]
			imgs = np.concatenate((imgs, img), axis=3)

		imgs = imgs.astype(np.float32) / 255.
		means, stds = [], []
		pixels = imgs[:, :, 0, :].ravel()
		means.append(np.mean(pixels))
		stds.append(np.std(pixels))
		print(stds, means)

		return stds[0], means[0]

	def __preprocessing(self, image):
		image = image.astype(np.float32) / 255.
		image = torch.from_numpy(image).type(torch.FloatTensor)
		image.sub_(self.means).div_(self.stds)
		return image

	def debug(self):
		print(type(self.labels))
		print(len(self.labels))
		print(self.labels)


if __name__ == '__main__':
	dataset = CRNNDataset('.\\data_set\\', '.\\label.txt', '', (32, 280))
	dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

	image = next(iter(dataloader))
	img = image[0][1].permute(1, 2, 0).squeeze().numpy()
	plt.imshow(img)
	plt.show()

	# python -m visdom.server
	# http://localhost:8097/
	# import visdom
	# viz = visdom.Visdom()
	# for i_batch, (image, index) in enumerate(dataloader):
	# 	img = image[0].permute(1, 2, 0).squeeze().numpy()
	# 	viz.image(img, win='sample', opts=dict(title='sample'))
	# 	break
