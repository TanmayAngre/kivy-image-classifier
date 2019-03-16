from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os, itertools
import pprint

def processDup():
	#del ll[:]
	#files = os.listdir("E:/BEproject/DatasetDup/")
	files = os.listdir("C:/Users/TANMAY/Desktop/DuplicateData/")
	plain = ['C:/Users/TANMAY/Desktop/TextData/as.jpg','C:/Users/TANMAY/Desktop/TextData/clk.jpg','C:/Users/TANMAY/Desktop/TextData/default.jpg','C:/Users/TANMAY/Desktop/TextData/df.jpg','C:/Users/TANMAY/Desktop/TextData/images.jpg','C:/Users/TANMAY/Desktop/TextData/images1.jpg','C:/Users/TANMAY/Desktop/TextData/FB_IMG.jpg','C:/Users/TANMAY/Desktop/TextData/end.jpg','C:/Users/TANMAY/Desktop/TextData/img6.jpg','C:/Users/TANMAY/Desktop/TextData/smi.jpg','C:/Users/TANMAY/Desktop/TextData/sd.jpg','C:/Users/TANMAY/Desktop/TextData/school.jpg','C:/Users/TANMAY/Desktop/TextData/mou.jpg','C:/Users/TANMAY/Desktop/TextData/zin.jpg']
	person = ['C:/Users/TANMAY/Desktop/TextData/ab.jpg','C:/Users/TANMAY/Desktop/TextData/abc.jpg','C:/Users/TANMAY/Desktop/TextData/cou.jpg','C:/Users/TANMAY/Desktop/TextData/images2.jpg','C:/Users/TANMAY/Desktop/TextData/download.jpg','C:/Users/TANMAY/Desktop/TextData/koli.jpg','C:/Users/TANMAY/Desktop/TextData/modi.jpg','C:/Users/TANMAY/Desktop/TextData/mlk.jpg','C:/Users/TANMAY/Desktop/TextData/text.jpg']
	scanned = ['C:/Users/TANMAY/Desktop/TextData/downloadd.jpg','C:/Users/TANMAY/Desktop/TextData/imagesdown.jpg','C:/Users/TANMAY/Desktop/TextData/imagest.jpg','C:/Users/TANMAY/Desktop/TextData/imagestext.jpg','C:/Users/TANMAY/Desktop/TextData/skim.jpg']
	dict = {}
	numberOfFiles = len(files)
	count = 0
	i = 0
	j = 1
	mselist = [[0 for x in range(numberOfFiles)] for y in range(numberOfFiles)]
	ssimlist = [[0 for x in range(numberOfFiles)] for y in range(numberOfFiles)]

	#read 2 images
	#resize images
	#convert to grayscale
	#calculate mse and ssim
	imagelist = []
	ll = []
	for file in files:
		const = "C:/Users/TANMAY/Desktop/DuplicateData/"+file
		image = cv2.imread(const)
		image = cv2.resize(image, (400, 300))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		imagelist.append(image)
		if const in actual:
			dict[const] = 1
		else:
			dict[const] = 0

	#print(files[1])
	for file1, file2 in itertools.combinations(imagelist, 2): 
		count = count + 1
		print count
		#file1 = "jura1.png"
		#file2 = "jura2.jpeg"
		#image1 = cv2.imread("E:/BEproject/DatasetDup/"+file1)
		#image2 = cv2.imread("E:/BEproject/DatasetDup/"+file2)
		#image1 = cv2.imread("C:/Users/TANMAY/Desktop/DuplicateData/"+file1)
		#image2 = cv2.imread("C:/Users/TANMAY/Desktop/DuplicateData/"+file2)
		#if image1:
		#image1 = cv2.resize(image1, (400, 300))
		#image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
		#if image2:
		#image2 = cv2.resize(image2, (400, 300))
		#image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
		meanSquareError, structuralSimilarity = findDuplicate(file1, file2)
		print("MeanSquareError... ", meanSquareError)
		print("SSIM... ", structuralSimilarity)
		mselist[i][j] = meanSquareError
		ssimlist[i][j] = structuralSimilarity
		if mselist[i][j]<1200 or ssimlist[i][j]>0.6:
			print('#########################################################')
			print(mselist[i][j])
			print(ssimlist[i][j])
		if j == (numberOfFiles-1):
			i = i+1
			j = i+1
		else:
			j = j+1
	

	#matrix for mse
	'''for i in range(numberOfFiles):
		for j in range(numberOfFiles):
			print(mselist[i][j]),
		print()

	#matrix for ssim
	for i in range(numberOfFiles):
		for j in range(numberOfFiles):
			print(ssimlist[i][j]),
		print()'''

	c = 0
	'''cv2.imwrite("C:\Users\hiral\Desktop\duplicate\abc.png",image1)
	cv2.imwrite("C:\Users\hiral\Desktop\duplicate\ab.png",image2)
	'''#example of border
	
	#images = "E:/BEproject/DatasetDup/"
	images = "C:/Users/TANMAY/Desktop/DuplicateData/"
	for i in range(numberOfFiles):
		for j in range(i+1,numberOfFiles):
			print(files[i],"  ",files[j])
			if mselist[i][j]<=1200 or ssimlist[i][j]>0.6:
				print(mselist[i][j])
				print(ssimlist[i][j])
				
				#print(images+files[i])
				#thumb = MyImage(source = images+files[i])
				#thumb2 = MyImage(source = images+files[j])
				#thumb.bind(on_touch_down = self.callback)
				#myImage.add_widget(thumb)
				#myImage.add_widget(thumb2)
				'''if((images+files[i])not in ll and (images+files[j]) not in ll):
					ll.append(images+files[i])
					ll.append(images+files[j])
				
				'''
				if((images+files[i]) not in ll):
					if(images+files[j] in ll):
						indexImg= ll.index(images+files[j])
						ll.insert(indexImg+1,(images+files[i]))
					else:
						ll.append(images+files[i])	
				if((images+files[j]) not in ll):
					if(images+files[i] in ll):
						indexImg= ll.index(images+files[i])
						ll.insert(indexImg+1,(images+files[j]))
					else:
						ll.append(images+files[j])
				#c = c + 1
	
	pprint.pprint(dict)
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for i in ll:
		j = dict.get(i)
		if j == 1:
			tp = tp + 1
		elif j == 0:
			fp = fp + 1
		dict.pop(i)
	for key, value in dict.iteritems():
		if value == 0:
			tn = tn + 1
		elif value == 1:
			fn = fn + 1

	print("True Positives: ",tp)
	print("False Positives: ",fp)
	print("True Negatives: ",tn)
	print("False Negatives: ",fn)
	print("total:",tp+tn+fp+fn)
	precision = (tp*1.0)/(tp+fp)
	print("precision:",precision)
	recall = (tp*1.0)/(tp+fn)
	print("recall:",recall)
	f1_score = (2*precision*recall*1.0)/(precision+recall)
	print("f1-score:",f1_score)
	accuracy = ((tp+tn)*1.0)/(tp+tn+fp+fn)
	print("accuracy:",accuracy)
	#self.parent.ids.scroll.add_widget(my)

def mse(image1, image2):
	error = np.sum((image1.astype("float")-image2.astype("float"))**2)
	error /= float(image1.shape[0]*image1.shape[1])
	return error

#function to calculate mse, ssim
def findDuplicate(image1, image2):
	meanSquareError = mse(image1, image2)
	structuralSimilarity = 1
	if meanSquareError>1200:
		structuralSimilarity = 	ssim(image1, image2)
	return meanSquareError, structuralSimilarity

processDup()