from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os, itertools
import pprint
from PIL import Image
import pytesseract

def processText():
	ll1 = []
	ll2 = []
	ll3 = []
	plain = ['C:/Users/TANMAY/Desktop/TextData/as.jpg','C:/Users/TANMAY/Desktop/TextData/clk.jpg','C:/Users/TANMAY/Desktop/TextData/default.jpg','C:/Users/TANMAY/Desktop/TextData/df.jpg','C:/Users/TANMAY/Desktop/TextData/images.jpg','C:/Users/TANMAY/Desktop/TextData/images1.jpg','C:/Users/TANMAY/Desktop/TextData/FB_IMG.jpg','C:/Users/TANMAY/Desktop/TextData/end.jpg','C:/Users/TANMAY/Desktop/TextData/img6.jpg','C:/Users/TANMAY/Desktop/TextData/smi.jpg','C:/Users/TANMAY/Desktop/TextData/sd.jpg','C:/Users/TANMAY/Desktop/TextData/school.jpg','C:/Users/TANMAY/Desktop/TextData/mou.jpg','C:/Users/TANMAY/Desktop/TextData/zin.jpg']
	person = ['C:/Users/TANMAY/Desktop/TextData/ab.jpg','C:/Users/TANMAY/Desktop/TextData/abc.jpg','C:/Users/TANMAY/Desktop/TextData/cou.jpg','C:/Users/TANMAY/Desktop/TextData/images2.jpg','C:/Users/TANMAY/Desktop/TextData/download.jpg','C:/Users/TANMAY/Desktop/TextData/koli.jpg','C:/Users/TANMAY/Desktop/TextData/modi.jpg','C:/Users/TANMAY/Desktop/TextData/mlk.jpg','C:/Users/TANMAY/Desktop/TextData/text.jpg']
	scanned = ['C:/Users/TANMAY/Desktop/TextData/downloadd.jpg','C:/Users/TANMAY/Desktop/TextData/imagesdown.jpg','C:/Users/TANMAY/Desktop/TextData/imagest.jpg','C:/Users/TANMAY/Desktop/TextData/imagestext.jpg','C:/Users/TANMAY/Desktop/TextData/skim.jpg']
	dict1 = {}
	dict2 = {}
	dict3 = {}
	files = os.listdir("C:/Users/TANMAY/Desktop/TextData/")
	cou = 0
	for file_name in files:
		cou = cou + 1
		print(cou)
		const = "C:/Users/TANMAY/Desktop/TextData/"+file_name
		img = cv2.imread(const)
		height, width = img.shape[:2]
		tot = height*width
		print('height',height)
		print('width',width)
		print('tot', tot)
		if const in plain:
			dict1[const] = 1
		else:
			dict1[const] = 0
		if const in person:
			dict2[const] = 1
		else:
			dict2[const] = 0
		if const in scanned:
			dict3[const] = 1
		else:
			dict3[const] = 0
		img_final = cv2.imread("C:/Users/TANMAY/Desktop/TextData/"+file_name)
		img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#cv2.imshow('img2gray',img2gray)
		ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
		#mask = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

		#cv2.imshow('mask',mask)
		image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
		#cv2.imshow('image_final',image_final)
		#ret, nnew_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
		nnew_img = cv2.adaptiveThreshold(image_final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		#cv2.imshow('nnew',nnew_img)
		#cv2.imshow('new_img',new_img)
	
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
		                                                     3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
		#dilated = cv2.dilate(new_img, kernel, iterations=2)  # dilate , more the iteration more the dilation
		dilated1 = cv2.dilate(nnew_img, kernel, iterations=2)
		#cv2.imshow('dil1',dilated1)
		#dilated1 = cv2.erode(dilated1, kernel, iterations=2)
		#cv2.imshow('erod1',dilated1)
		#cv2.imshow('dil',dilated)
		# for cv2.x.x

		_, contours, hierarchy = cv2.findContours(dilated1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

		# for cv3.x.x comment above line and uncomment line below

		#image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		print('contour length:',len(contours))
		index = 0
		area = 0
		for contour in contours:
		    # get rectangle bounding contour
		    [x, y, w, h] = cv2.boundingRect(contour)

		    # Don't plot small false positives that aren't text
		    if w < 35 and h < 35:
		        print('<35')
		        continue

		    # draw rectangle around contour on original image
		    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

		    
		    #you can crop image and send to OCR  , false detected will return no text :)
		    cropped = img_final[y :y + h , x : x + w]

		    s = 'F:/photos/images' + '/crop_' + str(index) + '.jpg' 
		    cv2.imwrite(s , cropped)
		    ab = Image.open(s)
		    t = pytesseract.image_to_string(ab)
		    print(len(t))
		    if t == 0:
		        print('passed')
		        pass
		    else:
		        if len(t)>=2:
		            arr = w*h
		            #print(len(t))
		            print(arr)
		            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		            area+=arr
		        else:
		            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
		    
		    index = index + 1
		    os.remove(s)

		print("outside contours loop")
		#cv2.imwrite('E:/BEproject/'+file_name,img)
		  
		#print('ll1.size = ',len(ll1)) 
		#print(area) 
		faces = face_cascade.detectMultiScale(img2gray, 1.3, 5)
		#faces1 = face_cascade1.detectMultiScale(img2gray, 1.3, 5)
		#faces2 = face_cascade2.detectMultiScale(img2gray, 1.3, 5)
		#faces3 = face_cascade3.detectMultiScale(img2gray, 1.3, 5)
		profiles = profile_cascade.detectMultiScale(img2gray)
		print("faces:",faces)
		print("profiles:",profiles)
		for (x,y,w,h) in faces:
		    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		    roi_gray = img2gray[y:y+h, x:x+w]
		    roi_color = img[y:y+h, x:x+w]

		rat = (area*1.0)/tot
		
		if rat > 0 and rat < 0.51 and (len(faces) > 0 or len(profiles) > 0):
			ll2.append("C:/Users/TANMAY/Desktop/TextData/"+file_name)
			print('Person:',rat)
		elif rat >= 0.21 and rat < 0.71 : 
			ll1.append("C:/Users/TANMAY/Desktop/TextData/"+file_name) 
			print('Plain:',rat)
		elif rat > 0.71:
			ll3.append("C:/Users/TANMAY/Desktop/TextData/"+file_name)
			print('Scanned:',rat)
		else:
			print('######### area/tot = ',rat)
	pprint.pprint(dict1)
	pprint.pprint(dict2)
	pprint.pprint(dict3)
	
	for ind in range(3):
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		if ind == 0:
			for i in ll1:
				j = dict1.get(i)
				if j == 1:
					tp = tp + 1
				elif j == 0:
					fp = fp + 1
				dict1.pop(i)
			for key, value in dict1.iteritems():
				if value == 0:
					tn = tn + 1
				elif value == 1:
					fn = fn + 1

			print("For plain quotes...")
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
		if ind == 1:
			for i in ll2:
				j = dict2.get(i)
				if j == 1:
					tp = tp + 1
				elif j == 0:
					fp = fp + 1
				dict2.pop(i)
			for key, value in dict2.iteritems():
				if value == 0:
					tn = tn + 1
				elif value == 1:
					fn = fn + 1

			print("For personality quotes...")
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
		if ind == 2:
			for i in ll3:
				j = dict3.get(i)
				if j == 1:
					tp = tp + 1
				elif j == 0:
					fp = fp + 1
				dict3.pop(i)
			for key, value in dict3.iteritems():
				if value == 0:
					tn = tn + 1
				elif value == 1:
					fn = fn + 1

			print("For scanned docs...")
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

face_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_profileface.xml')
processText()