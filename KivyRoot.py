from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget 
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
from PIL import Image

import glo
import numpy as np
import cv2
import os, itertools
import pytesseract
import glob
import webbrowser
import MyImageScreen,TextImageScreen,TextOne,TextTwo,TextThree,DuplicateScreen,Image_Thumb
###################################################################

class KivyRoot(BoxLayout):
	
	
	"""docstring for KivyRoot"""
	i1 = 0
	i2 = 0
	sc = 0
	def __init__(self, **kwargs):
		super(KivyRoot, self).__init__(**kwargs)
		self.screen_list = []

	def processDup(self):
		del glo.ll[:]
		#files = os.listdir("E:/BEproject/DatasetDup/")
		files = os.listdir("C:/Users/TANMAY/Desktop/DuplicateData/")
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
		for file in files:
			image = cv2.imread("C:/Users/TANMAY/Desktop/DuplicateData/"+file)
			image = cv2.resize(image, (400, 300))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			imagelist.append(image)

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
			meanSquareError, structuralSimilarity = self.findDuplicate(file1, file2)
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
					if((images+files[i]) not in glo.ll):
						if(images+files[j] in glo.ll):
							indexImg= glo.ll.index(images+files[j])
							glo.ll.insert(indexImg+1,(images+files[i]))
						else:
							glo.ll.append(images+files[i])	
					if((images+files[j]) not in glo.ll):
						if(images+files[i] in glo.ll):
							indexImg= glo.ll.index(images+files[i])
							glo.ll.insert(indexImg+1,(images+files[j]))
						else:
							glo.ll.append(images+files[j])
					#c = c + 1
		

		#self.parent.ids.scroll.add_widget(my)

	def mse(self,image1, image2):
		error = np.sum((image1.astype("float")-image2.astype("float"))**2)
		error /= float(image1.shape[0]*image1.shape[1])
		return error

	#function to calculate mse, ssim
	def findDuplicate(self,image1, image2):
		meanSquareError = self.mse(image1, image2)
		structuralSimilarity = 1
		if meanSquareError>1200:
			structuralSimilarity = 	ssim(image1, image2)
		return meanSquareError, structuralSimilarity

	def processText(self):
		del glo.ll1[:]
		del glo.ll2[:]
		del glo.ll3[:]
		files = os.listdir("C:/Users/TANMAY/Desktop/TextData1/")
		cou = 0
		for file_name in files:
			cou = cou + 1
			print(cou)
			img = cv2.imread("C:/Users/TANMAY/Desktop/TextData1/"+file_name)
			height, width = img.shape[:2]
			tot = height*width
			print('height',height)
			print('width',width)
			print('tot', tot)
			img_final = cv2.imread("C:/Users/TANMAY/Desktop/TextData1/"+file_name)
			img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			#cv2.imshow('img2gray',img2gray)
			ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
			#mask = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

			#cv2.imshow('mask_after THRESH_BINARY',mask)
			image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
			#cv2.imshow('bitwise_and',image_final)
			#ret, nnew_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
			nnew_img = cv2.adaptiveThreshold(image_final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
			#cv2.imshow('adaptiveThreshold',nnew_img)
			#cv2.imshow('new_img',new_img)
			'''
			        line  8 to 12  : Remove noisy portion 
			'''
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
			faces = glo.face_cascade.detectMultiScale(img2gray, 1.3, 5)
			#faces1 = face_cascade1.detectMultiScale(img2gray, 1.3, 5)
			#faces2 = face_cascade2.detectMultiScale(img2gray, 1.3, 5)
			#faces3 = face_cascade3.detectMultiScale(img2gray, 1.3, 5)
			profiles = glo.profile_cascade.detectMultiScale(img2gray)
			print("faces:",faces)
			print("profiles:",profiles)
			for (x,y,w,h) in faces:
			    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			    roi_gray = img2gray[y:y+h, x:x+w]
			    roi_color = img[y:y+h, x:x+w]
			for (x,y,w,h) in profiles:
			    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
			#cv2.imshow('final image with contours',img)
			rat = (area*1.0)/tot
			
			if rat > 0 and rat < 0.51 and (len(faces) > 0 or len(profiles) > 0):
				glo.ll2.append("C:/Users/TANMAY/Desktop/TextData1/"+file_name)
				print('Person:',rat)
			elif rat >= 0.21 and rat < 0.71 : 
				glo.ll1.append("C:/Users/TANMAY/Desktop/TextData1/"+file_name) 
				print('Plain:',rat)
			elif rat > 0.71:
				glo.ll3.append("C:/Users/TANMAY/Desktop/TextData1/"+file_name)
				print('Scanned:',rat)
			else:
				print('######### area/tot = ',rat)
		'''scroll1 = ScrollView()
		scroll2 = ScrollView()
		scroll3 = ScrollView()
		
		#print('my1 object created')
		my1.add(ll1)
		my1.height= sum(x.height for x in my1.children)
		scroll1.add_widget(my1)
		self.ids.textone.add_widget(scroll1)
		my2.add(ll2)
		my2.height= sum(x.height for x in my2.children)
		scroll2.add_widget(my2)
		self.ids.texttwo.add_widget(scroll2)
		my3.add(ll3)
		my3.height= sum(x.height for x in my3.children)
		scroll3.add_widget(my3)
		self.ids.textthree.add_widget(scroll3)'''
		

	def changeScreen(self, next_screen):
		
		if self.ids.screen_manager.current not in self.screen_list:
			self.screen_list.append(self.ids.screen_manager.current)

		if next_screen == 'about':
			self.ids.screen_manager.current = 'about_screen'
		
		if next_screen == 'duplicateimages':
			if self.sc != 1:
				glo.scroll.clear_widgets()
				glo.carousel.clear_widgets()
				glo.my.clear_widgets()
				if self.sc == 2:
					self.ids.textone.remove_widget(glo.scroll)
				elif self.sc == 3:
					self.ids.texttwo.remove_widget(glo.scroll)
				elif self.sc == 4:
					self.ids.textthree.remove_widget(glo.scroll)
				self.i1 += 1
				if self.i1 == 1:
					#function
					self.processDup()
				
				print('my object created')
				Image_Thumb.i = 0
				glo.my.add(glo.ll)
				print('ll added')
				glo.my.height= sum(x.height for x in glo.my.children)
				glo.scroll.add_widget(glo.my)
				self.ids.dupp.add_widget(glo.scroll)
				self.sc = 1
			self.ids.screen_manager.current = 'dup_screen'
		
		if next_screen == 'myimage':
			self.ids.screen_manager.current = 'myimage'
		
		if next_screen == 'textimages':
			self.i2 += 1
			if self.i2 == 1:
				self.processText()
			self.ids.screen_manager.current = 'screen'
			
			
		if next_screen == 'plainquote':
			if self.sc != 2:	
				print('my object created.....Plainquote')
				glo.scroll.clear_widgets()
				glo.carousel.clear_widgets()
				glo.my.clear_widgets()
				if self.sc == 1:
					self.ids.dupp.remove_widget(glo.scroll)
				elif self.sc == 3:
					self.ids.texttwo.remove_widget(glo.scroll)
				elif self.sc == 4:
					self.ids.textthree.remove_widget(glo.scroll)
				Image_Thumb.i = 0
				glo.my.add(glo.ll1)
				print('ll1 added')
				glo.my.height= sum(x.height for x in glo.my.children)
				glo.scroll.add_widget(glo.my)
				self.ids.textone.add_widget(glo.scroll)
				self.sc = 2
			self.ids.screen_manager.current = 'textone'
		
		if next_screen == 'personalityquote':
			if self.sc != 3:
				print('my object created.....Personaquote')
				glo.scroll.clear_widgets()
				glo.carousel.clear_widgets()
				glo.my.clear_widgets()
				if self.sc == 1:
					self.ids.dupp.remove_widget(glo.scroll)
				elif self.sc == 2:
					self.ids.textone.remove_widget(glo.scroll)
				elif self.sc == 4:
					self.ids.textthree.remove_widget(glo.scroll)
				Image_Thumb.i = 0
				glo.my.add(glo.ll2)
				print('ll2 added')
				glo.my.height= sum(x.height for x in glo.my.children)
				glo.scroll.add_widget(glo.my)
				self.ids.texttwo.add_widget(glo.scroll)
				self.sc = 3
			self.ids.screen_manager.current = 'texttwo'

		if next_screen == 'scanneddoc':
			if self.sc != 4:	
				print('my object created.....Scanneddoc')
				glo.scroll.clear_widgets()
				glo.carousel.clear_widgets()
				glo.my.clear_widgets()
				if self.sc == 1:
					self.ids.dupp.remove_widget(glo.scroll)
				elif self.sc == 2:
					self.ids.textone.remove_widget(glo.scroll)
				elif self.sc == 3:
					self.ids.texttwo.remove_widget(glo.scroll)
				Image_Thumb.i = 0
				glo.my.add(glo.ll3)
				print('ll3 added')
				glo.my.height= sum(x.height for x in glo.my.children)
				glo.scroll.add_widget(glo.my)
				self.ids.textthree.add_widget(glo.scroll)
				self.sc = 4
			self.ids.screen_manager.current = 'textthree'

	def onBack(self):
		if self.screen_list:
			self.ids.screen_manager.current = self.screen_list.pop()
			return True
		return False		