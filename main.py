
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.floatlayout import FloatLayout 
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image as img
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from kivy.uix.carousel import Carousel
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os, itertools
from PIL import Image

import pytesseract
import glob
import webbrowser
#from kivy.uix.button import Button
carousel = Carousel(direction='left')
scroll = ScrollView()
face_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
ll=[]
ll1 = []
ll2 = []
ll3 = []

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
		del ll[:]
		files = os.listdir("E:/BEproject/DatasetDup/")
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
		print(files[1])
		for file1, file2 in itertools.combinations(files, 2): 
			print file1, file2, ++count
			#file1 = "jura1.png"
			#file2 = "jura2.jpeg"
			image1 = cv2.imread("E:/BEproject/DatasetDup/"+file1)
			image2 = cv2.imread("E:/BEproject/DatasetDup/"+file2)
			#if image1:
			image1 = cv2.resize(image1, (300, 300))
			image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
			#if image2:
			image2 = cv2.resize(image2, (300, 300))
			image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
			meanSquareError, structuralSimilarity = self.findDuplicate(image1, image2)
			print("MeanSquareError... ", meanSquareError)
			print("SSIM... ", structuralSimilarity)
			mselist[i][j] = meanSquareError
			ssimlist[i][j] = structuralSimilarity
			if j == (numberOfFiles-1):
				i = i+1
				j = i+1
			else:
				j = j+1
		

		#matrix for mse
		for i in range(numberOfFiles):
			for j in range(numberOfFiles):
				print(mselist[i][j]),
			print()

		#matrix for ssim
		for i in range(numberOfFiles):
			for j in range(numberOfFiles):
				print(ssimlist[i][j]),
			print()

		c = 0
		'''cv2.imwrite("C:\Users\hiral\Desktop\duplicate\abc.png",image1)
		cv2.imwrite("C:\Users\hiral\Desktop\duplicate\ab.png",image2)
		'''#example of border
		
		images = "E:/BEproject/DatasetDup/"
		for i in range(numberOfFiles):
			for j in range(i+1,numberOfFiles):
				if mselist[i][j]<4000 or ssimlist[i][j]>0.5:
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
		

		#self.parent.ids.scroll.add_widget(my)

	def mse(self,image1, image2):
		error = np.sum((image1.astype("float")-image2.astype("float"))**2)
		error /= float(image1.shape[0]*image1.shape[1])
		return error

	#function to calculate mse, ssim
	def findDuplicate(self,image1, image2):
		meanSquareError = self.mse(image1, image2)
		structuralSimilarity = 	ssim(image1, image2)
		return meanSquareError, structuralSimilarity

	def captch_ex(self):
		del ll1[:]
		del ll2[:]
		del ll3[:]
		files = os.listdir("E:/BEproject/DatasetDup/")
		for file_name in files:
			img = cv2.imread("E:/BEproject/DatasetDup/"+file_name)
			height, width = img.shape[:2]
			tot = height*width
			print('height',height)
			print('width',width)
			print('tot', tot)
			img_final = cv2.imread("E:/BEproject/DatasetDup/"+file_name)
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
			faces = face_cascade.detectMultiScale(img2gray, 1.3, 5)
			print(faces)
			for (x,y,w,h) in faces:
			    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			    roi_gray = img2gray[y:y+h, x:x+w]
			    roi_color = img[y:y+h, x:x+w]

			rat = (area*1.0)/tot
			if rat >= 0.25 and rat < 0.75 : 
				ll1.append("E:/BEproject/DatasetDup/"+file_name) 
				print('Plain:',rat)
			elif rat > 0 and rat < 0.25 and len(faces) > 0:
				ll2.append("E:/BEproject/DatasetDup/"+file_name)
				print('Person:',rat)
			elif rat > 0.75:
				ll3.append("E:/BEproject/DatasetDup/"+file_name)
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
				scroll.clear_widgets()
				carousel.clear_widgets()
				my.clear_widgets()
				if self.sc == 2:
					self.ids.textone.remove_widget(scroll)
				elif self.sc == 3:
					self.ids.texttwo.remove_widget(scroll)
				elif self.sc == 4:
					self.ids.textthree.remove_widget(scroll)
				self.i1 += 1
				if self.i1 == 1:
					#function
					self.processDup()
				
				print('my object created')
				Image_Thumb.i = 0
				my.add(ll)
				print('ll added')
				my.height= sum(x.height for x in my.children)
				scroll.add_widget(my)
				self.ids.dupp.add_widget(scroll)
				self.sc = 1
			self.ids.screen_manager.current = 'dup_screen'
		
		if next_screen == 'myimage':
			self.ids.screen_manager.current = 'myimage'
		
		if next_screen == 'textimages':
			self.i2 += 1
			if self.i2 == 1:
				self.captch_ex()
			self.ids.screen_manager.current = 'screen'
			
			
		if next_screen == 'plainquote':
			if self.sc != 2:	
				print('my object created.....Plainquote')
				scroll.clear_widgets()
				carousel.clear_widgets()
				my.clear_widgets()
				if self.sc == 1:
					self.ids.dupp.remove_widget(scroll)
				elif self.sc == 3:
					self.ids.texttwo.remove_widget(scroll)
				elif self.sc == 4:
					self.ids.textthree.remove_widget(scroll)
				Image_Thumb.i = 0
				my.add(ll1)
				print('ll1 added')
				my.height= sum(x.height for x in my.children)
				scroll.add_widget(my)
				self.ids.textone.add_widget(scroll)
				self.sc = 2
			self.ids.screen_manager.current = 'textone'
		
		if next_screen == 'personalityquote':
			if self.sc != 3:
				print('my object created.....Personaquote')
				scroll.clear_widgets()
				carousel.clear_widgets()
				my.clear_widgets()
				if self.sc == 1:
					self.ids.dupp.remove_widget(scroll)
				elif self.sc == 2:
					self.ids.textone.remove_widget(scroll)
				elif self.sc == 4:
					self.ids.textthree.remove_widget(scroll)
				Image_Thumb.i = 0
				my.add(ll2)
				print('ll2 added')
				my.height= sum(x.height for x in my.children)
				scroll.add_widget(my)
				self.ids.texttwo.add_widget(scroll)
				self.sc = 3
			self.ids.screen_manager.current = 'texttwo'

		if next_screen == 'scanneddoc':
			if self.sc != 4:	
				print('my object created.....Scanneddoc')
				scroll.clear_widgets()
				carousel.clear_widgets()
				my.clear_widgets()
				if self.sc == 1:
					self.ids.dupp.remove_widget(scroll)
				elif self.sc == 2:
					self.ids.textone.remove_widget(scroll)
				elif self.sc == 3:
					self.ids.texttwo.remove_widget(scroll)
				Image_Thumb.i = 0
				my.add(ll3)
				print('ll3 added')
				my.height= sum(x.height for x in my.children)
				scroll.add_widget(my)
				self.ids.textthree.add_widget(scroll)
				self.sc = 4
			self.ids.screen_manager.current = 'textthree'

	def onBack(self):
		if self.screen_list:
			self.ids.screen_manager.current = self.screen_list.pop()
			return True
		return False		

###################################################################
class KivyApp(App):
	"""docstring for KivyApp"""
	def __init__(self, **kwargs):
		super(KivyApp, self).__init__(**kwargs)
		Window.bind(on_keyboard = self.onBack)


	def onBack(self,window,key,*args):
		if key == 27:
			return self.root.onBack()

	


	def getText(self):
		return ("This app was built using Kivy and Python..."
				"References:"
				"[b][ref=kivy]Kivy[/ref][/b]"
				"[b][ref=python]Python[/ref][/b]")

	def on_ref_press(self, instance, ref):
		dictt = {
			'kivy': 'https://kivy.org/#home',
			'python': 'https://www.python.org/'
		}
		webbrowser.open(dictt[ref])


	def build(self):
		return KivyRoot()

class MyImageScreen(Screen):
	def __init__(self, **kwargs):
		super(MyImageScreen, self).__init__(**kwargs)
		self.add_widget(carousel)
		#print(carousel.slides)

class TextImageScreen(Screen):
	def __init__(self, **kwargs):
		super(TextImageScreen, self).__init__(**kwargs)

class TextOne(Screen):
	"""docstring for TextOne"""
	def __init__(self, **kwargs):
		super(TextOne, self).__init__(**kwargs)

class TextTwo(Screen):
	"""docstring for TextOne"""
	def __init__(self, **kwargs):
		super(TextTwo, self).__init__(**kwargs)

class TextThree(Screen):
	"""docstring for TextOne"""
	def __init__(self, **kwargs):
		super(TextThree, self).__init__(**kwargs)		
		

class MyImage(img):
	ind=0
	def __init__(self, **kwargs):
		self.ind = kwargs.get('ind')
		super(MyImage, self).__init__(**kwargs)
		self.height = 200

	def on_touch_down(self, touch):
		if self.collide_point(*touch.pos):
			print(self.ind)
			carousel.index=self.ind
			print(self.height)

	'''def onPress(self):
		print('hello')
		carousel.remove_widget(self)
		my.remove_widget(self)		
'''
class  CarouselO(FloatLayout):
	"""docstring for  CarouselO"""
	src = None
	def __init__(self, **kwargs):
		super(CarouselO, self).__init__(**kwargs)
		#self.width = Window.width;
		self.src = kwargs.get('myimg')
		secimg = img(source = kwargs.get('myimg'))
		self.add_widget(secimg)

	'''def on_touch_down(self, touch):
		if self.collide_point(*touch.pos):
			#print(self.ind)
			#carousel.index=self.ind
			print(self.height)
'''
	def onPress(self):
		os.remove(self.src)
		if self.src in ll:
			ll.remove(self.src)
		if self.src in ll1:
			ll1.remove(self.src)
		if self.src in ll2:
			ll2.remove(self.src)
		if self.src in ll3:
			ll3.remove(self.src)	
		carousel.remove_widget(self)
		for child in my.children[:]:
			print(child.source)
			print(self.src)
			if(child.source == self.src):
				my.remove_widget(child)
		
		#print(self.src)
		#carousel.index = carousel.index+1
		#my.remove_widget(self.kwargs.get('myimg'))
		

class Image_Thumb(GridLayout):
	"""docstng for MyImage"""
	i=0
	#carousel = None
	def __init__(self, **kwargs):
		super(Image_Thumb, self).__init__(**kwargs)
		#print(glob.glob("E:/BEproject/*"))
		#print(os.listdir("E:/BEproject/"))
		#images = glob.glob("//storage//emulated//0//kivy//DatasetDup/*.jpeg")
		#self.rows = 4
		self.cols = 2
		#Image_Thumb.carousel = Carousel(direction = 'left')
		print('Object created')
		#self.add(ll)
		#self.size_hint_y = None
		

	#def callback(self, obj, touch):
	def add(self,ll):
		
		#images = glob.glob("E://BEproject/*.jpeg")
		for img in ll:
			thumb = MyImage(source = img, ind = Image_Thumb.i)
			thumb2 = CarouselO(myimg = img, ind = Image_Thumb.i)
			#thumb.remove_widget(thumb.ids.btn)
			Image_Thumb.i += 1
			#thumb.bind(on_touch_down = self.callback)
			self.add_widget(thumb)
			carousel.add_widget(thumb2)
			print(img)

my = Image_Thumb(size_hint_y=None)
'''my1 = Image_Thumb(size_hint_y=None)
my2 = Image_Thumb(size_hint_y=None)
my3 = Image_Thumb(size_hint_y=None)'''

class DuplicateScreen(Screen):
	def __init__(self,**kwargs):
		super(DuplicateScreen,self).__init__(**kwargs)
		


		#app.changeScreen(nextScreen)

if __name__ == '__main__':
	KivyApp().run()
		

'''ScrollView:
            size: self.size
            size_hint: (1,None)
            Image_Thumb:
                size_hint_y: None
                cols: 1
                spacing: 0, 0
                padding: 0, 0

                DuplicateScreen:
			name: 'dup_screen'
<CarouselO>:
	Button:
		text:'Delete'
		font_size: 28
		size_hint: 1, 0.2
		pos_hint: {'top':1} 
		on_press: root.onPress()


'''