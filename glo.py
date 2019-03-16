from kivy.uix.carousel import Carousel
from kivy.uix.scrollview import ScrollView

import Image_Thumb,cv2
carousel = Carousel(direction='left')
scroll = ScrollView()
ll=[]
ll1 = []
ll2 = []
ll3 = []
my = Image_Thumb.Image_Thumb(size_hint_y=None)
face_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_profileface.xml')
