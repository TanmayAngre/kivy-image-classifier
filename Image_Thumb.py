from kivy.uix.gridlayout import GridLayout

import MyImage,CarouselO,glo
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
			thumb = MyImage.MyImage(source = img, ind = Image_Thumb.i)
			thumb2 = CarouselO.CarouselO(myimg = img, ind = Image_Thumb.i)
			#thumb.remove_widget(thumb.ids.btn)
			Image_Thumb.i += 1
			#thumb.bind(on_touch_down = self.callback)
			self.add_widget(thumb)
			glo.carousel.add_widget(thumb2)
			print(img)