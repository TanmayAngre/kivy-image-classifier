from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image as img

import glo
class CarouselO(FloatLayout):
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
		if self.src in glo.ll:
			glo.ll.remove(self.src)
		if self.src in glo.ll1:
			glo.ll1.remove(self.src)
		if self.src in glo.ll2:
			glo.ll2.remove(self.src)
		if self.src in glo.ll3:
			glo.ll3.remove(self.src)	
		glo.carousel.remove_widget(self)
		for child in glo.my.children[:]:
			print(child.source)
			print(self.src)
			if(child.source == self.src):
				glo.my.remove_widget(child)
		
		#print(self.src)
		#carousel.index = carousel.index+1
		#my.remove_widget(self.kwargs.get('myimg'))