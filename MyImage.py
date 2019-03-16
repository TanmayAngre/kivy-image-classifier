from kivy.uix.image import Image as img

import glo
class MyImage(img):
	ind=0
	def __init__(self, **kwargs):
		self.ind = kwargs.get('ind')
		super(MyImage, self).__init__(**kwargs)
		self.height = 200

	def on_touch_down(self, touch):
		if self.collide_point(*touch.pos):
			print(self.ind)
			glo.carousel.index=self.ind
			print(self.height)

	'''def onPress(self):
		print('hello')
		carousel.remove_widget(self)
		my.remove_widget(self)		
	'''