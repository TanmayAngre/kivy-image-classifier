from kivy.uix.screenmanager import Screen
import glo

class MyImageScreen(Screen):
	def __init__(self, **kwargs):
		super(MyImageScreen, self).__init__(**kwargs)
		self.add_widget(glo.carousel)
		#print(carousel.slides)