
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout 
import webbrowser
#from kivy.uix.button import Button

###################################################################
class KivyRoot(BoxLayout):
	"""docstring for KivyRoot"""
	def __init__(self, **kwargs):
		super(KivyRoot, self).__init__(**kwargs)

	def changeScreen(self, next_screen):
		if next_screen == 'about':
			self.ids.screen_manager.current = 'about_screen'


###################################################################
class KivyApp(App):
	"""docstring for KivyApp"""
	def __init__(self, **kwargs):
		super(KivyApp, self).__init__(**kwargs)

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


if __name__ == '__main__':
	KivyApp().run()
		