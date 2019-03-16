from kivy.app import App
from kivy.core.window import Window


import KivyRoot
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
		return KivyRoot.KivyRoot()