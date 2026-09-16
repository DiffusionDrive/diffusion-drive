# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

from kivymd.app import MDApp
from kivymd.uix.label import MDLabel


class MainApp(MDApp):
    def build(self):
        return MDLabel(text="Hello, World", halign="center")


MainApp().run()
