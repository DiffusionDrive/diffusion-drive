# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

from kivymd.app import MDApp
# from kivymd.uix.label import MDLabel
from kivy_garden.mapview import MapView

class MainApp(MDApp):
    def build(self):
        mapview = MapView(zoom=11, lat=50.6394, lon=3.057)
        return mapview


MainApp().run()
