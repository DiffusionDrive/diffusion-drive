# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

# https://www.findlatitudeandlongitude.com/l/Nus+Singapore/4993767/#google_vignette 
# to find NUS lat and lon

from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy_garden.mapview import MapView, MapMarker, MapMarkerPopup, MarkerMapLayer

Window.size = (375,750)

class MainApp(MDApp):
    def build(self):
        self.screen = Builder.load_file("layout.kv")
        return self.screen

MainApp().run()