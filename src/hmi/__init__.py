# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

# https://www.findlatitudeandlongitude.com/l/Nus+Singapore/4993767/#google_vignette 
# to find NUS lat and lon

# <a href="https://www.flaticon.com/free-icons/location" title="location icons">Location icons created by MEDZ - Flaticon</a>

from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy_garden.mapview import MapView, MapMarker, MapMarkerPopup, MarkerMapLayer, MapSource

from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.widget import MDWidget
from kivymd.uix.button import MDButton

Window.size = (375,667)

class MainApp(MDApp):
    def build(self):
        self.screen = Builder.load_file("layout.kv")
        return self.screen

MainApp().run()