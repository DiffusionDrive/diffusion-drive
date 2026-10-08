# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

# https://www.findlatitudeandlongitude.com/l/Nus+Singapore/4993767/#google_vignette 
# to find NUS lat and lon

# <a href="https://www.flaticon.com/free-icons/location" title="location icons">Location icons created by MEDZ - Flaticon</a>

from kivy import utils
from kivy.utils import get_color_from_hex

from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
from kivy_garden.mapview import MapView, MapMarker, MapMarkerPopup, MarkerMapLayer, MapSource

from kivymd.uix.label import MDLabel
from kivymd.uix.widget import MDWidget
from kivymd.uix.button import MDButton, MDButtonText

from kivymd.uix.relativelayout import MDRelativeLayout

Window.size = (375,667)

class Demo(ScreenManager):
    pass

class MainApp(MDApp):
    def build(self):
            Builder.load_file("layoutdemo.kv")
            return Demo()

MainApp().run()