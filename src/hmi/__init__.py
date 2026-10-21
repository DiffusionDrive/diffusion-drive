# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

# https://www.findlatitudeandlongitude.com/l/Nus+Singapore/4993767/#google_vignette 
# to find NUS lat and lon

# <a href="https://www.flaticon.com/free-icons/location" title="location icons">Location icons created by MEDZ - Flaticon</a>

from kivymd.app import MDApp ##

from kivy.core.window import Window
from kivy_garden.mapview import MapView, MapMarker, MapMarkerPopup, MarkerMapLayer, MapSource

from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.textinput import TextInput 
from kivymd.uix.label import MDLabel
from kivymd.uix.widget import MDWidget
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.relativelayout import MDRelativeLayout

from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

Window.size = (375,667)
Window.clearcolor = (200/255.0, 255/255.0, 151/255.0, 0.6)

class MainScreen(Screen):
    pass

class Screen1(Screen):
    def press(self):
        goto = self.ids.goto_input.text
        print(goto)

class Screen2(Screen):
    def press(self):
        here = self.ids.from_input.text
        print(here)

class Screen3(Screen):
    pass

class ScreenManagement(ScreenManager):
    pass

presentation = Builder.load_file("layoutdemo.kv")

class MainApp(MDApp):
    def build(self):
        return presentation

MainApp().run()