# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

# https://www.findlatitudeandlongitude.com/l/Nus+Singapore/4993767/#google_vignette 
# to find NUS lat and lon

# https://time-ok.com/coordinates/flensburg
# Flensburg co-ords

# https://www.cleanpng.com/png-logo-brand-coupon-code-business-location-icon-dvlxs3/
# goto map marker

# https://www.cleanpng.com/png-computer-icons-clip-art-vector-graphics-portable-n-4i929o/
# reached map marker

from kivymd.app import MDApp ##

from kivy.core.window import Window
from kivy_garden.mapview import MapView, MapMarker, MapMarkerPopup, MarkerMapLayer, MapSource
'''
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.textinput import TextInput 
from kivymd.uix.label import MDLabel
from kivymd.uix.widget import MDWidget
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.relativelayout import MDRelativeLayout
'''
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

from kivy.clock import Clock

Window.size = (375,667)
Window.clearcolor = (200/255.0, 255/255.0, 151/255.0, 0.6)
txt_input = []

class MainScreen(Screen): # map screen (home page)
    pass

class Screen1(Screen): # goto (end goal) screen
    def press(self):
        goto = self.ids.goto_input.text
        print(goto)

        self.ids.goto_input.text = ''
        txt_input.append(goto)

class Screen2(Screen): # from (start goal) screen
    def press(self):
        here = self.ids.from_input.text
        print(here)

        self.ids.from_input.text = ''
        txt_input.append(here)

class Screen3(Screen): # mark (two goal confirmation) screen <to have initial planned route shown>
    def coord(self):
        s_lat = self.ids.start_mark.lat
        s_lon = self.ids.start_mark.lon
        print(s_lat, s_lon)
        e_lat = self.ids.end_mark.lat
        e_lon = self.ids.end_mark.lon
        print(e_lat, e_lon)
        print(txt_input)

class Screen4(Screen): # moving (progress + detours (if any)) screen
    def next(self, dt):
        current = self.ids.my_progress_bar.value
        if current < 1:
            current += 0.01
            self.ids.my_progress_bar.value = current
        if current == 1:
            current = 0
            self.ids.my_progress_bar.value = current
    
    def start_timer(self):
        Clock.schedule_interval(self.next, 1/25)

class Screen5(Screen): # reached (confirmation (successful run)) screen, can loop back to 'map' home page
    pass

class ScreenManagement(ScreenManager):
    pass

presentation = Builder.load_file("layoutdemo.kv")

class MainApp(MDApp):
    def build(self):
        return presentation

MainApp().run()