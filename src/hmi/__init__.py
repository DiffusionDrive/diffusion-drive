# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

# https://www.findlatitudeandlongitude.com/l/Nus+Singapore/4993767/#google_vignette 
# to find NUS lat and lon

from kivymd.app import MDApp
# from kivymd.uix.label import MDLabel
from kivy_garden.mapview import MapView

class MainApp(MDApp):
    def build(self):
        mapview = MapView(zoom=17, lat=1.296, lon=103.776)
        return mapview


MainApp().run()
