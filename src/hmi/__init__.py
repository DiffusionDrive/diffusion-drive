# https://kivymd.readthedocs.io/en/latest/getting-started/
# test for successful KivyMD installation

# https://github.com/kivy-garden/mapview
# test for successful MapView integration

# https://www.findlatitudeandlongitude.com/l/Nus+Singapore/4993767/#google_vignette 
# to find NUS lat and lon

from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy_garden.mapview import MapView, MapMarker

class LayoutMapView(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.map = self.ids.map
        self.map.add_widget(MapMarker(lat = 1.296, lon = 103.776))
        
class MainApp(MDApp):
    def build(self):
        Builder.load_file("layout.kv")
        return LayoutMapView()

MainApp().run()