from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy_garden.mapview import MapView, MapMarker
import geocode

Window.size = (375,667)
txt_input = []

class MapViewExample(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.map = self.ids.map
        self.map.add_widget(MapMarker(lat = 54.785, lon = 9.441, source = "goto.png"))

    def return_home(self):
        self.map.center_on(54.7843, 9.43961)
        print(txt_input)

    def find_here(self):
        here = self.ids.here_input.text
        txt_input.append(geocode.geocoding(here))

    def find_goto(self):
        goto = self.ids.goto_input.text
        txt_input.append(geocode.geocoding(goto))

class Main(App):
    def build(self):
        Builder.load_file("main1.kv")
        return MapViewExample()

Main().run()