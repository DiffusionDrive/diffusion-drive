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

    def return_home(self):
        self.map.center_on(54.7843, 9.43961)
        print(txt_input)

    def find_here(self):
        here = self.ids.here_input.text
        txt_input.append(geocode.geocoding(here))
        print(txt_input)
        self.map.add_widget(MapMarker(lat = txt_input[0][0], lon = txt_input[0][1]))
        txt_input.pop()

    def find_goto(self):
        goto = self.ids.goto_input.text
        txt_input.append(geocode.geocoding(goto))
        print(txt_input)
        self.map.add_widget(MapMarker(lat = txt_input[0][0], lon = txt_input[0][1], source = "goto.png"))
        txt_input.pop()

class Main(App):
    def build(self):
        Builder.load_file("main1.kv")
        return MapViewExample()

Main().run()