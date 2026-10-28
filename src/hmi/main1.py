from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy_garden.mapview import MapView, MapMarker
import geocode

Window.size = (375,667)
here_coord = []
goto_coord = []

class MapViewExample(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.map = self.ids.map

    def return_home(self):
        self.map.center_on(54.7843, 9.43961)

    def find_here(self):
        here = self.ids.here_input.text
        here_coord.append(geocode.geocoding(here))
        self.map.add_widget(MapMarker(lat = here_coord[0][0], lon = here_coord[0][1]))
        self.map.add_widget(MapMarker(lat = here_coord[0][0], lon = here_coord[0][1], source = 'driver.png'))
        here_coord.pop()

    def find_goto(self):
        goto = self.ids.goto_input.text
        goto_coord.append(geocode.geocoding(goto))
        self.map.add_widget(MapMarker(lat = goto_coord[0][0], lon = goto_coord[0][1], source = "goto.png"))
        goto_coord.pop()

class Main(App):
    def build(self):
        Builder.load_file("main1.kv")
        return MapViewExample()

Main().run()