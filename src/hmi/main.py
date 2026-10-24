from kivy.app import App
from kivy_garden.mapview import MapView, MapMarkerPopup, MapMarker, MapSource
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget

class fscreen(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ids.main_map.zoom = 15
        self.ids.main_map.center_on(36.5924, 2.9898)

        self.pin = MapMarkerPopup(lat=36.59248, lon=2.9898, source='goto.png')
        self.ids.main_map.add_widget(self.pin)

class theapp(App):
    def build(self):

        self.screenm = ScreenManager()

        self.fscreen = fscreen()
        screen = Screen(name = 'first screen')
        screen.add_widget(self.fscreen)
        self.screenm.add_widget(screen)

        return self.screenm

if __name__ == "__main__":
    theapp = theapp()
    theapp.run()