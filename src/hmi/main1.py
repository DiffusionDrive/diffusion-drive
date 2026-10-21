from kivy.app import App
from kivy_garden.mapview import MapMarkerPopup, MapView, MapMarker
from kivy.uix.button import Button

class MainApp(App):
    def on_start(self):
        marker = MapMarkerPopup(lat=52.515,lon=13.404)
        self.root.add_widget(marker)
    pass
        

MainApp().run()