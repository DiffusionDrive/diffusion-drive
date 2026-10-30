from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy_garden.mapview import MapView, MapMarker
import geocode

Window.size = (375,667)
here_coord = []
goto_coord = []
user_coord = []

class MapViewExample(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.map = self.ids.map
        self.goto_mark = None
        self.here_mark = None
        self.user_mark = None

    def return_home(self):
        self.map.center_on(54.7843, 9.43961)

    def find_here(self):
        if self.here_mark == None:
            here = self.ids.here_input.text
            here_coord.append(geocode.geocoding(here))
            self.here_mark = MapMarker(lat = here_coord[0][0], lon = here_coord[0][1])
            self.map.add_marker(self.here_mark)

            user_coord.append(here_coord[0])
            self.user_mark = MapMarker(lat = user_coord[0][0], lon = user_coord[0][1], source = 'driver.png')
            self.map.add_marker(self.user_mark)
            print('here coord = ' + str(here_coord))

    def find_goto(self):
        if self.goto_mark == None:
            goto = self.ids.goto_input.text
            goto_coord.append(geocode.geocoding(goto))
            self.goto_mark = MapMarker(lat = goto_coord[0][0], lon = goto_coord[0][1], source = 'goto.png')
            self.map.add_marker(self.goto_mark)
            print('goto coord = ' + str(goto_coord))

    def remove_here(self):
        if self.here_mark != None:
            self.map.remove_marker(self.here_mark)
            self.here_mark = None
            here_coord.pop()
            self.map.remove_marker(self.user_mark)
            self.user_mark = None
            user_coord.pop()
            print('here coord = ' + str(here_coord))

    def user_goto(self): #temporary function
        if self.user_mark != None:
            self.map.remove_marker(self.user_mark)
            self.user_mark = None
            user_coord.pop()
        
        user_coord.append(goto_coord[0])
        self.user_mark = MapMarker(lat = user_coord[0][0], lon = user_coord[0][1], source = 'driver.png')
        self.map.add_marker(self.user_mark)
        print('user had reached their destination!')

    def remove_goto(self):
        if self.goto_mark != None:
            self.map.remove_marker(self.goto_mark)
            self.goto_mark = None
            goto_coord.pop()
            print('goto coord = ' + str(goto_coord))

class Main(App):
    def build(self):
        Builder.load_file("main1.kv")
        return MapViewExample()

Main().run()