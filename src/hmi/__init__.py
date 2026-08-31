# https://www.youtube.com/watch?v=POwi9Don3pE
# test for successful Kivy installation

from kivy.app import App
from kivy.uix.button import Button


class MyApp(App):
    def build(self):
        return Button(text="Click Me!", on_press=self.on_button_click)

    def on_button_click(self, instance):
        instance.text = "Clicked!"


if __name__ == "__main__":
    MyApp().run()
