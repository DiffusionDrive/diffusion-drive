from kivy.app import App 
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.textinput import TextInput 
from kivy.uix.label import Label 
from kivy.uix.button import Button  

class TextUpdateApp(App): 
    def build(self): 
        self.layout = BoxLayout(orientation='vertical') 
        self.input = TextInput(hint_text="Enter text here") 
        self.label = Label(text="Waiting for input...") 
        button = Button(text="Submit", on_press=self.update_label)  

        self.layout.add_widget(self.input) 
        self.layout.add_widget(button) 
        self.layout.add_widget(self.label) 
        return self.layout  

    def update_label(self, instance): 
        self.label.text = self.input.text  

TextUpdateApp().run() 