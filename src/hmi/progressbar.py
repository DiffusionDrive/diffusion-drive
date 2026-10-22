from kivy.app import App 
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.button import Button 
from kivy.uix.progressbar import ProgressBar 
from kivy.clock import Clock  

class ProgressApp(App): 
    def build(self): 
        self.layout = BoxLayout(orientation='vertical') 
        self.pb = ProgressBar(max=1000) 
        self.running = False  

        start_btn = Button(text="Start", on_press=self.start_progress) 
        stop_btn = Button(text="Stop", on_press=self.stop_progress)  

        self.layout.add_widget(self.pb) 
        self.layout.add_widget(start_btn) 
        self.layout.add_widget(stop_btn) 
        return self.layout  

    def start_progress(self, instance): 
        if not self.running: 
            self.running = True 
            self.event = Clock.schedule_interval(self.update_bar, 0.01)  

    def stop_progress(self, instance): 
        self.running = False 
        if hasattr(self, 'event'): 
            self.event.cancel()  

    def update_bar(self, dt): 
        self.pb.value = (self.pb.value + 10) % self.pb.max  

ProgressApp().run() 