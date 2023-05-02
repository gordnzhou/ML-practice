from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, ImageOps
import numpy as np

model = load_model('mnist.h5')

def predict(img):
    img = img.resize((28, 28)) # resize of 28x28 pixels
    img.convert('L') # convert to grayscale
    img = ImageOps.invert(img)
    # img.show()
    img = np.array(img)
    img = img[:,:,0]
    img = img / 255.0
    img = img.reshape((1, 28*28))

    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.start_pos = (0, 0)

        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Comic Sans", 48))
        self.classify_btn = tk.Button(self, text = "Predict", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    
    def clear_all(self):
        self.canvas.delete("all")
    
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)
        digit, accuracy = predict(im)
        self.label.configure(text= f"I see a {str(digit)}!")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill="black")

app = App()
mainloop()