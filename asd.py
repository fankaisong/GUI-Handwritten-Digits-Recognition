import tkinter as tk 



import numpy as np
from PIL import ImageDraw
import PIL
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

r = tk.Tk() 
r.title('Counting Seconds') 
button = tk.Button(r, text='Stop', width=25, command=r.destroy) 
button.pack() 
r.mainloop() 
