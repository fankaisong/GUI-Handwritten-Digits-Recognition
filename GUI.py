import numpy as np
# import matplotlib.pyplot as plt
from PIL import ImageDraw
import PIL
import tkinter as tk
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session






width = 300
height = 300


def paint(event):
    x1, y1 = (event.x-4), (event.y-4)
    x2, y2 = (event.x+4), (event.y+4)
    cv.create_line(x1, y1, x2, y2, fill="white",width=15)
    draw.line([x1, y1, x2, y2],fill="white",width=15)
    

def display():
    im = image1.convert('L')
    newImage = im.resize((28, 28), PIL.Image.BILINEAR)  
    # graph = np.array(newImage.getdata()).reshape(28,28)
    # plt.imshow(graph, cmap='gray')
    newImage.show()


def CNN():
    cnn = load_model('cnn.model')
    test =[]
    im = image1.convert('L')
    newImage = im.resize((28, 28), PIL.Image.BILINEAR)      
    # NEAREST      # use nearest neighbour
    # BILINEAR     # linear interpolation in a 2x2 environment
    # BICUBIC      # cubic spline interpolation in a 4x4 environment
    # ANTIALIAS    # best down-sizing filter
    x = np.array(newImage.getdata()).reshape(28,28,1)
    test.append(x)
    test = np.array(test)
    prediction = cnn.predict(test)
    res = prediction[0].argsort()[-1:][::-1] #return the index that has the highest probability 
    clear_session()
    del cnn
    
    value.set("The prediciton: "+ str(res[0]))
    
    
def reset():
    cv.delete('all')
    i = PIL.Image.new("RGB", (width, height),'black')
    image1.paste(i)
    
    value.set("Reset is done")
    
    


root = tk.Tk()
root.title('Handwritten Digits Recognition')
# create a canvas to draw on
cv = tk.Canvas(root, width=width, height=height, bg='black')


# PIL create an empty image and draw object to draw on memory only, not visible
image1 = PIL.Image.new("RGB", (width, height))
draw = ImageDraw.Draw(image1)

#create a label for display prediction
value = tk.StringVar()
value.set("Draw a number")
label=tk.Label(root,textvariable=value)

#create buttons 
button1=tk.Button(text="CNN Model",command=CNN)
button2=tk.Button(text="Display",command=display)
button3=tk.Button(text="Reset",command=reset)

cv.pack()
cv.bind("<B1-Motion>", paint)

label.pack()

button1.pack()
button2.pack()
button3.pack()

root.mainloop()

