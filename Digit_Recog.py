import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw

try:
    gray = cv2.imread(cv2.samples.findFile('digits.png'), cv2.IMREAD_GRAYSCALE)
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)  # shape: (50,100,20,20)
    
    train = x[:, :50].reshape(-1, 400).astype(np.float32)  # 2500 samples
    resp = np.repeat(np.arange(10), 250)[:, np.newaxis].astype(np.float32)
    
    model = cv2.ml.KNearest_create()
    model.train(train, cv2.ml.ROW_SAMPLE, resp)
    samples = train  
except Exception as e:
    print("Training failed:", e)
    model = None
    samples = None

data = pd.DataFrame(columns=["Timestamp", "Predicted Digit", "Confidence", "Confirmed"])
current_prediction = None
current_confidence = None

def predict_digit():
    global current_prediction, current_confidence
    if model is None:
        messagebox.showerror("Error", "Model not trained.")
        return

    img_small = image.copy().resize((20, 20))
    arr = np.array(img_small)
    _, thresh = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)
    sample = thresh.reshape(1, -1).astype(np.float32)

    if sample.shape[1] != samples.shape[1]:
        messagebox.showerror("Error", f"Expected {samples.shape[1]} features, got {sample.shape[1]}.")
        return

    ret, res, neigh, dist = model.findNearest(sample, k=3)
    current_prediction = int(res[0][0])
    current_confidence = float(1 / (1 + dist[0][0]))
    messagebox.showinfo("Prediction", f"Digit: {current_prediction}\nConfidence: {current_confidence:.2f}")

def confirm_prediction():
    global data
    if current_prediction is None:
        messagebox.showwarning("Warning", "No prediction made yet.")
        return
    data.loc[len(data)] = [datetime.now(), current_prediction, current_confidence, True]
    data.to_excel("digit_predictions.xlsx", index=False)
    messagebox.showinfo("Saved", "Prediction confirmed and saved!")

def view_stats():
    if data.empty:
        messagebox.showinfo("Stats", "No data recorded.")
        return
    most_common = int(data["Predicted Digit"].mode()[0])
    avg_conf = data["Confidence"].mean()
    accuracy = data["Confirmed"].mean() * 100
    messagebox.showinfo("Stats",
        f"Most Frequent: {most_common}\nAccuracy: {accuracy:.2f}%\nAvg Confidence: {avg_conf:.2f}")

def draw(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    draw_img.ellipse([x-r, y-r, x+r, y+r], fill='black')

def clear_canvas():
    canvas.delete('all')
    global image, draw_img
    image = Image.new("L", (200, 200), 'white')
    draw_img = ImageDraw.Draw(image)

    
root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(root, width=200, height=200, bg='white')
canvas.grid(row=0, column=0, columnspan=4)
image = Image.new("L", (200, 200), 'white')
draw_img = ImageDraw.Draw(image)
canvas.bind("<B1-Motion>", draw)

tk.Button(root, text="Predict", command=predict_digit).grid(row=1, column=0)
tk.Button(root, text="Confirm", command=confirm_prediction).grid(row=1, column=1)
tk.Button(root, text="View Stats", command=view_stats).grid(row=1, column=2)
tk.Button(root, text="Clear", command=clear_canvas).grid(row=1, column=3)

root.mainloop()
