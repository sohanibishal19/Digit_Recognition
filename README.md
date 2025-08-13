# Digit_Recognition
A Python GUI application using Tkinter that lets users draw handwritten digits and recognizes them in real time using OpenCV’s KNN classifier trained on `digits.png`.

## Features
- Draw digits in a simple canvas.
- Recognizes digits via a pre-trained KNN model.
- Displays prediction and confidence score.
- Allows confirmation to record results.
- Saves predictions to `digit_predictions.xlsx`.
- Provides basic statistics: most frequent digit, average confidence, accuracy percentage.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- pandas
- Pillow (PIL)
- Tkinter (usually included in standard Python)
