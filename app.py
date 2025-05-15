import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Settings
image_size = (50, 50)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model, label_names = joblib.load("fruit_model.pkl")

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB').resize(image_size)
    img_array = np.array(img).flatten().reshape(1, -1)
    pred = model.predict(img_array)[0]
    return label_names[pred]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            prediction = predict_image(image_path)
    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
