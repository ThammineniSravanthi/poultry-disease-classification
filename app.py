import numpy as np
import os
import tensorflow as tf
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("poultry_disease_model.h5")

# Index route
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        f = request.files['pc_image']
        if f:
            # Save uploaded image
            upload_folder = 'static/uploads/'
            os.makedirs(upload_folder, exist_ok=True)
            img_path = os.path.join(upload_folder, f.filename)
            f.save(img_path)

            # Preprocess image
            img = load_img(img_path, target_size=(224, 224))
            image_array = img_to_array(img) / 255.0  # ✅ Normalize
            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            pred = np.argmax(model.predict(image_array), axis=1)
            labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
            prediction = labels[int(pred)]

            return render_template("prediction_page.html", prediction=prediction)

    return render_template("prediction_page.html", prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
