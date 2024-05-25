from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)


def load_my_model(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except OSError as e:
        print(f"Error loading model: {e}")
        return None


model = load_my_model('model.h5')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_file.save('image.jpg')

    image = tf.keras.preprocessing.image.load_img(
        'image.jpg', target_size=(32, 32))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.expand_dims(input_arr, 0)

    prediction = model.predict(input_arr)
    predicted_class = prediction.argmax()
    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = class_names[predicted_class]
    os.remove('image.jpg')

    return "<p id='prediction' class='flex items-center justify-center w-96 p-4 mt-4 bg-white shadow-md rounded-lg text-gray-800 font-bold'>" + class_name.capitalize() + "</p>"
