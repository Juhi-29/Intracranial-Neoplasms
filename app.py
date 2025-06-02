import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Prevent oneDNN initialization errors
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Reduce memory allocation issues
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

base_model = VGG19(include_top=False, input_shape=(240,240,3))
x = base_model.output
flat=Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights("vgg_unfrozen.h5")  # Ensure this path is correct
app = Flask(__name__)  # Fixed: Use __name__ instead of _name_

print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo==0:
        return "No Brain Tumor"
    elif classNo==1:
        return "Yes Brain Tumor"

def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model_03.predict(input_img)
    result01=np.argmax(result,axis=1)
    return result01

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # Also ensure __file__ is used elsewhere
        uploads_dir = os.path.join(basepath, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value)
        filename = secure_filename(f.filename)
        return render_template('result.html', prediction=result, filename=filename)
    return None

if __name__ == '__main__':  # Ensure this also uses __name__
    app.run(debug=True)