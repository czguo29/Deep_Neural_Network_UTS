from flask import Flask, render_template, request, redirect, url_for
import dnn
import os
from werkzeug.utils import secure_filename
import cv2


app = Flask(__name__)


@app.route('/', methods=['get', 'post'])
def mask_detection():
    if request.method == "POST":
        image = request.files.get("image")
        # prediction = dnn.input_predict(image, dnn.parameters)
        # print(prediction)
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('prediction'))
    return render_template('home.html')

# @app.route('/', methods=['get', 'post'])
# def mask_detection():
#     return render_template('home.html')

@app.route('/prediction', methods=['get', 'post'])
def prediction():
    prediction = None
    if request.method == "POST":
        image = request.files.get("image")
        prediction = dnn.input_predict(image, dnn.parameters)
        print(prediction)
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
    