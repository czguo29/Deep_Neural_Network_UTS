from flask import Flask, render_template, request, redirect, session
import dnn


app = Flask(__name__)


@app.route('/', methods=['get', 'post'])
def sign_in():
    if request.method == "POST":
        image = request.form(["image"])
        prediction = dnn.input_predict(image, dnn.parameters)
        print(prediction)
    return render_template('home.html', predict=prediction)


if __name__ == '__main__':
    app.run(debug=True)
    