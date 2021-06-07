from predict import predict
from PIL import Image
import os
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

from flask import Flask, request, jsonify


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    # return 'Hello World'
    return "Hello this sever is Running Celeba Dataset problem"


@app.route('/predict',methods=['POST'])
def foo():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    else:
        img = Image.open(request.files['file'])
        im1 = img.save("test.jpg")
        result = predict("test.jpg")
        print(result)

        return jsonify(result)


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True,host='0.0.0.0',port=port)
