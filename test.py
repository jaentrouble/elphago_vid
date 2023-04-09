from flask import Flask, request, jsonify
import numpy as np


app = Flask(__name__)
data_array = np.load('my_data.npy')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json # get the data from the request body
    index = tuple(data['index']) # get the index from the data
    value = data_array[index] # get the value from the data array
    return jsonify(value.tolist()) # return the prediction as a JSON object

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # run the app on port 5000


import requests

index = [1, 2, 3] # the index of the data we want to retrieve
data = {'index': index} # the data we want to send to the server
response = requests.post('http://localhost:5000/predict', json=data) # send the data to the server
prediction = response.json() # get the prediction from the server
print(prediction) # print the prediction