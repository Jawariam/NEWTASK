import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

    	int_features = request.form.get('petalLength')
    	final_features = [[np.array(int_features)]]
    	prediction = model.predict(final_features)[0][0]


    	return render_template('index.html', prediction_text='The predicted petal width using my linear regression model is: {}'.format(prediction))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)