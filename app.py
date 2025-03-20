import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
##Load the model

regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    # print(np.array(list(data.values())).reshape(1,-1))
    data_array = np.array(list(data.values())).reshape(1, -1)

    if data_array.shape[1] != 13:  # Ensure 13 features
        return jsonify({"error": "Expected 13 features, but got {}".format(data_array.shape[1])})
    # new_data=scalar.transform(np.array(list(data.values)).reshape(1,-1))
    new_data = scalar.transform(data_array)
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_data=scalar.transform(np.array(data).reshape(1,-1))
    print(final_data)
    output=regmodel.predict(final_data)[0]
    return render_template('home.html',predict_text="The House Price Predicted is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)