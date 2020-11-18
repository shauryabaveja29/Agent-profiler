from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
import imblearn
 
app = Flask(__name__)

model = load_model('knn_for_agent_score')
cols = ['Gender', 'Age Bracket', 'Age when insurance started Bracket', 'Education Level', 'Industry Experience bucket', 'Other Family Member Involved']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    data_dict = request.form.to_dict()
    for keys in data_dict.keys():
        data_dict[keys] = [str.strip(data_dict[keys])]
    #final = np.array(data_dict).reshape(1, 6) 
    data_unseen = pd.DataFrame(data_dict)
    data_unseen.columns = map(str.strip,data_unseen.columns)
    if (data_unseen.iloc[0]["Education Level"] == 'CA'):
        prediction ='Hire'
    else:    
        prediction = predict_model(model, data=data_unseen)
        prediction = prediction.Label[0]
        if (int(prediction)== 1): 
            prediction ='Hire'
        else: 
            prediction ="Don't Hire"
   # prediction = str(prediction)
    return render_template('index.html',prediction='Recommendation : {}'.format(prediction))
    #return render_template('predict.html',prediction= data_unseen)



if __name__ == '__main__':
    app.run(debug = False)