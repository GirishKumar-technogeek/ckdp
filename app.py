import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle
import requests

app = Flask(__name__)
model = pickle.load(open('CKD_NLP.pkl','rb'))
# with gzip.open("CKD_NLP.pkl", 'rb') as f:
#     model = pickle.load(f, fix_imports=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=["POST"])
def prediction():
    red_blood_cells = int(request.form['red_blood_cells'])
    pus_cell = int(request.form['pus_cell'])
    blood_glucose_random = float(request.form['blood_glucose_random'])
    blood_urea = float(request.form['blood_urea'])
    pedal_edema = int(request.form['pedal_edema'])
    anemia = int(request.form['anemia'])
    diabetesmellitus = int(request.form['diabetesmellitus'])
    coronary_artery_disease = int(request.form['coronary_artery_disease'])
    input_features = list()
    input_features.append(red_blood_cells)
    input_features.append(pus_cell)
    input_features.append(blood_glucose_random)
    input_features.append(blood_urea)
    input_features.append(pedal_edema)
    input_features.append(anemia)
    input_features.append(diabetesmellitus)
    input_features.append(coronary_artery_disease)
    features_value = [np.array(input_features)]
    features_name = ["red_blood_cells","pus_cell","blood_glucose_random","blood_urea","pedal_edema","anemia","diabetesmellitus","coronary_artery_disease"]
    API_KEY = "h4eVOrNTsRr2-OQ-neOXIY1NAfoDeL4CM2IFmoHgrmLW"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    payload_scoring = {"input_data": [{"fields": features_name, "values": [features_value]}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/bf7ce76a-f691-49d6-bd07-1e5770377f49/predictions?version=2022-11-06', json=payload_scoring,headers=header)
    prediction = response_scoring.json()['predictions'][0]['values'][0][0]
    if prediction == 1:
        prob = response_scoring.json()['predictions'][0]['values'][0][1][1]
    else:
        prob = response_scoring.json()['predictions'][0]['values'][0][1][0]
    print(prediction,prob)
    if prediction == 0:
        return render_template('predictionckd.html',{'prob':prob})
    else:
        return render_template('predictionnockd.html',{'prob':prob})

if __name__ == '__main__':
    app.run(debug=True)