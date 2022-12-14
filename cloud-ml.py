import requests

API_KEY = "h4eVOrNTsRr2-OQ-neOXIY1NAfoDeL4CM2IFmoHgrmLW"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

payload_scoring = {"input_data": [{"fields": ["red_blood_cells","pus_cell","blood_glucose_random","blood_urea","pedal_edema","anemia","diabetesmellitus","coronary_artery_disease"], "values": [[1,1,93,66,0,0,1,0]]}]}

#payload_scoring = {"input_data": [{"fields": ["red_blood_cells","pus_cell","blood_glucose_random","blood_urea","pedal_edema","anemia","diabetesmellitus","coronary_artery_disease"], "values": [[1,1,87,38,0,0,0,0]]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/bf7ce76a-f691-49d6-bd07-1e5770377f49/predictions?version=2022-11-06', json=payload_scoring,headers=header)

print("Scoring response")
prediction = response_scoring.json()['predictions'][0]['values'][0][0]
if prediction == 1:
    prob = response_scoring.json()['predictions'][0]['values'][0][1][1]
else:
    prob = response_scoring.json()['predictions'][0]['values'][0][1][0]
print(prediction)
print(prob)