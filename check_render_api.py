"""
Render Api module test
"""
import requests


data = {
    "age": 32,
    "hours_per_week": 60,
    "workclass": "Private",
    "education": "Some-college",
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "native_country": "United-States"
}
r = requests.post('https://mlops-app.onrender.com/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
