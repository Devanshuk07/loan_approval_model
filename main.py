import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd

app=FastAPI()

model=joblib.load('loan_model.pkl')

@app.get('/')
def home():
    return FileResponse('index.html')

@app.get('/predict')
def predict(inc:float,lam:float,crd:float,ag:float,rtm:float):
    nwdata=pd.DataFrame({
    'Income':[inc],
    'Loan_Amount':[lam],
    'Credit_Score':[crd],
    'Age':[ag],
    'Repayment_Tenure_Months':[rtm]
    })
    nwdtres=model.predict(nwdata)
    print(nwdtres[0])
    if nwdtres[0]=='Yes':
        result='YES !! ELIGIBLE FOR LOAN GRANTING...'
    else:
        result='NO YOU CANNOT GET LOAN SORRYY ...'
    return result

