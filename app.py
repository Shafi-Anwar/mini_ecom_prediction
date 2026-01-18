from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from fastapi.responses import FileResponse



model = joblib.load("model.pkl")    
ohe = joblib.load("ohe.pkl")          
scaler = joblib.load("scaler.pkl")  
cat_cols = ['Gender','Country','BrowsedCategory','Price_range','Age_group','HighViewedTime']
num_cols = ['Age','ViewedTimeMin','ProductPrice','Age_log','ViewedTimeMin_log','ProductPrice_sqrt']
def preprocess_input(df_input):
    # feature engineering
    df_input['Age_log'] = np.log1p(df_input['Age'])
    df_input['ViewedTimeMin_log'] = np.log1p(df_input['ViewedTimeMin'])
    df_input['ProductPrice_sqrt'] = np.sqrt(df_input['ProductPrice'])
    
    # categorical
    X_cat = ohe.transform(df_input[cat_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(cat_cols))
    
    # numerical
    X_num = scaler.transform(df_input[num_cols])
    X_num_df = pd.DataFrame(X_num, columns=num_cols)
    
    # concat
    X_processed = pd.concat([X_num_df.reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1)
    return X_processed

app = FastAPI(title="Ecommerce Purchase Prediction App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return FileResponse("index.html")


class UserData(BaseModel):
    Age: int
    Gender: str
    Country: str
    BrowsedCategory: str
    ViewedTimeMin: float
    ProductPrice: float
    Price_range: str
    Age_group: str
    HighViewedTime: str

@app.post("/predict")
def predict_purchase(data: UserData):

    df_input = pd.DataFrame([data.dict()])
    X_processed = preprocess_input(df_input)

    prob = model.predict_proba(X_processed)[0][1]
    
    prediction = int(prob > 0.5)
    
    return {
        "prediction": prediction,
        "probability": round(prob, 3)
    }
