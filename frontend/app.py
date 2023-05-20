from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
import pickle
from tensorflow import keras


class DataType(BaseModel):
    Precipitation: float
    Salinity: float
    Evaporation: float

app = FastAPI()

"""
Sample JSON Input:- 
{
    "radius": 23,
    "texture": 12,
    "perimeter": 151,
    "area": 954,
    "smoothness": 0.143,
    "compactness": 0.278,
    "symmetry": 0.252,
    "fractal_dimension": 0.079
}
"""

def scale_data(data):
    min_max_list = [[123.4666667, 58.43333333], [663.114, 189.293], [89.0, 56.0], [17.5, 11.325]]
    for i,col in zip(range(len(min_max_list)),data.columns):
        X=data[col]
        X_scaled = abs((X - min_max_list[i][1]) / (min_max_list[i][0] - min_max_list[i][1]))
        data[col]=X_scaled
    return data

model = keras.models.load_model('model.h5')

@app.post("/predict")
async def predict(item: DataType):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    # df = df.values.reshape(-1, 1)
    data = scale_data(df)
    ans1 = model.predict([data])
    ans1 = list(ans1)
    print(type(ans1))
    print(ans1[0])
    return ""+str(ans1[0])
    """
    ans1 = int(ans1[0])
    ans1 = ans1 *  1000
    ans1 = ans1 + 35
    return ""+str(ans1)
    """
    # if ans1[0] == 0:
    #     return "Benign Prostatic Hyperplasia (BPH)"
    # else:
    #     return "Malignant Prostate Cancer (MPC)"

@app.get("/")
async def root():
    return {"message": "This API Only Has Get Method as of now"}


