import logging

import azure.functions as func
import numpy as np
import joblib
from sklearn import preprocessing

def normalize(X):
    scaler_class = preprocessing.StandardScaler().fit(X)
    normalized = scaler_class.transform(X)
    return normalized

def main(req: func.HttpRequest) -> func.HttpResponse:
    
    logging.info('Python HTTP trigger function processed a request.')
    
    regressor_model = joblib.load('model_desafio3_LGBMRegressor.pkl')
    
    X_received =  np.array(req.get_json()['data'])
    
    print("Valor solicitado: ", X_received[0][8])
    
    j_data = normalize(X_received)
    vec_pred = regressor_model.predict(j_data)
    
    
    if(vec_pred > X_received[0][8]):
        sol = np.format_float_positional(np.float64(X_received[0][8]), unique=False, precision=2)
        return func.HttpResponse(str(sol))
    
    pred = np.format_float_positional(np.float64(vec_pred), unique=False, precision=2)
    return func.HttpResponse(str(pred))
