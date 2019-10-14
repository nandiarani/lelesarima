from app import app
from flask import Flask,jsonify
import json
import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import *
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

@app.route('/forecastdata',methods=['GET'])
def forecast():
    URL = 'https://lele.learnme.web.id/api/dataRawForecast'
    headers = {'Accept': 'application/json'}
    r = requests.get(url=URL, headers=headers)
    data = json.loads(r.text)
    df = pd.io.json.json_normalize(data)

    month = df['month'].apply(str)
    year = df['year'].apply(str)
    y = month + '-01-' + year
    y = pd.to_datetime(y)
    x = df['jumlah_ikan'].apply(pd.to_numeric, errors='coerce')
    x_data = x.values
    y_date = list(pd.to_datetime(y))
    df = pd.DataFrame({'Date': y_date,
                       'Jumlah': x_data})
    df = df.set_index('Date')
    start = df.index[-3]
    end = start + relativedelta(months=+2)
    mod = sm.tsa.statespace.SARIMAX(df, order=(21, 2, 0), seasonal_order=(1, 0, 0, 12))
    results = mod.fit()
    pred = results.get_prediction(start=start, end=end, dynamic=True)

    #   mean
    mean = round(pred.predicted_mean, 0)

    index = list(mean.index.values)
    index = np.asarray(index)
    value = mean.values
    mIndex = []
    for x in index:
        x = pd.to_datetime(x)
        mIndex.append(str(x.date()))

    arrMean = mean.values.tolist()

    mArrMean = []
    i = 0
    for x in arrMean:
        mData = [x, mIndex[i]]
        mArrMean.append(mData)
        i += 1
    dfMean = pd.DataFrame(data=mArrMean, columns=['predict', 'period'])
    dfMean=dfMean.to_json(orient='records')
    return dfMean

