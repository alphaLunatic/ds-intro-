import os
import dill
import pandas as pd
import json
from pydantic import BaseModel
from datetime import datetime

def predict():
    # формат входных данных из data/test
    class FormIn(BaseModel):
        description: str
        fuel: str
        id: int
        image_url: str
        lat: float
        long: float
        manufacturer: str
        model: str
        odometer: float
        posting_date: str
        price: int
        region: str
        region_url: str
        state: str
        title_status: str
        transmission: str
        url: str
        year: float

    # формат выходных данных предсказания
    class FormOut(BaseModel):
        id: float
        Result: str

    path = os.path.expanduser('~/airflow_hw')

    # достаём модель
    with open(f'{path}/data/models/cars_pipe_202404031847.pkl', 'rb') as file:
        model = dill.load(file)

    # предсказание
    def predict(predict_parama: FormIn) -> FormOut:
        x = pd.DataFrame.from_dict([predict_parama])  # данные запроса из data/test/*
        y = model.predict(x)  # предсказание
        return dict(id=predict_parama['id'], Result=y[0])


    # загружаем data/test - json файлы
    lst = list()
    for file in os.listdir(f'{path}/data/test/'):
        with open(f"{path}/data/test/{file}", 'r') as file:
            data = json.load(file)
            next_pred = predict(data)
            lst.append(next_pred)

    df = pd.DataFrame(lst)
    df.set_index('id', inplace=True)

    df.to_csv(f'{path}/data/predictions/predicted_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
