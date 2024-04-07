import os
import sys
import datetime as dt
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from modules.pipeline import pipeline
from modules.predict import predict


path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)


args = {
    'owner': 'airflow',                                      # информация о владельцах
    'start_date': dt.datetime(2022, 6, 10),   # время начала выполнения пайплайна
    'retries': 1,                                            # количество повторений в случае неудач
    'retry_delay': dt.timedelta(minutes=1),                  # пауза между повторами
    'depends_on_past': False,                                # зависимость от успешного окончания предыдущего запуска
}

with DAG(
        dag_id='car_price_prediction',                  # имя дага
        schedule="00 15 * * *",                         # периодичность запуска
        default_args=args,                              # и множество других
) as dag:

    # data/modules/pipline.py - находит лучшую модуль и делает пикл-файл в data/models/cars_pipe_..
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    # data/modules/predict.py - выполняет предсказание используя модель data/models/cars_pipe_..
    # и сохраняет в файл data/predictions/predicted...
    predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
    )
    pipeline >> predict




