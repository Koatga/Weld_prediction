from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) # http://127.0.0.1:5000/
def predict():
    if request.method == 'POST':

        # Загрузка моделей
        with open('models/model_xgboost_depth.pkl', 'rb') as f:      
            model_depth = pickle.load(f)
        with open('models/model_xgboost_width.pkl', 'rb') as f:
            model_width = pickle.load(f)

        # Получение данных из формы
        iw = float(request.form['iw'])
        i = float(request.form['i'])
        vw = float(request.form['vw'])
        fp = float(request.form['fp'])

        # Предсказываем глубину и ширину сварного шва
        depth = model_depth.predict(np.array([[iw, i, vw, fp]]))
        width = model_width.predict(np.array([[iw, i, vw, fp]]))

        # Передаем глубину и ширину на фронтэнд для отображения (пришлось прибегнуть к ужасному преобразованию, 
        # т.к. при других способах вывод идет в квадратных скобках)
        return render_template('index.html', predict_depth=np.round(float(depth), 2), predict_width=np.round(float(width), 2))
    

    return render_template('index.html')

#if __name__ == '__main__':
    app.run(debug=False)