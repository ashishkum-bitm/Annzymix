from flask import Flask, render_template, request, redirect, url_for
from itertools import product
import pickle
import numpy as np

app = Flask(__name__)

with open('enzyme_predict_fungas.pkl', 'rb') as model_file:
    fungus_model = pickle.load(model_file)

with open('enzyme_predict_bact.pkl', 'rb') as model_file:
    bacteria_model = pickle.load(model_file)

with open('scaler_fungas.pkl', 'rb') as scaler_file:
    fungus_scaler = pickle.load(scaler_file)

with open('scaler_bact.pkl', 'rb') as scaler_file:
    bacteria_scaler = pickle.load(scaler_file)
    
def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_prediction', methods=['POST'])
def select_prediction():
    prediction_type = request.form['prediction_type']
    if prediction_type == 'fungus':
        return redirect(url_for('fungus_prediction'))
    elif prediction_type == 'bacteria':
        return redirect(url_for('bacteria_prediction'))
    elif prediction_type == 'fungus_opt':
        return redirect(url_for('fungus_optimization'))
    elif prediction_type == 'bacteria_opt':
        return redirect(url_for('bacteria_optimization'))

@app.route('/fungus_prediction')
def fungus_prediction():
    return render_template('fungus_prediction.html')

@app.route('/predict_fungus', methods=['POST'])
def predict_fungus():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    std_features = fungus_scaler.transform(features)
    prediction = fungus_model.predict(std_features)
    prediction_element = prediction.item()
    prediction_float = float(prediction_element)
    return render_template('result.html', prediction=prediction_float) 

@app.route('/bacteria_prediction')
def bacteria_prediction():
    return render_template('bacteria_prediction.html')

@app.route('/predict_bacteria', methods=['POST'])
def predict_bacteria():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    std_features = bacteria_scaler.transform(features)
    prediction = bacteria_model.predict(std_features)
    prediction_element = prediction.item()
    prediction_float = float(prediction_element)
    return render_template('result.html', prediction=prediction_float)

@app.route('/fungus_optimization')
def fungus_optimization():
    return render_template('fungus_optimization.html')

@app.route('/optimize_fungus', methods=['GET', 'POST'])
def optimize_fungus():
    if request.method == 'POST':
        incubation_time_min = int(request.form['incubation_time_min'])
        incubation_time_max = int(request.form['incubation_time_max'])
        pH_min = float(request.form['pH_min'])
        pH_max = float(request.form['pH_max'])
        agitation_speed_min = int(request.form['agitation_speed_min'])
        agitation_speed_max = int(request.form['agitation_speed_max'])
        temperature_min = int(request.form['temperature_min'])
        temperature_max = int(request.form['temperature_max'])
        carbon_min = float(request.form['carbon_min'])
        carbon_max = float(request.form['carbon_max'])
        nitrogen_min = float(request.form['nitrogen_min'])
        nitrogen_max = float(request.form['nitrogen_max'])
        incubation_time_increment_input = request.form['incubation_time_increment']
        pH_increment_input = request.form['pH_increment']
        agitation_speed_increment_input = request.form['agitation_speed_increment']
        temperature_increment_input = request.form['temperature_increment']
        carbon_increment_input = request.form['carbon_increment']
        nitrogen_increment_input = request.form['nitrogen_increment']

        if not incubation_time_increment_input:
            incubation_time_increment = 1
        else:
            incubation_time_increment = int(incubation_time_increment_input)

        if not pH_increment_input:
            pH_increment = 0.1  
        else:
            pH_increment = float(pH_increment_input)

        if not agitation_speed_increment_input:
            agitation_speed_increment = 1
        else:
            agitation_speed_increment = int(agitation_speed_increment_input)

        if not temperature_increment_input:
            temperature_increment = 1
        else:
            temperature_increment = int(temperature_increment_input)

        if not carbon_increment_input:
            carbon_increment = 0.1  
        else:
            carbon_increment = float(carbon_increment_input)

        if not nitrogen_increment_input:
            nitrogen_increment = 0.1
        else:
            nitrogen_increment = float(nitrogen_increment_input)

        ranges = {
            'incubation_time': range(incubation_time_min, incubation_time_max + incubation_time_increment, incubation_time_increment),
            'pH': frange(pH_min, pH_max + pH_increment, pH_increment),
            'agitation_speed': range(agitation_speed_min, agitation_speed_max + agitation_speed_increment, agitation_speed_increment),
            'temperature': range(temperature_min, temperature_max + temperature_increment, temperature_increment),
            'carbon': frange(carbon_min, carbon_max + carbon_increment, carbon_increment),
            'nitrogen': frange(nitrogen_min, nitrogen_max + nitrogen_increment, nitrogen_increment)
        }

        combinations = product(*ranges.values())

        predictions = []
        for comb in combinations:
            features = np.array(comb).reshape(1, -1)
            std_features = fungus_scaler.transform(features)
            prediction = fungus_model.predict(std_features)
            prediction_float = float(prediction.item())
            predictions.append((comb, prediction_float))

        best_combination = max(predictions, key=lambda x: x[1])

        return render_template('opt_result.html', best_combination=best_combination)
    else:
        return render_template('fungus_optimization.html')
    
@app.route('/bacteria_optimization')
def bacteria_optimization():
    return render_template('bacteria_optimization.html')

@app.route('/optimize_bacteria', methods=['GET', 'POST'])
def optimize_bactria():
    if request.method == 'POST':
        incubation_time_min = int(request.form['incubation_time_min'])
        incubation_time_max = int(request.form['incubation_time_max'])
        pH_min = float(request.form['pH_min'])
        pH_max = float(request.form['pH_max'])
        agitation_speed_min = int(request.form['agitation_speed_min'])
        agitation_speed_max = int(request.form['agitation_speed_max'])
        temperature_min = int(request.form['temperature_min'])
        temperature_max = int(request.form['temperature_max'])
        carbon_min = float(request.form['carbon_min'])
        carbon_max = float(request.form['carbon_max'])
        nitrogen_min = float(request.form['nitrogen_min'])
        nitrogen_max = float(request.form['nitrogen_max'])
        incubation_time_increment_input = request.form['incubation_time_increment']
        pH_increment_input = request.form['pH_increment']
        agitation_speed_increment_input = request.form['agitation_speed_increment']
        temperature_increment_input = request.form['temperature_increment']
        carbon_increment_input = request.form['carbon_increment']
        nitrogen_increment_input = request.form['nitrogen_increment']

        if not incubation_time_increment_input:
            incubation_time_increment = 1
        else:
            incubation_time_increment = int(incubation_time_increment_input)

        if not pH_increment_input:
            pH_increment = 0.1  
        else:
            pH_increment = float(pH_increment_input)

        if not agitation_speed_increment_input:
            agitation_speed_increment = 1
        else:
            agitation_speed_increment = int(agitation_speed_increment_input)

        if not temperature_increment_input:
            temperature_increment = 1
        else:
            temperature_increment = int(temperature_increment_input)

        if not carbon_increment_input:
            carbon_increment = 0.1  
        else:
            carbon_increment = float(carbon_increment_input)

        if not nitrogen_increment_input:
            nitrogen_increment = 0.1
        else:
            nitrogen_increment = float(nitrogen_increment_input)


        ranges = {
            'incubation_time': range(incubation_time_min, incubation_time_max + incubation_time_increment, incubation_time_increment),
            'pH': frange(pH_min, pH_max + pH_increment, pH_increment),
            'agitation_speed': range(agitation_speed_min, agitation_speed_max + agitation_speed_increment, agitation_speed_increment),
            'temperature': range(temperature_min, temperature_max + temperature_increment, temperature_increment),
            'carbon': frange(carbon_min, carbon_max + carbon_increment, carbon_increment),
            'nitrogen': frange(nitrogen_min, nitrogen_max + nitrogen_increment, nitrogen_increment)
        }

        combinations = product(*ranges.values())

        predictions = []
        for comb in combinations:
            features = np.array(comb).reshape(1, -1)
            std_features = bacteria_scaler.transform(features)
            prediction = bacteria_model.predict(std_features)
            prediction_float = float(prediction.item())
            predictions.append((comb, prediction_float))

        best_combination = max(predictions, key=lambda x: x[1])

        return render_template('opt_result.html', best_combination=best_combination)
    else:
        return render_template('bacteria_optimization.html')

if __name__ == '__main__':
    app.run(debug=True)