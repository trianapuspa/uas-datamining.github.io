from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       
        usia = float(request.form['Usia'])
        status = float(request.form['Status'])
        kelamin = float(request.form['Kelamin'])
        memilikiMobil = float(request.form['Memiliki_Mobil'])
        penghasilan = float(request.form['Penghasilan'])

        input_data = [[usia, status, kelamin, memilikiMobil, penghasilan]]

        with open('decision_tree_model.pkl', 'rb') as model_file:
            decision_tree = pickle.load(model_file)

        with open('logistic_regression_model.pkl', 'rb') as model_file:
            logistic_regression = pickle.load(model_file)

        decision_tree_prediction = decision_tree.predict(input_data)[0]

        logistic_regression_prediction = logistic_regression.predict(input_data)[0]

        prediction_results = {
            'decision_tree': decision_tree_prediction,
            'logistic_regression': logistic_regression_prediction
        }

        return render_template('index.html', prediction_results=prediction_results)

    else:
        error_message = "An error occurred. Please check your input."
        return render_template('index.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
