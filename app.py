from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and encoders
with open("models/random_forest_model.pkl", "rb") as model_file:
    my_rf_model = pickle.load(model_file)

with open("models/label_encoders.pkl", "rb") as encoder_file:
    my_encoders = pickle.load(encoder_file)

categorical_features = ["HomeTeam", "AwayTeam"]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get data from the form
        user_input = {
            "HomeTeam": request.form["HomeTeam"],
            "AwayTeam": request.form["AwayTeam"],
            "FTHG": float(request.form["FTHG"]),
            "FTAG": float(request.form["FTAG"]),
            "HS": float(request.form["HS"]),
            "AS": float(request.form["AS"]),
            "HST": float(request.form["HST"]),
            "AST": float(request.form["AST"]),
            "HC": float(request.form["HC"]),
            "AC": float(request.form["AC"]),
            "HF": float(request.form["HF"]),
            "AF": float(request.form["AF"]),
            "HY": float(request.form["HY"]),
            "AY": float(request.form["AY"]),
            "HR": float(request.form["HR"]),
            "AR": float(request.form["AR"]),
            "Year": int(request.form["Year"]),
            "Month": int(request.form["Month"]),
            "Day": int(request.form["Day"]),
        }

        # Convert to DataFrame
        user_df = pd.DataFrame([user_input])

        HomeTeam = str(user_df['HomeTeam'].iloc[0])
        AwayTeam = str(user_df['AwayTeam'].iloc[0])

        # Encode categorical features
        for feature in categorical_features:
            user_df[feature] = user_df[feature].map(lambda x: my_encoders[feature].transform([x])[0] if x in my_encoders[feature].classes_ else -1)

        # Ensure column order matches training data 
        # user_df = user_df[X_train.columns]
        user_df = user_df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
       'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY',
       'AY', 'HR', 'AR', 'Year', 'Month', 'Day']]
    
        # Make prediction
        prediction = my_rf_model.predict(user_df)
        result_mapping = {1: "Home Team Win", 2: "Away Team Win", 3: "Draw the Match"}
        result = result_mapping[prediction[0]]


        message=''
        if prediction[0]==1:
            message=f'Winner Team is {HomeTeam}'
        elif prediction[0]==2:
            message=f'Winner Team is {AwayTeam}'
        elif prediction[0]==3:
            message=''
        else:
            message='Invalid Result'

        return render_template("prediction-result.html", result=result, message=message)
    return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=True)
