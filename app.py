from flask import Flask, request, render_template  
import numpy as np   
import pickle
import pandas as pd  

model = pickle.load(open('heart.pkl','rb'))
app = Flask(__name__, template_folder='Template')  
@app.route('/')  
def home():  
   return render_template("home.html")

@app.route("/predict", methods=["POST"])  
def predict():  
   age = request.form.get("age")  
   sex = request.form.get("sex")
   trestbps = request.form.get("trestbps")  
   chol = request.form.get("chol")  
   oldpeak = request.form.get("oldpeak")  
   thalach = request.form.get("thalach")  
   fbs = request.form.get("fbs")  
   exang = request.form.get("exang")  
   slope = request.form.get("slope")  
   cp = request.form.get("cp")
   thal = request.form.get("thal")  
   ca = request.form.get("ca")
   restecg = request.form.get("restecg")  
   arr = np.array([[age, sex, cp, trestbps,  
            chol, fbs, restecg, thalach,  
            exang, oldpeak, slope, ca,  
            thal]])  
   pred = model.predict(arr)  
   if pred == 0:  
     res_val = "NO HEART PROBLEM"  
   else:  
     res_val = "HEART PROBLEM"  
   return render_template('home.html', prediction_text='PATIENT HAS {}'.format(res_val))  
if __name__ == "__main__":  
   app.run(debug=True)  