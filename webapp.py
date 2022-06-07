import numpy as np
from flask import Flask, render_template,request
import pickle
import ML_model_app

app = Flask(__name__)

@app.route('/')
def home():
    names = pickle.load(open('names.pkl','rb'))
    return render_template('index.html', rests=names)#, result=model('Cinnamon'))

@app.route('/recommend', methods=['POST'])
def recommend():
    model = pickle.load(open('model.pkl','rb'))
    restaurant = request.form.get('rest')
    recommendation = model(restaurant)
    #recommendation = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                   #         columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return recommendation.to_html()


if __name__== "__main__":
    app.run()
