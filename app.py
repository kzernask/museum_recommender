from flask import Flask, render_template, request
from sklearn.externals import joblib
import os

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/recommend_museum', methods=['POST', 'GET'])
def recommend_museum():
    # get the parameters
    special_exhibit = 1 if request.form.get("special_exhibit", False) else 0
    interactive = 1 if request.form.get("interactive", False) else 0
    family = 1 if request.form.get("family", False) else 0
    audio = 1 if request.form.get("audio", False) else 0
    art = 1 if request.form.get("art", False) else 0
    display = 1 if request.form.get("display", False) else 0
    rainy = 1 if request.form.get("rainy", False) else 0
    natural_history = 1 if request.form.get("natural_history", False) else 0
    eye_opening = 1 if request.form.get("eye_opening", False) else 0
    learn = 1 if request.form.get("learn", False) else 0
    history = 1 if request.form.get("history", False) else 0
    permanent_collection = 1 if request.form.get("permanent_collection", False) else 0
    
    traveler_type = request.form['traveler_type']
    couple = 1 if traveler_type == "Couple" else 0
    family = 1 if traveler_type == "Family" else 0
    friends = 1 if traveler_type == "Friends" else 0
    solo = 1 if traveler_type == "Solo" else 0
    
    # load the scaler model and normalize data
    scaler = joblib.load('model/scaler.pkl')
    norm_data = scaler.transform([[special_exhibit, interactive, family, audio, art, display, rainy, natural_history, eye_opening, learn, history, permanent_collection, couple, family, friends, solo]])   
    
    # load the model and predict
    model = joblib.load('model/model.pkl')
    prediction = model.predict(norm_data)[0]
    
    clusters = {0: 'History Museums, Landmarks, Historic Sites',
                1: 'Natural History Museums',
                2: 'Art Museums, History Museums, Landmarks',
                3: "Science Museums, Children's Museums, Natural History Museums",
                4: 'Art Museums',
                5: 'Specialty Museums, History Museums, Military Museums',
                6: 'Specialty Museums, Art Museums',
                7: 'Specialty Museums'}
    
    recommendation = clusters[prediction]

    attributes = [special_exhibit, interactive, family, audio, art, display, rainy, natural_history, eye_opening, learn, history, permanent_collection]
    attribute_labels = ['Special Exhibits', 'Interactive', 'Family Friendly', 'Audio Guide', 'Works of Art', 'On Display', 'Rainy Day', 'Stuffed Animals', 'Eye Opening', 'Educational', 'Historical Artifacts', 'Permanent Collection']
    
    preferences = []
    for attribute, label in zip(attributes, attribute_labels):
        if attribute==1:
            preferences.append(label)
    preferences = ', '.join(preferences)

    return render_template('results.html',
                           traveler_type = traveler_type,
                           preferences = preferences,
                           recommendation = recommendation
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
