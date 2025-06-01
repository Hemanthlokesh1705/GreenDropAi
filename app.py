from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from gtts import gTTS
from gtts.tts import gTTSError
from flask_cors import CORS 
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)  


try:
    crop_df = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\imd_maitri.csv')
    crop_df = crop_df.rename(columns={'District': 'city'})
    crop_df['city'] = crop_df['city'].str.strip().str.lower()
except Exception as e:
    print(f"Error loading crop data: {e}")
 
    crop_df = pd.DataFrame(columns=['city', 'SoilType', 'Texture', 'pH'])

def predict_next_5_years(city_name):
    rainfall = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\160a665a7b6d47e19e01105ebc5a46ec..csv')
    groundwater = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\groundwater level.csv')
    temperature = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\temp.csv')
    population = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\pop.csv')
    drought = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\drought percentage.csv')

    def melt_city(df, value_name):
        return df.melt(id_vars='year', var_name='city', value_name=value_name)

    rain_long = melt_city(rainfall, 'rainfall')
    ground_long = melt_city(groundwater, 'groundwater')
    temp_long = melt_city(temperature, 'temperature')
    pop_long = melt_city(population, 'population')
    drought_long = melt_city(drought, 'drought_percentage')

    df = drought_long.merge(rain_long, on=['year', 'city'], how='inner')
    df = df.merge(ground_long, on=['year', 'city'], how='inner')
    df = df.merge(temp_long, on=['year', 'city'], how='inner')
    df = df.merge(pop_long, on=['year', 'city'], how='inner')

    df = df[df['city'].str.lower() == city_name.lower()].copy()
    
    if df.empty:
        raise ValueError(f"No data available for city: {city_name}")
        
    avg_temp = round(df['temperature'].mean(), 2)
    avg_population = round(df['population'].mean(), 2)

    df['rainfall_per_person'] = df['rainfall'] / (df['population'] + 1)
    df['rain_groundwater_ratio'] = df['rainfall'] / (df['groundwater'] + 1)
    df['temperature_rolling_3'] = df['temperature'].rolling(window=3, min_periods=1).mean()

    df.dropna(inplace=True)

    features = ['rainfall', 'rainfall_per_person', 'temperature', 'temperature_rolling_3', 'rain_groundwater_ratio']
    X = df[features]
    y = df['drought_percentage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    simplified_accuracy = 100 - (rmse / y.mean() * 100)

    importances = model.feature_importances_
    top_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:5]

    last_row = df.iloc[-1]
    last_year = int(last_row['year'])

    future_data = []
    for i in range(1, 6):
        year = last_year + i
        rainfall_pred = last_row['rainfall'] * (1 + np.random.uniform(-0.05, 0.05))
        temperature_pred = last_row['temperature'] * (1 + np.random.uniform(-0.02, 0.02))
        groundwater_pred = last_row['groundwater'] * (1 + np.random.uniform(-0.05, 0.05))
        population_pred = last_row['population'] * (1 + 0.015 * i)

        rainfall_per_person = rainfall_pred / (population_pred + 1)
        rain_groundwater_ratio = rainfall_pred / (groundwater_pred + 1)
        temperature_rolling = (last_row['temperature'] + temperature_pred) / 2

        X_future = pd.DataFrame([{
            'rainfall': rainfall_pred,
            'rainfall_per_person': rainfall_per_person,
            'temperature': temperature_pred,
            'temperature_rolling_3': temperature_rolling,
            'rain_groundwater_ratio': rain_groundwater_ratio
        }])

        predicted_drought = model.predict(X_future)[0]

        future_data.append({
            'year': year,
            'predicted_drought_percentage': round(predicted_drought, 2),
            'predicted_groundwater': round(groundwater_pred, 2),
            'predicted_rainfall': round(rainfall_pred, 2)
        })

    return {
        "city": city_name.title(),
        "average_temperature": avg_temp,
        "average_population": avg_population,
        "model_performance": {
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 2),
            "Simplified_Accuracy": round(simplified_accuracy, 1)
        },
        "top_factors": [{k: round(v, 3)} for k, v in top_features],
        "predictions": future_data
    }

def generate_audio(city, prediction_data):
    """Generate audio file for the prediction data and return the file path or None if failed"""
    try:
        # Format the text in Kannada
        kannada_text = f"""
        {city} ನಗರದ ಬಗ್ಗೆ ಮುಂಗಾಣಲು ಪ್ರಯತ್ನಿಸುತ್ತಿದೆ. 
        ಸರಾಸರಿ ತಾಪಮಾನ {prediction_data['average_temperature']} ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್.
        ಸರಾಸರಿ ಜನಸಂಖ್ಯೆ {prediction_data['average_population']} ಲಕ್ಷ.
        ಮುಂದಿನ 5 ವರ್ಷಗಳ ಬರಗಾಲದ ಸಾಧ್ಯತೆ:
        """
        
        for year in prediction_data['predictions']:
            kannada_text += f"""
            {year['year']}ನೇ ವರ್ಷ: ಬರಗಾಲ {year['predicted_drought_percentage']}%, 
            ಮಳೆ {year['predicted_rainfall']} ಮಿಮೀ, 
            ಭೂಗತ ನೀರು {year['predicted_groundwater']} ಮೀಟರ್.
            """
        
        # Clean city name for filename
        safe_city = city.strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
        audio_filename = f'drought_{safe_city}.mp3'
        
        # Ensure directory exists
        audio_dir = os.path.join(app.root_path, 'static', 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Generate and save the audio file
        tts = gTTS(text=kannada_text, lang='kn', slow=False)
        tts.save(audio_path)
        print(f"Audio saved successfully at: {audio_path}")
        
        # Return the relative path to be used in templates
        return f'/static/audio/{audio_filename}'
    
    except gTTSError as e:
        print(f"gTTS Error: {e}")
        print(f"Text that caused the error: {kannada_text[:100]}...")
        return None
    except Exception as e:
        print(f"Unexpected audio generation error: {e}")
        print(traceback.format_exc())
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def drought():
    if request.method == 'POST':
        try:
            # Check if the request has JSON content
            if request.is_json:
                data = request.get_json()
                city = data.get('city', '')
                if not city:
                    return jsonify({"error": "City name is required"}), 400
                
                result = predict_next_5_years(city)
                audio_file = generate_audio(city, result)
                result['audio_file'] = audio_file or None
                return jsonify(result)
            else:
                # Handle form submission if not JSON
                city = request.form.get('city', '')
                if not city:
                    return jsonify({"error": "City name is required"}), 400
                
                result = predict_next_5_years(city)
                audio_file = generate_audio(city, result)
                return render_template('predict.html', 
                                     city=city,
                                     prediction=result,
                                     audio_file=audio_file)
                
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            print(f"Error in POST request: {e}")
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
    else:
        # Handle GET request
        city = request.args.get('city', '')
        if not city:
            return render_template('predict.html')

        try:
            # Get prediction data
            prediction_data = predict_next_5_years(city)
            
            # Generate audio file
            audio_file = generate_audio(city, prediction_data)
            
            # Render template with or without audio depending on success
            return render_template('predict.html', 
                                 city=city,
                                 prediction=prediction_data,
                                 audio_file=audio_file)
        
        except ValueError as e:
            return render_template('predict.html', error=str(e))
        except Exception as e:
            print(f"Error processing request: {e}")
            print(traceback.format_exc())
            return render_template('predict.html', error=str(e))

@app.route('/blog')
def blog():
    return render_template('blog.html')
client=MongoClient('mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.4.2')
db=client['Forms']
collection=db['Submission']
@app.route('/form')
def form():
    return render_template('form.html')
@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.get_json()
    name = data.get('name')
    district = data.get('district')
    crops = data.get('crops')
    crop_yield = data.get('yield')
    description = data.get('description')

    if not all([name, district, crops, crop_yield, description]):
        return jsonify({"error": "Missing required fields"}), 400

    collection.insert_one({
        "Name": name,
        "District": district,
        "Crop": crops,
        "Yield": crop_yield,
        "Description": description
    })

    return jsonify({"message": "Data submitted successfully"}), 200
 
  
@app.route('/crop')
def details():
    city = request.args.get('city', '')
    return render_template('crop.html', city=city)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    city_input = request.args.get('city', '').strip().lower()
    city_data = crop_df[crop_df['city'] == city_input]
    
    if city_data.empty:
        similar = crop_df[crop_df['city'].str.contains(city_input, case=False)]
        suggestions = similar['city'].str.title().unique().tolist()
        return jsonify({'error': 'City not found', 'suggestions': suggestions}), 404

    info = city_data.iloc[0]
    recommendations = [
        {"success_rate": int(col.replace('Crop_', '').replace('%', '')), "crop": info[col]}
        for col in crop_df.columns if col.startswith('Crop_')
    ]
    
    return jsonify({
        'city': info['city'].title(),
        'soil_type': info['SoilType'],
        'texture': info['Texture'],
        'ph': info['pH'],
        'recommendations': recommendations
    })

@app.route('/test-audio')
def test_audio():
    try:
        test_text = "ಪರೀಕ್ಷೆ. ಇದು ಕನ್ನಡದಲ್ಲಿ ಧ್ವನಿ ಪರೀಕ್ಷೆ."
        test_filename = "test_audio.mp3"
        
        audio_dir = os.path.join(app.root_path, 'static', 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_path = os.path.join(audio_dir, test_filename)
        
        tts = gTTS(text=test_text, lang='kn', slow=False)
        tts.save(audio_path)
        
        return jsonify({
            "success": True,
            "message": "Test audio generated successfully",
            "path": f"/static/audio/{test_filename}"
        })
    except Exception as e:
        print(f"Test audio error: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    
    os.makedirs(os.path.join(app.root_path, 'static', 'audio'), exist_ok=True)
    app.run(debug=True)