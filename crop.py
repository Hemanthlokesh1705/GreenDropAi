from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the crop recommendation dataset
crop_df = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\imd_maitri.csv')
# Rename 'District' column to 'city' for consistency
crop_df = crop_df.rename(columns={'District': 'city'})
crop_df['city'] = crop_df['city'].str.strip().str.lower()

def predict_next_5_years(city_name):
    rainfall = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\160a665a7b6d47e19e01105ebc5a46ec..csv')
    groundwater = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\groundwater level.csv')
    temperature = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\temp.csv')
    population = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\pop.csv')
    drought = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\drought percentage.csv')
    
    # Melt to long format
    def melt_city(df, value_name):
        return df.melt(id_vars='year', var_name='city', value_name=value_name)

    rain_long = melt_city(rainfall, 'rainfall')
    ground_long = melt_city(groundwater, 'groundwater')
    temp_long = melt_city(temperature, 'temperature')
    pop_long = melt_city(population, 'population')
    drought_long = melt_city(drought, 'drought_percentage')

    # Merge all
    df = drought_long.merge(rain_long, on=['year', 'city'], how='inner')
    df = df.merge(ground_long, on=['year', 'city'], how='inner')
    df = df.merge(temp_long, on=['year', 'city'], how='inner')
    df = df.merge(pop_long, on=['year', 'city'], how='inner')

    # Filter city
    df = df[df['city'].str.lower() == city_name.lower()].copy()
    avg_temp = round(df['temperature'].mean(), 2)
    avg_population = round(df['population'].mean(), 2)
    
    # Feature engineering
    df['rainfall_per_person'] = df['rainfall'] / (df['population'] + 1)
    df['rain_groundwater_ratio'] = df['rainfall'] / (df['groundwater'] + 1)
    df['temperature_rolling_3'] = df['temperature'].rolling(window=3, min_periods=1).mean()

    # Drop missing
    df.dropna(inplace=True)

    # Features and target
    features = ['rainfall', 'rainfall_per_person', 'temperature', 'temperature_rolling_3', 'rain_groundwater_ratio']
    X = df[features]
    y = df['drought_percentage']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    simplified_accuracy = 100 - (rmse / y.mean() * 100)

    # Feature importances
    importances = model.feature_importances_
    top_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:5]

    # Forecast future
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop')
def details():
    return render_template('crop.html')

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    city = data.get('city', '')
    if not city:
        return jsonify({"error": "City name is required"}), 400
    
    try:
        result = predict_next_5_years(city)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)