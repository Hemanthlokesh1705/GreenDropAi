import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def predict_next_5_years(city_name):
    # Load wide-format data
    rainfall = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\rainfall.csv')
    groundwater = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\groundwater.csv')
    temperature = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\temperature.csv')
    population = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\population.csv')
    drought = pd.read_csv(r'C:\Users\LATITUDE\Desktop\python\Water\Yearly_Drought_2010_2024.csv')

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
        "model_performance": {
            "RMSE": round(rmse, 2),
            "R² Score": round(r2, 2),
            "Simplified Accuracy": f"{round(simplified_accuracy, 1)}%"
        },
        "top_5_factors": [{k: round(v, 3)} for k, v in top_features],
        "future_predictions": future_data
    }


# Example run
results = predict_next_5_years("raichur")

# Print
from pprint import pprint
pprint(results)
