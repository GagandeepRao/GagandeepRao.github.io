import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
def load_dataset(csv_file):
    return csv_file == pd.read_csv('Fertilizer Prediction.csv')

# Initialize LabelEncoders
soil_type_encoder = LabelEncoder()
crop_type_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Preprocess the data
def preprocess_data(df):
    df['Soil Type'] = soil_type_encoder.fit_transform(df['Soil Type'])
    df['Crop Type'] = crop_type_encoder.fit_transform(df['Crop Type'])
    df['Fertilizer_Name'] = fertilizer_encoder.fit_transform(df['Fertilizer_Name'])

    # Calculate total fertilizer amount (N + P + K)
    df['Total_Fertilizer'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    return df

# Split the data into training and testing sets
def split_data(df):
    X = df.drop(['Fertilizer_Name', 'Total_Fertilizer'], axis=1)
    y_type = df['Fertilizer_Name']
    y_amount = df['Total_Fertilizer']
    
    X_train, X_test, y_type_train, y_type_test, y_amount_train, y_amount_test = train_test_split(
        X, y_type, y_amount, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_type_train, y_type_test, y_amount_train, y_amount_test

# Train Random Forest models
def train_models(X_train, y_type_train, y_amount_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_type_train)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_amount_train)
    
    return rf_classifier, rf_regressor

# Evaluate the models
def evaluate_models(rf_classifier, rf_regressor, X_test, y_type_test, y_amount_test):
    y_type_pred = rf_classifier.predict(X_test)
    y_amount_pred = rf_regressor.predict(X_test)

    type_accuracy = accuracy_score(y_type_test, y_type_pred)
    amount_mse = mean_squared_error(y_amount_test, y_amount_pred)
    amount_r2 = r2_score(y_amount_test, y_amount_pred)

    print(f"Fertilizer Type Prediction Accuracy: {type_accuracy:.2f}")
    print(f"Fertilizer Amount Prediction MSE: {amount_mse:.2f}")
    print(f"Fertilizer Amount Prediction R2 Score: {amount_r2:.2f}")
    print("\nClassification Report for Fertilizer Type:")
    print(classification_report(y_type_test, y_type_pred))

# Plot feature importance
def plot_feature_importance(rf_classifier, rf_regressor, X):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_classifier.feature_importances_ + rf_regressor.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Fertilizer Recommendation')
    plt.tight_layout()
    plt.show()

# Function to recommend fertilizer and amount
def recommend_fertilizer(rf_classifier, rf_regressor, temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous, land_area=1):
    soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
    crop_type_encoded = crop_type_encoder.transform([crop_type])[0]

    input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded,
                            nitrogen, potassium, phosphorous]])

    fertilizer_type = rf_classifier.predict(input_data)
    fertilizer_amount = rf_regressor.predict(input_data)

    recommended_fertilizer = fertilizer_encoder.inverse_transform(fertilizer_type)[0]
    optimized_amount = fertilizer_amount[0] / land_area

    return recommended_fertilizer, optimized_amount

# Main execution function
def main():
    csv_file = input("Please provide the path to your .csv file: ")
    df = load_dataset(csv_file)
    df = preprocess_data(df)
    
    X_train, X_test, y_type_train, y_type_test, y_amount_train, y_amount_test = split_data(df)
    
    rf_classifier, rf_regressor = train_models(X_train, y_type_train, y_amount_train)
    
    evaluate_models(rf_classifier, rf_regressor, X_test, y_type_test, y_amount_test)
    
    plot_feature_importance(rf_classifier, rf_regressor, X_train)
    
    print("\nExample Fertilizer Recommendation:")
    temperature = 30
    humidity = 60
    moisture = 40
    soil_type = "Loamy"
    crop_type = "Wheat"
    nitrogen = 20
    potassium = 15
    phosphorous = 10
    land_area = 1

    recommended_fertilizer, recommended_amount = recommend_fertilizer(
        rf_classifier, rf_regressor, temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous, land_area
    )
    
    print(f"Recommended Fertilizer: {recommended_fertilizer}")
    print(f"Recommended Amount: {recommended_amount:.2f} units")

if __name__ == "__main__":
    main()
