from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

app = Flask(__name__)

# Load the data from Excel sheets
excel_file = 'capstone.xlsx'
xls = pd.ExcelFile(excel_file)

# Get the sheet names from the Excel file
sheet_names = xls.sheet_names

# Initialize an empty list to store the dataframes
dataframes = []

# Iterate over each sheet name and read the corresponding sheet into a dataframe
for sheet_name in sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    dataframes.append(df)

# Concatenate all the dataframes into a single dataframe
df = pd.concat(dataframes)

# Separate features and target variables
X = df[['Section Name', 'Year']]
y = df[['IRI (m/km)', 'Cracking Area (%)', 'Potholes (no.km)', 'Rut Depth (mm)']]

# Perform one-hot encoding on the features
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a linear regression model
lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)


@app.route('/')
def index():
    section_names = df['Section Name'].unique()
    data = {}
    return render_template('index.html', section_names=section_names, predictions=[], data=data)

@app.route('/', methods=['POST', 'GET'])
def predict():

    if request.method == 'POST':
        section_name = request.form.get('section_name')
        start_year = int(request.form.get('start_year'))
        end_year = int(request.form.get('end_year'))

        # Create an empty DataFrame to store all predictions
        combined_predictions = pd.DataFrame()

        # Iterate over each year
        for year in range(start_year, end_year + 1):
            # Create a DataFrame for the specific year
            upcoming_data = pd.DataFrame({'Section Name': [section_name], 'Year': [year]}).set_index('Section Name')

            # Perform one-hot encoding on the upcoming data using the same encoder as the training data
            upcoming_data_encoded = pd.get_dummies(upcoming_data, drop_first=False)
            missing_cols = set(X_train.columns) - set(upcoming_data_encoded.columns)
            for col in missing_cols:
                upcoming_data_encoded[col] = 0
            upcoming_data_encoded = upcoming_data_encoded[X_train.columns]

            # Make predictions for the upcoming year
            predictions = lr.predict(upcoming_data_encoded)

            # Create a DataFrame for the predictions
            predictions_df = pd.DataFrame(predictions, columns=y.columns)

            # Add 'Section Name' and 'Year' columns to the predictions DataFrame
            predictions_df['Section Name'] = section_name
            predictions_df['Year'] = year

            # Append the predictions to the combined_predictions DataFrame
            combined_predictions = pd.concat([combined_predictions, predictions_df], ignore_index=True)
            

        # Define the alternatives
        alternatives = {
            'Resealing + Thin Overlay': lambda row: row[0] > 8 and 5.8 >= row[1] >= 4,
            'Thick Overlay': lambda row: 8 >= row[1] >= 5.8,
            'Reconstruction': lambda row: row[1] >= 8
        }

        # Add alternative column based on conditions
        combined_predictions['Alternative'] = combined_predictions.apply(
            lambda row: next(
                (alt for alt, condition in alternatives.items() if condition((row['Cracking Area (%)'], row['IRI (m/km)']))),
                'Not applicable'
            ) if 'Cracking Area (%)' in row and 'IRI (m/km)' in row else 'Undefined',
            axis=1
        )

        section_names = df['Section Name'].unique()
        # Save the predictions to an Excel file
        predictions_file = 'predictions.xlsx'
        combined_predictions.to_excel(predictions_file, index=False)

        data = {
            'y_iri': combined_predictions["IRI (m/km)"].tolist(),
            'y_crackingarea': combined_predictions["Cracking Area (%)"].tolist(),
            'alternative_labels': combined_predictions["Alternative"].tolist(),  # Add alternative labels
            "y_potholes": combined_predictions["Potholes (no.km)"].tolist(),
            'y_rutdepth': combined_predictions["Rut Depth (mm)"].tolist(),
            "x": combined_predictions["Year"].tolist(),
            "section_name": section_name,
        }

        return render_template('index.html', predictions=combined_predictions.to_dict('records'), section_names=section_names, data=data)
    
    return render_template('index.html', data = {})


@app.route('/download')
def download():
    predictions_file = 'predictions.xlsx'
    return send_file(predictions_file, as_attachment=True)


if __name__ == '__main__':
    app.run()
