Demand Forecasting for Retail Stores

Overview
This project aims to predict the weekly sales of retail stores, specifically Walmart stores, using machine learning and deep learning techniques. The goal is to analyze historical sales data and forecast future sales trends, which can help in inventory planning and decision-making processes.

Dataset
The dataset used in this project is Walmart_Store_sales.csv. It includes the following key attributes:
* Store
* Date
* Weekly_ Sales
* Holiday_Flag
* Temperature
* …

Dataset Source: https://www.kaggle.com/datasets/bharatkumar0925/walmart-store-sales

Dependencies
To run this project, ensure you have the following Python libraries installed:
pip install pandas matplotlib scikit-learn xgboost optuna tensorflow joblib

Data Preprocessing
* Date Feature Engineering: We converted the Date column into multiple useful features like Year, Month, Week, and Day.
* Missing Values: Checked and handled missing values.
* Scaling: Sales data (Weekly_Sales) was scaled using MinMaxScaler to improve the performance of machine learning models.

Exploratory Data Analysis (EDA)
Sales Trend Analysis
* We explored the sales trend over time to understand fluctuations and seasonality.
Feature Engineering
* Additional features like Year, Month, Week, Day, DayofWeek, and isWeekend were derived from the Date column to provide more predictive power to the models.

Visualizations
The project visualizes various insights:
* Sales trends over time.
* Sales distribution by departments and stores.
* Weekend vs weekday sales patterns.

Machine Learning & Deep Learning Models

XGBoost (Gradient Boosting)
* Model Description: XGBoost is a powerful gradient boosting algorithm used for regression tasks. In this project, it was used to predict weekly sales.
* Evaluation Metrics: The model was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* Hyperparameter Tuning: We used Optuna for hyperparameter optimization to improve model performance.
LSTM (Long Short-Term Memory) Model
* Model Description: LSTM, a type of Recurrent Neural Network (RNN), was used to predict sales based on historical time series data. It captures sequential patterns in the data, making it well-suited for this problem.
* Evaluation: The LSTM model was evaluated by comparing predicted and actual sales trends.

Model Saving and Deployment
The trained models were saved for future use:
* XGBoost model saved as sales_forecasting_model.pkl.
* LSTM model saved as sales_forecasting_model.keras.

Additional Notes
* Ensure all dependencies from requirements.txt are installed.
* If running in Jupyter Notebook, run cells sequentially to avoid errors.
* The project outputs visualizations and model metrics.

Future Improvements
* Model Performance: Test advanced models like LightGBM or Prophet for better accuracy.
* External Factors: Incorporate factors like promotions, weather, or holidays for more accurate forecasts.
* Interactive Dashboard: Build an interactive dashboard using Streamlit or Dash.
* Model Deployment: Deploy the model as a web app using Flask or FastAPI.

Author
Harris Hussain harrishussain2408@gmail.com

License
This project is open-source under the MIT License.


