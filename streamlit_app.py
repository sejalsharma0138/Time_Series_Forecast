import pandas as pd
import streamlit as st
import io
from datetime import datetime
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import statsmodels.api as sm
from prophet import Prophet
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def read_data(path):
    df = pd.read_excel(path)
    return df

def extracting_cleaning(df, item):
    # Extracting prtno and its Quantity from entire data
    item1 = item                                                          
    df1 = df[df['Product Name'] == item1]
    df1.drop(columns=['Product ID', 'Product Name', 'Category', 'Brand', 'Unit Price',
        'Total Price', 'Seasonality', 'Promotion',
       'COVID Impact', 'Festive Period'], inplace=True)
    
    # Ensure 'Purchase Date' is in datetime format
    df1['Purchase Date'] = pd.to_datetime(df1['Purchase Date'])
    
    df1.sort_values(by='Purchase Date', inplace=True)

    # Aggregating data monthly
    df1 = df1.groupby(df1['Purchase Date'].dt.to_period("M")).agg({'Quantity': 'sum'}).reset_index()
    df1['Purchase Date'] = df1['Purchase Date'].dt.to_timestamp()
    df1.set_index('Purchase Date', inplace=True)
    
    # Handling missing values 
    complete_date_range = pd.date_range(start='2018-01-01', end='2024-06-01', freq='MS')
    df1_reindexed = df1.reindex(complete_date_range)
    df1_reindexed.fillna(0, inplace=True)
    df1_reindexed.index.name = 'Purchase Date'
    df1 = df1_reindexed

    # Aggregating data quarterly
    df1.reset_index(inplace=True)
    df1_q = df1.groupby(df1['Purchase Date'].dt.to_period("Q")).agg({'Quantity': 'sum'}).reset_index()
    df1_q['Purchase Date'] = df1_q['Purchase Date'].dt.to_timestamp()
    df1_q.set_index('Purchase Date', inplace=True)
    df1.set_index('Purchase Date', inplace=True)
    total = df1['Quantity'].sum()

    return df1, df1_q, total


def ml_model_regression(df, algorithm):
    model=LinearRegression()
    df = df.reset_index()
    X = np.array((df['Purchase Date'] - df['Purchase Date'].min()).dt.days).reshape(-1, 1)
    y = df['Quantity'].values
    if algorithm == 'LinearRegression':
        model = LinearRegression()
        
    elif algorithm == 'RandomForest':
        model = RandomForestRegressor()
        
    elif algorithm == 'SVR':
        model = SVR()
        
    elif algorithm == 'DecisionTree':
        model = DecisionTreeRegressor()
        
    model.fit(X, y)
    return model

def forecast_regression(model, df, periods, start):
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
    future_days = (future_dates - df.index.min()).days.values  # Convert to NumPy array
    future_days = future_days.reshape(-1, 1)  # Reshape properly
    forecast_values = model.predict(future_days)
    forecast_df = pd.DataFrame({'Purchase Date': future_dates, 'Quantity': forecast_values})
    forecast_df.set_index('Purchase Date', inplace=True)
    return forecast_df[start:]


def lstm_model(df, n_steps=7):  # Using 7 steps as an example
    df = df.reset_index()
    df['Days'] = (df['Purchase Date'] - df['Purchase Date'].min()).dt.days
    X = df['Days'].values.reshape(-1, 1)
    y = df['Quantity'].values
    
    if len(X) <= n_steps:
        raise ValueError("Not enough data to create sequences for LSTM forecasting.")

    X = np.array([X[i:i+n_steps] for i in range(len(X) - n_steps)])
    y = y[n_steps:]
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model.fit(X, y, epochs=200, verbose=0)
    
    return model


def forecast_lstm(model, df, n_steps=7, periods=12):
    df = df.reset_index()
    df['Days'] = (df['Purchase Date'] - df['Purchase Date'].min()).dt.days
    last_days = df['Days'].values[-n_steps:]
    
    # Prepare the input for forecasting
    if len(last_days) < n_steps:
        raise ValueError("Not enough data to create sequences for LSTM forecasting.")
    
    X = np.array([last_days[i:i+n_steps] for i in range(len(last_days) - n_steps + 1)])
    
    # Reshape for LSTM input
    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    
    forecast_values = model.predict(X[-1].reshape(1, n_steps, 1))  # Predict using the latest sequence
    
    # Generate future dates
    future_dates = pd.date_range(start=df['Purchase Date'].max(), periods=periods + 1, freq='MS')[1:]
    
    # Create DataFrame for forecast
    forecast_df = pd.DataFrame({'Purchase Date': future_dates, 'Quantity': np.concatenate([forecast_values.flatten(), np.zeros(periods-1)])})
    forecast_df.set_index('Purchase Date', inplace=True)
    
    return forecast_df


def ets_model(df):
    df = df.reset_index()
    df = df.rename(columns={'Purchase Date': 'ds', 'Quantity': 'y'})
    model = sm.tsa.ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    
    return model_fit

def forecast_ets(model_fit, df, periods):
    df = df.reset_index()
    last_date = df['Purchase Date'].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
    forecast = model_fit.forecast(periods)
    
    forecast_df = pd.DataFrame({'Purchase Date': future_dates, 'Quantity': forecast})
    forecast_df.set_index('Purchase Date', inplace=True)
    
    return forecast_df

def arima_model(df):
    df = df.reset_index()
    df = df.rename(columns={'Purchase Date': 'ds', 'Quantity': 'y'})
    model = sm.tsa.ARIMA(df['y'], order=(5,1,0))
    model_fit = model.fit()
    
    return model_fit

def forecast_arima(model_fit, df, periods):
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
    forecast = model_fit.forecast(steps=periods)
    
    forecast_df = pd.DataFrame({'Purchase Date': future_dates, 'Quantity': forecast})
    forecast_df.set_index('Purchase Date', inplace=True)
    
    return forecast_df

def prophet_model(df):
    df = df.reset_index()
    df = df.rename(columns={'Purchase Date': 'ds', 'Quantity': 'y'})
    model = Prophet()
    model.fit(df)
    
    return model

def forecast_prophet(model, df, periods):
    df = df.reset_index()
    df = df.rename(columns={'Purchase Date': 'ds', 'Quantity': 'y'})
    last_date = df['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
    future_df = pd.DataFrame({'ds': future_dates})
    
    forecast = model.predict(future_df)
    
    forecast_df = pd.DataFrame({'Purchase Date': future_dates, 'Quantity': forecast['yhat'].values})
    forecast_df.set_index('Purchase Date', inplace=True)
    
    return forecast_df

def evaluate_models(train, test, algorithms):
    #Stores intermediate results 
    results = []
    

    # test is aggregated quarterly
    test_q = test.resample('Q').sum()

    for algorithm in algorithms:
        if algorithm == 'ETS':
            model_fit = ets_model(train)
            forecast_df = forecast_ets(model_fit, train, 12)
           
        elif algorithm == 'ARIMA':
            model_fit = arima_model(train)
            forecast_df = forecast_arima(model_fit, train, 12)
           
        elif algorithm == 'Prophet':
            model_fit = prophet_model(train)
            forecast_df = forecast_prophet(model_fit, train, 12)
        elif algorithm == 'LSTM':
            model = lstm_model(train)
            forecast_df = forecast_lstm(model, train, n_steps=7, periods=12)
        else:
            model = ml_model_regression(train, algorithm)
            forecast_df = forecast_regression(model, train, 12, '2024-01-01')
          
        # Modifying the forecast_df
        forecast_df.index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=len(forecast_df), freq='MS')
        forecast_df_q = forecast_df.resample('Q').sum()
        
        common_index = test_q.index.intersection(forecast_df_q.index)
        test_q = test_q.loc[common_index]
        forecast_df_q = forecast_df_q.loc[common_index]
        
       #MAE Calculation
        mae = mean_absolute_error(test_q[['Quantity']], forecast_df_q[['Quantity']])
            
        results.append({
                'Algorithm': algorithm,
                'MAE': mae,
                'Forecast': forecast_df
            })
            
    
    return results,test_q

def hybrid_modelling(results,test_q):
    best_mae = float('inf')
    best_forecast = None
    best_algorithm = None

    # Extract forecasts and errors from results
    forecasts = [result['Forecast'] for result in results]
    errors = [result['MAE'] for result in results]

    # Create hybrid model if there are more than one forecast
    if len(forecasts) > 1:
        # Calculate inverse errors and weights
        inverse_errors = [1 / error for error in errors]
        total_inverse_error = sum(inverse_errors)
        weights = [ie / total_inverse_error for ie in inverse_errors]

        # Combine forecasts using the calculated weights
        combined_forecast = sum(weight * forecast for weight, forecast in zip(weights, forecasts))
        combined_forecast_df = pd.DataFrame(combined_forecast, index=forecasts[0].index, columns=['Quantity'])

        # Calculate MAE for the combined forecast
        combined_forecast_df_q = combined_forecast_df.resample('Q').sum()
     

        common_index = test_q.index.intersection(combined_forecast_df_q.index)
        test_q = test_q.loc[common_index]
        combined_forecast_df_q = combined_forecast_df_q.loc[common_index]

        combined_mae = mean_absolute_error(test_q[['Quantity']], combined_forecast_df_q[['Quantity']])
        # Append hybrid results to results
        results.append({
            'Algorithm': 'Hybrid',
            'MAE': combined_mae,
            'Forecast': combined_forecast_df
        })

        # Update best model if hybrid model has lower MAE
        if combined_mae < best_mae:
            best_mae = combined_mae
            best_forecast = combined_forecast_df
            best_algorithm = 'Hybrid'

    # Find the best model based on MAE
    for result in results:
        if result['MAE'] < best_mae:
            best_mae = result['MAE']
            best_forecast = result['Forecast']
            best_algorithm = result['Algorithm']
   
    
    return best_mae, best_forecast, best_algorithm

# def main():
#     st.title("Product Forecasting Dashboard")
    
#     # File upload
#     uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
#     if uploaded_file is not None:
#         df = pd.read_excel(uploaded_file)
        
#         # Input product name
#         product = st.text_input("Enter the product name:")
        
#         if product:
#             # Data extraction and cleaning
#             df1, df1_q, total = extracting_cleaning(df, product)
            
#             # Split data into training and testing
#             train = df1.iloc[:72]
#             test = df1.iloc[72:]
            
#             # Algorithms to be evaluated
#             algorithms_ml = ['LSTM', 'ETS', 'ARIMA', 'Prophet', 'LinearRegression', 'RandomForest', 'SVR', 'DecisionTree']
            
#             # Evaluate models
#             results, test_q = evaluate_models(train, test, algorithms_ml)
            
#             # Perform hybrid modelling
#             best_mae, best_forecast, best_algorithm = hybrid_modelling(results, test_q)
            
#             # Plot existing trend
#             st.subheader("Existing Trend")
#             fig1, ax1 = plt.subplots(figsize=(14, 7))
#             ax1.plot(df1.index, df1['Quantity'], label='Monthly Quantity', marker='o')
#             ax1.plot(df1_q.index, df1_q['Quantity'], label='Quarterly Quantity', marker='s')
#             ax1.set_title(f'Quantity Trend for {product}')
#             ax1.set_xlabel('Purchase Date')
#             ax1.set_ylabel('Quantity')
#             ax1.legend()
#             ax1.grid(True)
#             st.pyplot(fig1)
            
#             # Plot forecast
#             st.subheader("Forecast Plot")
#             fig2, ax2 = plt.subplots(figsize=(14, 7))
#             ax2.plot(train.index, train['Quantity'], label='Training Data')
#             ax2.plot(test.index, test['Quantity'], label='Test Data')
#             ax2.plot(best_forecast.index, best_forecast['Quantity'], label='Forecast', color='green')
#             ax2.legend()
#             ax2.set_title(f'Forecast for {product}')
#             ax2.set_xlabel('Date')
#             ax2.set_ylabel('Quantity')
#             ax2.set_xticks(np.arange(0, len(df1.index), step=6))
#             ax2.set_xticklabels(df1.index.strftime('%Y-%m-%d')[::6], rotation=45)
#             ax2.grid(True)
#             st.pyplot(fig2)

# if __name__ == "__main__":
#     main()
def main():
    st.title('Sales Forecasting App')

    # File uploader for the input file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        product = st.selectbox("Select a product", df['Product Name'].unique())
        
        if st.button("Generate Forecast"):
            df1, df1_q, total = extracting_cleaning(df, product)
            
            # Split data into training and testing
            train = df1.iloc[:72]
            test = df1.iloc[72:]
            
            # Algorithms to be evaluated
            algorithms_ml = ['LSTM', 'ETS', 'ARIMA', 'Prophet', 'LinearRegression', 'RandomForest', 'SVR', 'DecisionTree']
            
            # Evaluate models
            results, test_q = evaluate_models(train, test, algorithms_ml)
            
            # Perform hybrid modelling
            best_mae, best_forecast, best_algorithm = hybrid_modelling(results, test_q)
            
            # Plotting the existing trend
            fig_existing, ax_existing = plt.subplots(figsize=(14, 7))
            ax_existing.plot(df1.index, df1['Quantity'], label='Monthly Quantity', marker='o')
            ax_existing.plot(df1_q.index, df1_q['Quantity'], label='Quarterly Quantity', marker='s')
            ax_existing.set_title(f'Quantity Trend for {product}')
            ax_existing.set_xlabel('Purchase Date')
            ax_existing.set_ylabel('Quantity')
            ax_existing.legend()
            ax_existing.grid(True)
            
            # Convert to Streamlit compatible format
            buf_existing = io.BytesIO()
            plt.savefig(buf_existing, format='png')
            buf_existing.seek(0)
            st.image(buf_existing, caption='Existing Trend Plot')
            
            # Plotting the forecast
            fig_forecast, ax_forecast = plt.subplots(figsize=(14, 7))
            ax_forecast.plot(train.index, train['Quantity'], label='Training Data')
            ax_forecast.plot(test.index, test['Quantity'], label='Test Data')
            ax_forecast.plot(best_forecast.index, best_forecast['Quantity'], label='Forecast', color='green')
            ax_forecast.set_title(f'Forecast for {product}')
            ax_forecast.set_xlabel('Purchase Date')
            ax_forecast.set_ylabel('Quantity')
            ax_forecast.legend()
            ax_forecast.grid(True)
            
            # Convert to Streamlit compatible format
            buf_forecast = io.BytesIO()
            plt.savefig(buf_forecast, format='png')
            buf_forecast.seek(0)
            st.image(buf_forecast, caption='Forecast Plot')

            # Display the final results table
            final_df = pd.DataFrame({
                'Product': [product],
                'MAE_Quarterly': [best_mae],
                'Best_Model': [best_algorithm],
                **{date: [best_forecast.loc[date, 'Quantity'] if date in best_forecast.index else None] for date in ['2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01']}
            })
            st.write("Final Results", final_df)

if __name__ == "__main__":
    main()