# import relevant libraries 
import streamlit as st 
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# enter start & end date
START = "2011-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# title of webapp 
st.title("Stock Prediction Web App")

stocks = ("FB","TWTR","AAPL", "MSFT", "AMZN")
selected_stock =st.selectbox("select dataset for prediction",stocks)

#number of years and periods for prediction
n_years = st.slider("Years of Prediction:" , 1 , 5)
period = n_years * 365 

# caching the data
@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

# details on the load data state .
data_load_state = st.text("load data ....")
data = load_data(selected_stock)
data_load_state.text("Loading data ....done!")

# subheader details 
st.subheader('Raw data')
st.write(data.tail())

# plot raw data 
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    plot_raw_data()

# Forecasting 
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast= m.predict(future)

# forecast data details 
st.subheader('Forecast data')
st.write(forecast.tail())

# ploting  fig 1 
st.write('forecast data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

# plot fig 2 
st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
