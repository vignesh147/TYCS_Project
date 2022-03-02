import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st


# the time of start and close
start = '2011-12-31'
end = '2021-12-31'

st.title('stock Trend Prediction by vignesh halwai')
user_input = st.text_input('Enter stock Ticker', 'AAPL')
#using datareader to take data, 'AAPL'is the company ticket
df = data.DataReader(user_input,'yahoo', start, end)

#describing data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 moving average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'g')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200 moving average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'g')
plt.plot(ma200, 'y')
plt.plot(df.Close)
st.pyplot(fig)

# spliting Data into training and testing and i think so we are using test model for it

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])# taking 70 % data fot training
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])# and 30 for testing

from sklearn.preprocessing  import MinMaxScaler # all the input will be scaled in between 1 to 0 
scaler = MinMaxScaler(feature_range = (0,1))# converting to 1 to 0

data_training_array = scaler.fit_transform(data_training)

#load my model
model = load_model('vignesh_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final graph

st.subheader('Prediction vs Original')
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)