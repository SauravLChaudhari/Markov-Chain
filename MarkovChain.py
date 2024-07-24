import numpy as np
import pandas as pd
import yfinance as yf

# Download historical stock price data for Nifty 50
tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 
           'ICICIBANK.NS', 'KOTAKBANK.NS', 'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS']

start_date = '2022-01-01'
end_date = '2023-01-01'

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Discretize returns into states
states = pd.qcut(returns.stack(), 10, labels=False).unstack()

# Create transition matrix
transition_matrix = pd.DataFrame(np.zeros((10, 10)), index=range(10), columns=range(10))

for stock in states.columns:
    for i in range(len(states) - 1):
        current_state = states.iloc[i][stock]
        next_state = states.iloc[i + 1][stock]
        transition_matrix.loc[current_state, next_state] += 1

# Normalize the transition matrix to get probabilities
transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

# Function to predict next state
def predict_next_state(current_state, transition_matrix):
    return np.random.choice(transition_matrix.columns, p=transition_matrix.loc[current_state])

# Predict future stock price movements
def predict_stock_price(initial_state, transition_matrix, steps=10):
    state = initial_state
    predictions = [state]
    for _ in range(steps):
        state = predict_next_state(state, transition_matrix)
        predictions.append(state)
    return predictions

# Example prediction for RELIANCE.NS
initial_state = states['RELIANCE.NS'].iloc[-1]
predicted_states = predict_stock_price(initial_state, transition_matrix, steps=10)

print(predicted_states)
