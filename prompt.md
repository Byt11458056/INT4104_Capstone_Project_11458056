You are an expert software engineer and machine learning developer.

Your task is to build a prototype of an AI-powered financial analysis assistant designed to help retail investors analyze market information.

The system should be implemented in Python and structured as a small but complete project.

GOAL
Build a prototype web application that:
1. Retrieves stock market data
2. Retrieves financial news
3. Performs sentiment analysis on news
4. Generates AI summaries using an LLM
5. Provides a simple investment signal
6. Performs a simple historical backtest
7. Displays results in a web dashboard

The system should prioritize simplicity and clarity rather than production-level complexity.

------------------------------------------------

TECH STACK

Language:
Python

Libraries:
pandas
numpy
yfinance
requests
scikit-learn
transformers
streamlit
plotly
vectorbt (for backtesting)
openai (or compatible API)

------------------------------------------------

PROJECT STRUCTURE

Create the following project structure:

finance_ai_app/

app.py
data_pipeline.py
sentiment_analysis.py
price_model.py
signal_engine.py
backtest.py
llm_summary.py
utils.py
requirements.txt

------------------------------------------------

FUNCTIONAL REQUIREMENTS

1. MARKET DATA COLLECTION

Use yfinance to download historical stock data.

Example:

ticker
price history
volume

Functions:

get_stock_data(ticker, start, end)

Return dataframe with:
date
open
high
low
close
volume

------------------------------------------------

2. FEATURE ENGINEERING

Create basic indicators:

moving average
RSI
returns
volatility

Function:

create_features(df)

------------------------------------------------

3. PRICE PREDICTION MODEL

Build a simple machine learning model that predicts next-day direction.

Use:

RandomForestClassifier or XGBoost.

Features:

returns
RSI
moving averages
volatility

Target:

next_day_return > 0

Function:

train_price_model(df)

Return trained model.

------------------------------------------------

4. NEWS COLLECTION

Use a free news API or simple scraping.

Return latest headlines for a ticker.

Function:

get_news(ticker)

Return list of news headlines.

------------------------------------------------

5. SENTIMENT ANALYSIS

Use a pretrained transformer sentiment model such as:

"ProsusAI/finbert"

Compute sentiment score for each headline.

Function:

analyze_sentiment(news_list)

Return average sentiment score.

------------------------------------------------

6. LLM MARKET SUMMARY

Use an LLM API.

Input:

stock price trend
technical indicators
news headlines
sentiment score

Prompt format:

"You are a financial analyst. Analyze the following data and produce a short summary of the market outlook."

Output:

short market insight
risk factors
bullish/bearish interpretation

Function:

generate_llm_summary(data)

------------------------------------------------

7. SIGNAL ENGINE

Combine model prediction and sentiment.

Example rule:

score =
0.6 * price_prediction_probability +
0.4 * normalized_sentiment

Signal:

score > 0.6 → BUY
score < 0.4 → SELL
otherwise HOLD

Function:

generate_signal()

------------------------------------------------

8. BACKTEST

Simulate simple strategy:

BUY when signal = BUY
SELL when signal = SELL
otherwise HOLD

Use vectorbt or simple pandas logic.

Compare performance to buy-and-hold.

Metrics:

cumulative return
max drawdown
sharpe ratio

Function:

run_backtest(df)

------------------------------------------------

9. STREAMLIT DASHBOARD

Build a web interface with:

Input:
ticker symbol

Display:

price chart
technical indicators
news headlines
sentiment score
AI summary
buy/hold/sell signal
backtest performance chart

Use Plotly for charts.

------------------------------------------------

10. USER FLOW

User enters stock ticker.

System:

fetches data
calculates indicators
runs model
collects news
runs sentiment analysis
generates LLM insight
displays signal
runs backtest

------------------------------------------------

CODE REQUIREMENTS

Write clean modular Python code.

Include comments explaining each step.

Make the application runnable with:

streamlit run app.py

------------------------------------------------

OUTPUT

Generate all required Python files with working example code.

Include a requirements.txt file.

Use example ticker:

AAPL

Ensure the prototype runs locally.