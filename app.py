
from flask import Flask, jsonify
from run_model import *

app = Flask(__name__)

@app.route("/")
def home():
    return "I m  here"

@app.route("/api/v1.0/model/<symbol>")
def run_model(symbol):
    print(symbol)
    stock_df = get_stock_data(symbol)
    return run_facebook_prophet_model(stock_df)
     


if __name__ == '__main__':
    app.run(port=5002, debug=True)