
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/api/v1.0/model")
def run_model(stock_symbol):
    stock_df = get_stock_data('GME')
    run_facebook_prophet_model(stock_df)
