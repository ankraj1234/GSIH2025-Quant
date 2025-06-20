import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

def main():
    data = sys.stdin.read().strip().split()
    if len(data) < 2:
        return

    portfolio_id = data[0]
    pnl = np.array(data[1:], dtype=float)

    returns_df = pd.read_csv('stocks_returns.csv', parse_dates=['Date'])
    tickers = [c for c in returns_df.columns if c != 'Date']
    R = returns_df[tickers].values / 100.0  # shape: (T, N)
    T, N = R.shape

    meta = pd.read_csv('stocks_metadata.csv')
    meta = meta.set_index('Stock_Id')

    # Align costs to tickers
    costs = meta.loc[tickers, 'Capital_Cost'].astype(float).values  # shape: (N,)

    # Cost-aware feature scaling: multiply returns by cost_weights (inverse cost)
    cost_weights = 1.0 / costs
    X = R * cost_weights[np.newaxis, :]  # shape: (T, N)

    # Target: negative P&L
    y = -pnl

    # Fit LassoCV
    model = LassoCV(cv=5,
                     fit_intercept=False,
                     max_iter=20000,
                     random_state=42,
                     selection='random')
    model.fit(X, y)

    # Positions: coef * cost_weights
    raw_coef = model.coef_  # shape: (N,)
    positions = raw_coef * cost_weights

    qty = np.rint(positions).astype(int)

    for ticker, q in zip(tickers, qty):
        if q != 0:
            print(f"{ticker} {q}")

if __name__ == '__main__':
    main()