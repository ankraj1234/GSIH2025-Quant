import numpy as np
import pandas as pd
import io
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cholesky

# Black-Scholes call price
def bs_call_price(S, K, T, r, sigma):
    if T == 0 or sigma == 0:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Implied volatility via Newton-Raphson
def implied_volatility(S, K, T, r, market_price):
    sigma = 0.5
    for _ in range(100):
        price = bs_call_price(S, K, T, r, sigma)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if vega < 1e-8:
            break
        diff = price - market_price
        if abs(diff) < 1e-5:
            return sigma
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)
    return sigma

# Build 2D local vol interpolators for each stock

def build_local_vol_surfaces(calib_df):
    local_vol = {}
    for stock in calib_df['Stock'].unique():
        df = calib_df[calib_df['Stock'] == stock]
        strikes = np.sort(df['Strike'].unique())
        mats    = np.sort(df['Maturity'].unique())
        vol_grid = np.zeros((len(strikes), len(mats)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(mats):
                row = df[(df['Strike']==K)&(df['Maturity']==T)]
                if not row.empty:
                    vol_grid[i,j] = implied_volatility(100, K, T, 0.05, float(row['Price']))
        interp = RegularGridInterpolator((strikes, mats), vol_grid,
                                         bounds_error=False, fill_value=None)
        local_vol[stock] = interp
    return local_vol

# Vectorized path simulation
def simulate_paths(S0, T, n_steps, n_paths, L, lv_interp, drift=0.05):
    dt = T / n_steps
    times = np.linspace(0, T, n_steps+1)
    paths = np.zeros((3, n_paths, n_steps+1))
    paths[:,:,0] = S0[:,None]
    
    for k in range(1, n_steps+1):
        Z = np.random.randn(3, n_paths)
        dW = L @ Z * np.sqrt(dt)
        t = times[k]
        for i, stock in enumerate(['DTC','DFC','DEC']):
            S_prev = paths[i,:,k-1]
            pts = np.vstack((S_prev, np.full(n_paths, t))).T
            vols = lv_interp[stock](pts)
            paths[i,:,k] = S_prev * np.exp((drift - 0.5*vols**2)*dt + vols*dW[i])
    return paths

# Price a single basket option
def price_basket_option(option, L, lv_interp, n_paths=5000, n_steps_per_year=50):
    T = option['Maturity']
    n_steps = int(T * n_steps_per_year)
    S0 = np.array([100., 100., 100.])
    paths = simulate_paths(S0, T, n_steps, n_paths, L, lv_interp)
    basket = paths.mean(axis=0)  # shape: (n_paths, n_steps+1)
    barrier, strike = option['KnockOut'], option['Strike']
    knocked = (basket >= barrier).any(axis=1)
    final = basket[:,-1]
    if option['Type'].lower()=='call':
        pay = np.maximum(final - strike, 0)
    else:
        pay = np.maximum(strike - final, 0)
    pay[knocked] = 0
    return np.exp(-0.05*T) * pay.mean()

# Main execution block
if __name__ == '__main__':
    # Load basket options
    data = '''Id,Asset,KnockOut,Maturity,Strike,Type
    1,Basket,150,2y,50,Call
    2,Basket,175,2y,50,Call
    3,Basket,200,2y,50,Call
    4,Basket,150,5y,50,Call
    5,Basket,175,5y,50,Call
    6,Basket,200,5y,50,Call
    7,Basket,150,2y,100,Call
    8,Basket,175,2y,100,Call
    9,Basket,200,2y,100,Call
    10,Basket,150,5y,100,Call
    11,Basket,175,5y,100,Call
    12,Basket,200,5y,100,Call
    13,Basket,150,2y,125,Call
    14,Basket,175,2y,125,Call
    15,Basket,200,2y,125,Call
    16,Basket,150,5y,125,Call
    17,Basket,175,5y,125,Call
    18,Basket,200,5y,125,Call
    19,Basket,150,2y,75,Put
    20,Basket,175,2y,75,Put
    21,Basket,200,2y,75,Put
    22,Basket,150,5y,75,Put
    23,Basket,175,5y,75,Put
    24,Basket,200,5y,75,Put
    25,Basket,150,2y,100,Put
    26,Basket,175,2y,100,Put
    27,Basket,200,2y,100,Put
    28,Basket,150,5y,100,Put
    29,Basket,175,5y,100,Put
    30,Basket,200,5y,100,Put
    31,Basket,150,2y,125,Put
    32,Basket,175,2y,125,Put
    33,Basket,200,2y,125,Put
    34,Basket,150,5y,125,Put
    35,Basket,175,5y,125,Put
    36,Basket,200,5y,125,Put
    '''

    basket_df = pd.read_csv(io.StringIO(data))
    basket_df['Maturity'] = basket_df['Maturity'].str.rstrip('y').astype(float)

    # Calibration data
    calib_data = '''CalibIdx,Stock,Type,Strike,Maturity,Price
    1,DTC,Call,50,1y,52.44
    2,DTC,Call,50,2y,54.77
    3,DTC,Call,50,5y,61.23
    4,DTC,Call,75,1y,28.97
    5,DTC,Call,75,2y,33.04
    6,DTC,Call,75,5y,43.47
    7,DTC,Call,100,1y,10.45
    8,DTC,Call,100,2y,16.13
    9,DTC,Call,100,5y,29.14
    10,DTC,Call,125,1y,2.32
    11,DTC,Call,125,2y,6.54
    12,DTC,Call,125,5y,18.82
    13,DTC,Call,150,1y,0.36
    14,DTC,Call,150,2y,2.34
    15,DTC,Call,150,5y,11.89
    16,DFC,Call,50,1y,52.45
    17,DFC,Call,50,2y,54.9
    18,DFC,Call,50,5y,61.87
    19,DFC,Call,75,1y,29.11
    20,DFC,Call,75,2y,33.34
    21,DFC,Call,75,5y,43.99
    22,DFC,Call,100,1y,10.45
    23,DFC,Call,100,2y,16.13
    24,DFC,Call,100,5y,29.14
    25,DFC,Call,125,1y,2.8
    26,DFC,Call,125,2y,7.39
    27,DFC,Call,125,5y,20.15
    28,DFC,Call,150,1y,1.26
    29,DFC,Call,150,2y,4.94
    30,DFC,Call,150,5y,17.46
    31,DEC,Call,50,1y,52.44
    32,DEC,Call,50,2y,54.8
    33,DEC,Call,50,5y,61.42
    34,DEC,Call,75,1y,29.08
    35,DEC,Call,75,2y,33.28
    36,DEC,Call,75,5y,43.88
    37,DEC,Call,100,1y,10.45
    38,DEC,Call,100,2y,16.13
    39,DEC,Call,100,5y,29.14
    40,DEC,Call,125,1y,1.96
    41,DEC,Call,125,2y,5.87
    42,DEC,Call,125,5y,17.74
    43,DEC,Call,150,1y,0.16
    44,DEC,Call,150,2y,1.49
    45,DEC,Call,150,5y,9.7'''

    calib_df = pd.read_csv(io.StringIO(calib_data))
    calib_df['Maturity'] = calib_df['Maturity'].str.rstrip('y').astype(float)

    # Build local vol surfaces
    lv_interp = build_local_vol_surfaces(calib_df)

    # Correlation Cholesky
    corr = np.array([[1.0,0.75,0.5],[0.75,1.0,0.25],[0.5,0.25,1.0]])
    L = cholesky(corr, lower=True)

    # Price all options
    results = []
    for _, opt in basket_df.iterrows():
        price = price_basket_option(opt, L, lv_interp)
        results.append({'Id': opt['Id'], 'Price': price})

    # Output
    out = pd.DataFrame(results)
    print(out.to_csv(index=False, float_format='%.6f'))