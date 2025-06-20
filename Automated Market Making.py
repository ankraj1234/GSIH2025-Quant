import numpy as np
import pandas as pd

class AutomatedMarketMaking:
    def __init__(self, tick_size=0.1, lot_size=2):
        self.tick_size  = tick_size
        self.lot_size   = lot_size
        self.max_inventory = 20
        self.reset_simulator()
        
    def reset_simulator(self):
        self.inventory  = 0
        self.active_bid = None
        self.active_ask = None
        self.valid_from = None
        
        # Initialize volatility smoother
        self.vol_ema     = None
        self.alpha_vol   = 0.2
        
    def update_quote(self, timestamp, bid_price, ask_price):
        # Post or update your quote at timestamp It takes effect at t+1
        self.active_bid = bid_price
        self.active_ask = ask_price
        self.valid_from = timestamp + 1
        
    def process_trades(self, timestamp, trades_at_t):
        # Process all public trades at timestamp Returns updated inventory
        if self.valid_from is None or timestamp < self.valid_from:
            return self.inventory
        filled = False
        # sellside fill against your bid
        sells = trades_at_t[trades_at_t.side == 'sell']
        if self.active_bid is not None and not sells.empty:
            if self.active_bid >= sells.price.max():
                self.inventory += self.lot_size
                self.active_bid = None
                filled = True
        # buyside fill against your ask
        buys = trades_at_t[trades_at_t.side == 'buy']
        if self.active_ask is not None and not buys.empty:
            if self.active_ask <= buys.price.min():
                self.inventory -= self.lot_size
                self.active_ask = None
                filled = True
        if filled:
            # deactivate until next update
            self.valid_from = float('inf')
        return self.inventory
    
    def calculate_volatility(self, ob_recent, default_vol=0.01):
        """Smoothed volatility: EMA of mid-price returns plus range-based floor"""
        mids = (ob_recent.bid_1_price + ob_recent.ask_1_price) / 2
        # Use log returns for volatility
        if len(mids) > 1:
            log_returns = np.diff(np.log(mids))
            inst_vol = np.std(log_returns) * np.sqrt(len(log_returns))
        else:
            inst_vol = default_vol
        # Initialize or update EMA
        if self.vol_ema is None:
            self.vol_ema = inst_vol
        else:
            self.vol_ema = self.alpha_vol * inst_vol + (1 - self.alpha_vol) * self.vol_ema
        # Range-based check
        price_range = mids.max() - mids.min()
        range_vol = price_range / mids.mean() if mids.mean() > 0 else default_vol
        vol = max(self.vol_ema, range_vol, default_vol)
        return vol
    
    def strategy(self, ob_df, tr_df, inventory, t):
            
        ob_recent = ob_df[ob_df.timestamp <= t].tail(50)
        if ob_recent.empty:
            return None, None
        row = ob_recent.iloc[-1]
        best_bid = row.bid_1_price
        best_ask = row.ask_1_price
        mid_price = (best_bid + best_ask) / 2
        market_spread = best_ask - best_bid
        
        # Calculate price volatility
        vol = self.calculate_volatility(ob_recent)
        
        # Time parameter
        time_factor = 1.0
        
        # Calculate risk parameter that increases with inventory
        risk_parameter = 0.1 + 0.005 * abs(inventory)
        
        # Calculate liquidity parameter 
        liquidity_parameter = 1.5
        
        # Calculate quote center with inventory adjustment
        quote_center = mid_price - inventory * risk_parameter * vol**2 * time_factor
        
        # Calculate half-spread width
        spread_width = risk_parameter * vol**2 * time_factor + (2 / risk_parameter) * np.log(1 + risk_parameter / liquidity_parameter)
        
        # Calculate inventory impact on quote placement
        inventory_impact = 0.5 * (inventory / self.max_inventory)
        
        # Determine preliminary quotes
        bid_price = quote_center - spread_width + inventory_impact * spread_width
        ask_price = quote_center + spread_width + inventory_impact * spread_width
        
        # Limit maximum spread
        max_allowed_spread = market_spread * 3
        if (ask_price - bid_price) > max_allowed_spread:
            center = (ask_price + bid_price) / 2
            bid_price = center - max_allowed_spread / 2
            ask_price = center + max_allowed_spread / 2
        
        # Handle extreme inventory positions
        if abs(inventory) > 0.9 * self.max_inventory:
            if inventory > 0:  # Too much inventory, need to sell
                ask_price = min(ask_price, best_bid + 2 * self.tick_size)
            else:  # Too little inventory, need to buy
                bid_price = max(bid_price, best_ask - 2 * self.tick_size)
                
        # Round to tick size
        bid_price = np.floor(bid_price / self.tick_size) * self.tick_size
        ask_price = np.ceil(ask_price / self.tick_size) * self.tick_size
        
        # Ensure quotes don't cross
        if bid_price >= ask_price:
            ask_price = bid_price + self.tick_size
            
        return round(bid_price, 1), round(ask_price, 1)
    
    def run(self, ob_df, tr_df):
        self.reset_simulator()
        quotes = []
        all_ts = sorted(ob_df.timestamp.unique())
        for t in all_ts:
            trades_t = tr_df[tr_df.timestamp == t]
            inv      = self.process_trades(t, trades_t)
            bid, ask = self.strategy(ob_df, tr_df, inv, t)
            self.update_quote(t, bid, ask)
            quotes.append({
                'timestamp': t,
                'bid_price': bid,
                'ask_price': ask
            })
        return pd.DataFrame(quotes)
    
    
if __name__ == "__main__":
    ob_obj = pd.read_csv(input().strip())
    tr_obj = pd.read_csv(input().strip())
    
    #pick top 3k timestamps
    ob_obj = ob_obj.head(3000); 
    tr_obj = tr_obj.head(3000);
    amm = AutomatedMarketMaking(tick_size=0.1, lot_size=2)
    df_submission = amm.run(ob_obj, tr_obj)
    df_submission.to_csv('submission.csv', index=False)