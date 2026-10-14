# app.py
import streamlit as st
import pandas as pd
import pandas_ta as ta
import requests
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------
# 1. Page Config & Auto-Refresh (Every 15 Sec)
# ---------------------------------------------
st.set_page_config(page_title="Smart Money Pro Scanner", layout="wide")
st_autorefresh(interval=15000, limit=10000, key="data_refresh")

st.title("⚡ Top 30 + Gold (XAUUSDT) Sniper Scanner")
st.caption("Includes: VWAP + FVG + Consolidation Filter + Liquidity Sweeps + Hyperliquid Whale Orders")

# ---------------------------------------------
# 2. Get Top 30 Binance Coins + XAUUSDT
# ---------------------------------------------
@st.cache_data(ttl=300) 
def get_top_symbols(limit=30):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        resp = requests.get(url).json()
        usdt_pairs = [d for d in resp if d['symbol'].endswith('USDT')]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
        symbols = [x['symbol'] for x in sorted_pairs[:limit]]
        if "XAUUSDT" not in symbols:
            symbols.append("XAUUSDT")
        return symbols
    except:
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT"] 

# ---------------------------------------------
# 3. Fetch Binance OHLCV & Calculate Signals
# ---------------------------------------------
def analyze_price_action(symbol):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=50"
    try:
        resp = requests.get(url, timeout=3).json()
        df = pd.DataFrame(resp, columns=['time', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # FVG Detection
        df['bullish_fvg'] = df['low'].shift(0) > df['high'].shift(2)
        df['bearish_fvg'] = df['high'].shift(0) < df['low'].shift(2)
        
        # Consolidation Filter (ADX)
        df.ta.adx(length=14, append=True)
        
        # Liquidity Sweep
        df['rolling_low'] = df['low'].rolling(window=20).min().shift(1)
        df['rolling_high'] = df['high'].rolling(window=20).max().shift(1)
        
        latest = df.iloc[-1]
        
        # Conditions
        is_trending = latest.get('ADX_14', 0) > 25
        bull_sweep = (latest['low'] < latest['rolling_low']) and (latest['close'] > latest['rolling_low'])
        bear_sweep = (latest['high'] > latest['rolling_high']) and (latest['close'] < latest['rolling_high'])
        
        bias = "BULLISH" if latest['close'] > latest['vwap'] else "BEARISH"
        
        # Determine Final Signal
        signal = "No Signal"
        if not is_trending:
            signal = "AVOID ⚠️ (Choppy)"
        elif bias == "BULLISH" and df['bullish_fvg'].iloc[-1]:
            signal = "STRONG BUY 🚀 (Sweep+FVG)" if bull_sweep else "BUY 🟢 (FVG)"
        elif bias == "BEARISH" and df['bearish_fvg'].iloc[-1]:
            signal = "STRONG SELL 💥 (Sweep+FVG)" if bear_sweep else "SELL 🔴 (FVG)"
            
        return latest['close'], bias, signal
    except:
        return 0, "Error", "Error"

# ---------------------------------------------
# 4. Fetch Hyperliquid Whale Orders ($100k+)
# ---------------------------------------------
def get_whale_orders(symbol, min_value_usd=100000):
    hl_coin = symbol.replace("USDT", "") 
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "l2Book", "coin": hl_coin}
    
    try:
        resp = requests.post(url, json=payload, timeout=2).json()
        bids = resp['levels'][0]
        asks = resp['levels'][1]
        
        whale_bids = sum(1 for b in bids if (float(b['sz']) * float(b['px'])) >= min_value_usd)
        whale_asks = sum(1 for a in asks if (float(a['sz']) * float(a['px'])) >= min_value_usd)
        
        res = []
        if whale_bids > 0: res.append(f"🟢 {whale_bids} Buy Walls")
        if whale_asks > 0: res.append(f"🔴 {whale_asks} Sell Walls")
        
        return " | ".join(res) if res else "No Whale Activity"
    except:
        return "N/A"

# ---------------------------------------------
# 5. Main Execution Engine
# ---------------------------------------------
symbols = get_top_symbols(30)

with st.spinner("Scanning Real-Time Data (Binance & Hyperliquid)..."):
    results = []
    for sym in symbols:
        price, bias, signal = analyze_price_action(sym)
        whales = get_whale_orders(sym)
        
        results.append({
            "Asset": sym,
            "Price": f"${price:,.4f}",
            "VWAP Trend": bias,
            "Action Signal": signal,
            "Hyperliquid Whales ($100k+)": whales
        })

# ---------------------------------------------
# 6. Display Dashboard UI
# ---------------------------------------------
df_results = pd.DataFrame(results)

def highlight_signals(val):
    if "STRONG BUY" in str(val) or "BUY" in str(val): return 'background-color: #004d00; color: white'
    elif "STRONG SELL" in str(val) or "SELL" in str(val): return 'background-color: #4d0000; color: white'
    elif "AVOID" in str(val): return 'color: #ffcc00'
    return ''

st.dataframe(
    df_results.style.applymap(highlight_signals, subset=['Action Signal']),
    use_container_width=True, 
    height=800
)