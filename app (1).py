import streamlit as st
import ccxt
import pandas as pd
import numpy as np

st.set_page_config(page_title="Crypto Multi-TF Confluence Scanner", layout="wide")

# ─────────────────────────────────────────────
# 1. EXCHANGE DATA FETCHER (BINANCE -> OKX -> KUCOIN)
# ─────────────────────────────────────────────
@st.cache_resource
def get_exchanges():
    return {
        'binance': ccxt.binance({'enableRateLimit': True}),
        'okx': ccxt.okx({'enableRateLimit': True}),
        'kucoin': ccxt.kucoin({'enableRateLimit': True}),
    }

exchanges = get_exchanges()


def fetch_ohlcv(symbol, timeframe, limit=300):
    """Returns (df, exchange_name, error_text). df is None if every exchange failed."""
    errors = []
    for ex_name, ex in exchanges.items():
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) >= 60:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df, ex_name, None
            errors.append(f"{ex_name}: not enough candles")
        except Exception as e:
            errors.append(f"{ex_name}: {type(e).__name__}")
    return None, None, " | ".join(errors)


# ─────────────────────────────────────────────
# 2. INDICATORS (ATR, SUPERTREND, PIVOTS, TRENDLINE)
# ─────────────────────────────────────────────
def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_supertrend(df, period=10, multiplier=3.0):
    df = df.copy()
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    hl2 = (high + low) / 2.0
    atr = calculate_atr(df, period).values

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    n = len(df)
    trend = np.ones(n, dtype=int)
    st_line = np.full(n, np.nan)
    st_line[0] = final_lower[0]

    for i in range(1, n):
        # Final lower band
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Final upper band
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Trend switch
        if trend[i - 1] == 1 and close[i] < final_lower[i]:
            trend[i] = -1
        elif trend[i - 1] == -1 and close[i] > final_upper[i]:
            trend[i] = 1
        else:
            trend[i] = trend[i - 1]

        st_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    df['supertrend'] = st_line
    df['trend'] = trend
    return df


def find_pivot_lows(low, strength=6):
    pl_p, pl_i = [], []
    n = len(low)
    for i in range(strength, n - strength):
        window = low[i - strength: i + strength + 1]
        if low[i] == window.min():
            pl_p.append(low[i])
            pl_i.append(i)
    return pl_p, pl_i


def find_closest_support_zone(close, pl_prices, pl_indices, max_threshold,
                              back_bars=150, min_touches=2, trend_only=True):
    size = len(pl_prices)
    if size < max(min_touches, 2):
        return None

    last_idx = len(close) - 1
    current_close = close[-1]
    lookback = last_idx - back_bars

    best_zone = None
    min_dist = np.inf

    for i in range(size - 1):
        if pl_indices[i] < lookback:
            continue
        p1_y, p1_x = pl_prices[i], pl_indices[i]

        for j in range(i + 1, size):
            p2_y, p2_x = pl_prices[j], pl_indices[j]
            slope = (p2_y - p1_y) / (p2_x - p1_x)

            if trend_only and slope <= 0:
                continue

            touches, is_broken = 0, False
            max_up, max_down = 0.0, 0.0

            for k in range(i, size):
                pk_x = pl_indices[k]
                pk_y = pl_prices[k]
                exp_y = p1_y + slope * (pk_x - p1_x)
                diff = pk_y - exp_y

                if abs(diff) <= max_threshold:
                    touches += 1
                    if diff > max_up:
                        max_up = diff
                    if diff < max_down:
                        max_down = diff
                if diff < -max_threshold:
                    is_broken = True
                    break

            if is_broken or touches < min_touches:
                continue

            # Extra check: no candle CLOSE broke the line between the first pivot and now
            xs = np.arange(p1_x, last_idx + 1)
            line_vals = p1_y + slope * (xs - p1_x)
            if np.any(close[p1_x: last_idx + 1] < line_vals - max_threshold):
                continue

            proj_y = p1_y + slope * (last_idx - p1_x)
            dist = abs(proj_y - current_close)
            if dist < min_dist:
                min_dist = dist
                best_zone = {'top': proj_y + max_up, 'bottom': proj_y + max_down}

    return best_zone


def flip_df(df):
    """Mirror prices (multiply by -1). A SHORT setup on resistance becomes a LONG setup
    on support in the mirrored data, so one set of logic serves both directions."""
    f = df.copy()
    f['open'] = -df['open']
    f['close'] = -df['close']
    f['high'] = -df['low']
    f['low'] = -df['high']
    return f


def get_trend(df, closed_only=True):
    """Supertrend direction of the last (closed) candle: +1 bullish, -1 bearish."""
    if closed_only:
        df = df.iloc[:-1]
    df = calculate_supertrend(df.reset_index(drop=True))
    return int(df['trend'].iloc[-1])


# ─────────────────────────────────────────────
# 3. CONFLUENCE DETECTION LOGIC
#    LONG : Supertrend bullish + candle low touches rising trendline support zone
#           + lower wick >= min_wick of candle range
#    SHORT: Supertrend bearish + candle high touches falling trendline resistance zone
#           + upper wick >= min_wick of candle range
# ─────────────────────────────────────────────
def check_confluence(df, timeframe, direction='long', closed_only=True,
                     trend_only=True, min_wick=0.20):
    if direction == 'short':
        df = flip_df(df)

    # ccxt's last candle is still forming. Drop it so the signal does not repaint.
    if closed_only:
        df = df.iloc[:-1]
    df = df.reset_index(drop=True)

    strength = 5 if timeframe in ('1m', '3m', '5m') else 8
    touches = 2 if timeframe in ('1m', '3m') else 3

    df = calculate_supertrend(df)
    atr = calculate_atr(df).iloc[-1]
    max_thresh = atr * 0.5

    pl_p, pl_i = find_pivot_lows(df['low'].values, strength=strength)
    zone = find_closest_support_zone(
        df['close'].values, pl_p, pl_i, max_thresh,
        min_touches=touches, trend_only=trend_only,
    )

    row = df.iloc[-1]
    candle_range = row['high'] - row['low']
    body_bottom = min(row['open'], row['close'])
    wick = body_bottom - row['low']          # lower wick (= upper wick for SHORT, after the flip)

    wick_ratio = wick / candle_range if candle_range > 0 else 0.0
    has_wick_rejection = wick_ratio >= min_wick

    supertrend_ok = row['trend'] == 1

    zone_touched = False
    if zone:
        # Wick dips into the zone, and the close holds on the right side of the zone
        if row['low'] <= (zone['top'] + max_thresh * 0.2) and row['close'] >= zone['bottom']:
            zone_touched = True

    is_confluence = bool(supertrend_ok and zone_touched and has_wick_rejection)

    # Convert back to real prices for SHORT
    sign = -1.0 if direction == 'short' else 1.0
    zone_top = zone_bot = None
    if zone:
        if direction == 'short':
            zone_top, zone_bot = -zone['bottom'], -zone['top']
        else:
            zone_top, zone_bot = zone['top'], zone['bottom']

    candle_time_ist = row['timestamp'] + pd.Timedelta(hours=5, minutes=30)

    return {
        'confluence': is_confluence,
        'price': sign * row['close'],
        'supertrend': sign * row['supertrend'],
        'zone_top': zone_top,
        'zone_bot': zone_bot,
        'wick_ratio': round(wick_ratio * 100, 1),
        'timestamp': candle_time_ist.strftime('%d-%b %H:%M'),
    }


# ─────────────────────────────────────────────
# 4. STREAMLIT UI & SCANNER
# ─────────────────────────────────────────────
st.title("🎯 Trendline + Supertrend Wick Confluence Scanner")
st.caption("Long + Short | Higher-TF trend filter | Auto failover: Binance Spot ➔ OKX ➔ KuCoin")

pairs_input = st.text_input(
    "Crypto Pairs (Comma separated):",
    "RENDER/USDT, BTC/USDT, ETH/USDT, SOL/USDT, SUI/USDT",
)
symbol_list = [s.strip().upper() for s in pairs_input.split(",") if s.strip()]

ALL_TFS = ['1m', '3m', '5m', '15m', '1h', '4h']
BUCKET_OF = {'1m': 'scalp', '3m': 'scalp', '5m': 'scalp', '15m': 'm15', '1h': 'htf', '4h': 'htf'}
# Higher timeframe whose Supertrend must agree with the signal
HTF_OF = {'1m': '15m', '3m': '15m', '5m': '15m', '15m': '1h', '1h': '4h', '4h': '1d'}

with st.expander("⚙️ Settings"):
    direction_choice = st.selectbox("Direction", ["Both", "Long only", "Short only"])
    use_htf = st.checkbox(
        "Higher-TF trend must agree (1m/3m/5m → 15m, 15m → 1h, 1h → 4h, 4h → 1d)",
        value=True,
        help="LONG needs the higher-TF Supertrend bullish, SHORT needs it bearish",
    )
    min_wick_pct = st.slider("Min wick size (% of candle range)", 5, 60, 20)
    closed_only = st.checkbox(
        "Use only CLOSED candles (signal won't repaint)", value=True,
        help="Off = checks the live, still-forming candle",
    )
    trend_only = st.checkbox(
        "Only sloped trendlines (rising for LONG support, falling for SHORT resistance)",
        value=True,
        help="Off = flat and opposite-slope lines are allowed too",
    )

if direction_choice == "Long only":
    directions = ['long']
elif direction_choice == "Short only":
    directions = ['short']
else:
    directions = ['long', 'short']

col_scalp, col_15m, col_htf = st.columns(3)

with col_scalp:
    st.subheader("⚡ Scalping (1m, 3m, 5m)")
with col_15m:
    st.subheader("⏱️ Intraday (15m)")
with col_htf:
    st.subheader("📈 High Timeframe (1h & 4h)")


def show_results(col, rows, empty_msg):
    with col:
        if rows:
            st.success(f"{len(rows)} Setups Found!")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.info(empty_msg)


if st.button("Start Scan / Refresh"):
    results = {'scalp': [], 'm15': [], 'htf': []}
    failed = []

    fetch_tfs = ALL_TFS + (['1d'] if use_htf else [])
    progress = st.progress(0.0)
    total_steps = max(len(symbol_list) * len(fetch_tfs), 1)
    step = 0

    for sym in symbol_list:
        data, ex_of = {}, {}

        # Fetch every timeframe once per symbol
        for tf in fetch_tfs:
            df, ex_used, err = fetch_ohlcv(sym, tf)
            step += 1
            progress.progress(min(step / total_steps, 1.0))
            if df is None:
                failed.append({'Symbol': sym, 'TF': tf, 'Reason': err})
            else:
                data[tf] = df
                ex_of[tf] = ex_used

        # Evaluate
        for tf in ALL_TFS:
            df = data.get(tf)
            if df is None:
                continue

            htf_trend = None
            if use_htf:
                htf_df = data.get(HTF_OF[tf])
                if htf_df is None:
                    continue  # HTF fetch failure is already listed in the failures table
                try:
                    htf_trend = get_trend(htf_df, closed_only)
                except Exception as e:
                    failed.append({'Symbol': sym, 'TF': HTF_OF[tf],
                                   'Reason': f"trend calc error: {type(e).__name__}: {e}"})
                    continue

            for d in directions:
                if use_htf and htf_trend != (1 if d == 'long' else -1):
                    continue
                try:
                    res = check_confluence(
                        df, tf, direction=d, closed_only=closed_only,
                        trend_only=trend_only, min_wick=min_wick_pct / 100.0,
                    )
                except Exception as e:
                    failed.append({'Symbol': sym, 'TF': tf,
                                   'Reason': f"calc error: {type(e).__name__}: {e}"})
                    continue

                if res['confluence']:
                    results[BUCKET_OF[tf]].append({
                        'Symbol': sym,
                        'Dir': '🟢 LONG' if d == 'long' else '🔴 SHORT',
                        'TF': tf,
                        'Price': res['price'],
                        'Wick%': f"{res['wick_ratio']}%",
                        'Zone Top': round(res['zone_top'], 6),
                        'Zone Bottom': round(res['zone_bot'], 6),
                        'Supertrend': round(res['supertrend'], 6),
                        'HTF': (f"{HTF_OF[tf]} {'▲' if htf_trend == 1 else '▼'}" if use_htf else '-'),
                        'Candle (IST)': res['timestamp'],
                        'Exchange': ex_of[tf],
                    })

    progress.empty()

    show_results(col_scalp, results['scalp'], "No 1m/3m/5m Confluence setups right now.")
    show_results(col_15m, results['m15'], "No 15m Confluence setups right now.")
    show_results(col_htf, results['htf'], "No 1h/4h Confluence setups right now.")

    if failed:
        with st.expander(f"⚠️ {len(failed)} fetch/calc failures"):
            st.dataframe(pd.DataFrame(failed), hide_index=True)
