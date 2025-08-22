import numpy as np
import pandas as pd
from typing import Tuple, Dict, Literal

def trend_following_cross(
    data: pd.DataFrame,
    price_col: str = "Close",
    short_window: int = 50,
    long_window: int = 200,
    avg: Literal["EMA", "SMA"] = "EMA",   # <-- choose EMA or SMA
    allow_short: bool = False,            # flip short on death cross if True; else go flat
    cost_per_trade: float = 0.0,          # round-turn approximation per position change
    periods_per_year: int = 252,
    copy: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Trend-Following crossover with EMA/SMA option, next-bar execution, costs, and metrics.

    Rules
    -----
    - Compute two MAs (short & long) using EMA or SMA.
    - Golden cross (short > long): go LONG.
    - Death cross (short < long): go SHORT if allow_short=True else go FLAT.
    - Apply position on NEXT bar to avoid look-ahead bias.
    """
    df = data.copy() if copy else data
    df = df.sort_index()

    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in data.")
    if short_window >= long_window:
        raise ValueError("short_window must be strictly less than long_window.")

    # --- Moving averages
    if avg.upper() == "EMA":
        df["MA_Short"] = df[price_col].ewm(span=short_window, adjust=False, min_periods=short_window).mean()
        df["MA_Long"]  = df[price_col].ewm(span=long_window,  adjust=False, min_periods=long_window).mean()
    elif avg.upper() == "SMA":
        df["MA_Short"] = df[price_col].rolling(short_window, min_periods=short_window).mean()
        df["MA_Long"]  = df[price_col].rolling(long_window,  min_periods=long_window).mean()
    else:
        raise ValueError("avg must be 'EMA' or 'SMA'.")

    # --- Cross detection
    rel = (df["MA_Short"] > df["MA_Long"]).astype(int)
    cross = rel.diff().fillna(0)  # +1 golden cross; -1 death cross

    # Signals at detection bar
    df["Signal"] = 0
    df.loc[cross == +1, "Signal"] = 1
    if allow_short:
        df.loc[cross == -1, "Signal"] = -1
    else:
        # we’ll interpret -1 cross as exit to flat
        df.loc[cross == -1, "Signal"] = 0

    # --- Position state machine
    df["Position"] = 0
    pos = 0
    for i in range(len(df)):
        sig = df["Signal"].iat[i]
        if sig == 1:
            pos = 1
        elif sig == -1:
            pos = -1
        elif sig == 0 and cross.iat[i] == -1 and not allow_short:
            pos = 0
        df["Position"].iat[i] = pos

    # Next-bar execution (no look-ahead)
    df["Position"] = df["Position"].shift(1).fillna(0)

    # --- Returns, costs, equity
    df["MarketRet"] = np.log(df[price_col] / df[price_col].shift(1))
    df["StratRet_gross"] = df["Position"] * df["MarketRet"]

    changes = df["Position"].diff().fillna(df["Position"]).ne(0).astype(int)
    df["Costs"] = -cost_per_trade * changes
    df["StratRet"] = df["StratRet_gross"] + df["Costs"]

    df["EquityCurve"] = np.exp(df["StratRet"].cumsum())

    # --- Metrics
    if df["StratRet"].notna().any():
        total_ret = float(df["EquityCurve"].iloc[-1] - 1)
        avg_ret, vol = float(df["StratRet"].mean()), float(df["StratRet"].std(ddof=0))
        ann_ret = periods_per_year * avg_ret
        ann_vol = np.sqrt(periods_per_year) * vol
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rollmax = df["EquityCurve"].cummax()
        dd = df["EquityCurve"] / rollmax - 1.0
        mdd = float(dd.min())
        trades = int(changes.sum())
    else:
        total_ret = ann_ret = ann_vol = sharpe = mdd = np.nan
        trades = 0

    metrics = {
        "TotalReturn": total_ret,
        "AnnualizedReturn": ann_ret,
        "AnnualizedVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "Trades": trades,
        "AvgType": avg.upper(),
    }
    return df, metrics
