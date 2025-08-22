import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict

def statistical_arbitrage_pairs(
    data: pd.DataFrame,
    asset_y: str = "Asset1",        # dependent variable
    asset_x: str = "Asset2",        # independent variable
    window: int = 60,
    entry_z: float = 2.0,           # entry threshold
    exit_z: float = 0.0,            # exit threshold
    max_holding: int | None = None, # optional time stop (bars)
    cost_per_leg: float = 0.0,      # transaction cost per leg
    periods_per_year: int = 252,
    copy: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Statistical Arbitrage (Pairs Trading) using rolling OLS regression.

    Logic
    -----
    - Estimate alpha and beta from rolling OLS:  y_t = alpha + beta * x_t + e_t
    - Spread_t = y_t - (alpha + beta * x_t)
    - Z-score of spread: (spread - mean) / std, rolling over `window`
    - Entry:
        Z <= -entry_z -> long spread (long y, short beta*x)
        Z >=  entry_z -> short spread (short y, long beta*x)
    - Exit:
        |Z| <= exit_z or max holding reached
    - Position applied on the NEXT bar to avoid look-ahead.
    - Strategy return: pos * (r_y - beta_{t-1} * r_x)
    - Costs: 2 legs per entry/exit. Flips (long→short) count as 2 events.

    Returns
    -------
    df : pd.DataFrame with alpha, beta, spread, z-score, signals, positions, returns, equity curve
    metrics : dict with performance statistics
    """
    df = data.copy() if copy else data
    df = df.sort_index()

    if asset_y not in df.columns or asset_x not in df.columns:
        raise KeyError(f"Columns '{asset_y}' and/or '{asset_x}' not found in data.")

    df["Alpha"], df["Beta"], df["Spread"] = np.nan, np.nan, np.nan

    # Rolling OLS for alpha and beta
    for i in range(window, len(df)):
        hist = df.iloc[i - window:i][[asset_y, asset_x]].dropna()
        if len(hist) < window // 2:
            continue
        X = sm.add_constant(hist[asset_x].values)
        y = hist[asset_y].values
        res = sm.OLS(y, X).fit()
        alpha, beta = res.params[0], res.params[1]
        idx = df.index[i]
        df.at[idx, "Alpha"], df.at[idx, "Beta"] = alpha, beta
        df.at[idx, "Spread"] = df.at[idx, asset_y] - (alpha + beta * df.at[idx, asset_x])

    # Z-score of spread
    df["MeanSpread"] = df["Spread"].rolling(window=window, min_periods=window).mean()
    df["StdSpread"]  = df["Spread"].rolling(window=window, min_periods=window).std(ddof=0)
    df["Z"] = (df["Spread"] - df["MeanSpread"]) / df["StdSpread"]

    # Signals and positions
    df["Signal"] = 0
    df.loc[df["Z"] <= -entry_z, "Signal"] = 1
    df.loc[df["Z"] >=  entry_z, "Signal"] = -1
    df.loc[df["Z"].abs() <= exit_z, "Signal"] = 0

    # Position state machine
    df["Position"] = 0
    pos, held = 0, 0
    for i in range(len(df)):
        z = df["Z"].iat[i]
        if np.isnan(z):
            df["Position"].iat[i] = pos
            continue
        if pos != 0 and max_holding is not None:
            if held >= max_holding:
                pos, held = 0, 0
        sig = df["Signal"].iat[i]
        if sig == 0 and abs(z) <= exit_z:
            pos, held = 0, 0
        elif sig == 1:
            if pos != 1: pos, held = 1, 0
            else: held += 1
        elif sig == -1:
            if pos != -1: pos, held = -1, 0
            else: held += 1
        df["Position"].iat[i] = pos

    # Apply NEXT-bar execution
    df["Position"] = df["Position"].shift(1).fillna(0)

    # Returns
    df["RetY"] = np.log(df[asset_y] / df[asset_y].shift(1))
    df["RetX"] = np.log(df[asset_x] / df[asset_x].shift(1))
    df["BetaLag"] = df["Beta"].shift(1)

    df["StratRet_gross"] = df["Position"] * (df["RetY"] - df["BetaLag"] * df["RetX"])

    # Costs: each trade event = 2 legs, flips = 2 events
    pos_change = df["Position"].diff().fillna(df["Position"]).abs()
    flip = (df["Position"].shift(1).fillna(0) * df["Position"]).lt(0).astype(int)
    events = (pos_change > 0).astype(int) + flip
    legs = 2 * events
    df["Costs"] = -cost_per_leg * legs

    df["StratRet"] = df["StratRet_gross"] + df["Costs"]
    df["EquityCurve"] = np.exp(df["StratRet"].cumsum())

    # Metrics
    if df["StratRet"].notna().any():
        total_ret = float(df["EquityCurve"].iloc[-1] - 1)
        avg, vol = float(df["StratRet"].mean()), float(df["StratRet"].std(ddof=0))
        ann_ret = periods_per_year * avg
        ann_vol = np.sqrt(periods_per_year) * vol
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rollmax = df["EquityCurve"].cummax()
        dd = df["EquityCurve"] / rollmax - 1.0
        mdd = float(dd.min())
        trades = int(events.sum())
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
    }

    return df, metrics
