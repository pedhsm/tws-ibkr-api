import numpy as np
import pandas as pd

def mean_reversion_strategy(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = "Close",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Basic mean-reversion strategy using Bollinger Bands + z-score.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain the price column specified by `price_col`.
    window : int, default 20
        Rolling window for the moving average (SMA) and standard deviation.
    num_std : float, default 2.0
        Number of standard deviations for the bands.
    price_col : str, default "Close"
        Name of the price column to use.
    copy : bool, default True
        If True, work on a copy to avoid mutating the original DataFrame.

    Returns
    -------
    pd.DataFrame
        With the following columns added:
        - SMA, StdDev, UpperBand, LowerBand
        - Z (z-score)
        - Signal (signal at the current close)
        - Position (signal shifted to apply on the next bar, to avoid look-ahead)
        - MarketRet, StratRet, EquityCurve
    """
    if price_col not in data.columns:
        raise KeyError(f"Column '{price_col}' not found in data.columns")

    df = data.copy() if copy else data

    # Ensure chronological order by index
    df = df.sort_index()

    # Rolling stats (min_periods=window to control NaNs at the start)
    roll = df[price_col].rolling(window=window, min_periods=window)
    df["SMA"] = roll.mean()
    df["StdDev"] = roll.std(ddof=0)  # ddof=0 for population std (optional)

    df["UpperBand"] = df["SMA"] + num_std * df["StdDev"]
    df["LowerBand"] = df["SMA"] - num_std * df["StdDev"]

    # Z-score shows how stretched price is from the mean
    df["Z"] = (df[price_col] - df["SMA"]) / df["StdDev"]

    # ---- Signals ----
    #  1 = long when close < lower band
    # -1 = short when close > upper band
    #  0 = flat when within the bands
    df["Signal"] = 0
    df.loc[df[price_col] < df["LowerBand"], "Signal"] = 1
    df.loc[df[price_col] > df["UpperBand"], "Signal"] = -1

    # Apply the position on the NEXT bar to avoid look-ahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)

    # ---- Returns and equity curve ----
    # Use log returns for stable compounding
    df["MarketRet"] = np.log(df[price_col] / df[price_col].shift(1))
    df["StratRet"] = df["Position"] * df["MarketRet"]
    df["EquityCurve"] = np.exp(df["StratRet"].cumsum())

    return df
