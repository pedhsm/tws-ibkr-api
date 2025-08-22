import pandas as pd
import numpy as np
from typing import Tuple, Dict

def market_making_strategy(
    data: pd.DataFrame,
    price_col: str = "MidPrice",
    spread_bps: float = 10.0,          # total half-spread in basis points (10 bps = 0.10%)
    order_size: int = 1,               # units per fill
    inventory_limit: int = 100,        # symmetric (+/-)
    skew_bps: float = 0.0,             # max mid shift in bps at full inventory utilization
    maker_fee_bps: float = 0.0,        # apply per-side (negative means rebate)
    allow_both_side_fills: bool = True,# if True, both bid/ask can fill in same bar
    copy: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simple market-making backtest:
      - Quotes around mid with optional inventory-based skew.
      - Fills occur when the *current* mid crosses the *previous bar's* quotes.
      - Updates inventory, cash, realized/mark-to-market PnL, and basic metrics.

    Notes
    -----
    - spread_bps is PER SIDE: bid = mid*(1 - s), ask = mid*(1 + s), where s = spread_bps/1e4.
    - Skew shifts the quoting mid proportionally to inventory utilization:
        eff_mid = mid * (1 + skew_bps/1e4 * (inventory / inventory_limit)).
      Positive inventory -> quotes shift up (harder to buy more, easier to sell).
    - Fees are applied on notional of each fill: cash -= price*qty*(fee_bps/1e4).
      Use negative maker_fee_bps to model rebates (cash increases).

    Returns
    -------
    df : DataFrame with quotes, fills, inventory, cash, PnL, equity
    metrics : dict with summary stats
    """
    if price_col not in data.columns:
        raise KeyError(f"Column '{price_col}' not found in data.")

    df = data.copy() if copy else data
    df = df.sort_index()

    s = spread_bps / 1e4
    fee = maker_fee_bps / 1e4
    inv_lim = float(inventory_limit)

    # Preallocate columns
    df["BidQuote"] = np.nan
    df["AskQuote"] = np.nan
    df["FillSide"] = 0          # +1 buy @ bid, -1 sell @ ask (can be +/-2 if both sides fill and order_size>1)
    df["FillPrice"] = np.nan
    df["FillQty"] = 0
    df["Inventory"] = 0
    df["Cash"] = 0.0
    df["RealizedPnL"] = 0.0
    df["Equity"] = 0.0

    # Initialize day 0 quotes around the first mid
    if len(df) == 0:
        return df, {"Trades": 0, "Turnover": 0.0, "MaxInv": 0, "FinalEquity": 0.0}

    inv = 0
    cash = 0.0
    realized = 0.0
    trades = 0
    turnover = 0.0
    max_inv_abs = 0

    # Set first bar quotes (no fills before we have prior quotes)
    mid0 = float(df[price_col].iloc[0])
    eff_mid0 = mid0  # inv = 0
    df.iloc[0, df.columns.get_loc("BidQuote")] = eff_mid0 * (1 - s)
    df.iloc[0, df.columns.get_loc("AskQuote")] = eff_mid0 * (1 + s)
    df.iloc[0, df.columns.get_loc("Inventory")] = inv
    df.iloc[0, df.columns.get_loc("Cash")] = cash
    df.iloc[0, df.columns.get_loc("RealizedPnL")] = realized
    df.iloc[0, df.columns.get_loc("Equity")] = cash + inv * mid0

    for i in range(1, len(df)):
        mid = float(df[price_col].iloc[i])

        # Quote using prior inventory (inventory affects today’s skew)
        skew_frac = (skew_bps / 1e4) * (inv / inv_lim) if inv_lim > 0 else 0.0
        eff_mid = mid * (1.0 + skew_frac)
        bid = eff_mid * (1 - s)
        ask = eff_mid * (1 + s)

        # Previous bar quotes determine fills
        prev_bid = float(df["BidQuote"].iloc[i-1])
        prev_ask = float(df["AskQuote"].iloc[i-1])

        fill_side = 0
        fill_qty = 0
        fill_price = np.nan

        # Check fills against *previous* quotes to avoid look-ahead
        # Buy @ bid if mid <= prev_bid and inventory below limit
        if mid <= prev_bid and inv < inventory_limit:
            qty = min(order_size, inventory_limit - inv)
            inv += qty
            # cash decreases by price*qty plus fees (fees positive reduce cash; negative rebate increases cash)
            cash -= prev_bid * qty
            cash -= prev_bid * qty * fee
            fill_side += +1
            fill_qty += qty
            fill_price = prev_bid
            trades += 1
            turnover += prev_bid * qty

        # Sell @ ask if mid >= prev_ask and inventory above -limit
        if (allow_both_side_fills or fill_side == 0) and mid >= prev_ask and inv > -inventory_limit:
            qty = min(order_size, inv + inventory_limit)  # how much we can sell without exceeding -limit
            if qty > 0:
                inv -= qty
                cash += prev_ask * qty
                cash -= prev_ask * qty * fee
                fill_side += -1
                fill_qty += qty
                # if both sides fill, prefer last price for record; PnL already handled in cash/inv
                fill_price = prev_ask if np.isnan(fill_price) else prev_ask
                trades += 1
                turnover += prev_ask * qty

        realized = cash  # realized component kept in cash; mark-to-market below

        # Record state at i
        df.iloc[i, df.columns.get_loc("FillSide")] = fill_side
        df.iloc[i, df.columns.get_loc("FillQty")] = fill_qty
        if not np.isnan(fill_price):
            df.iloc[i, df.columns.get_loc("FillPrice")] = fill_price

        df.iloc[i, df.columns.get_loc("Inventory")] = inv
        df.iloc[i, df.columns.get_loc("Cash")] = cash
        df.iloc[i, df.columns.get_loc("RealizedPnL")] = realized
        df.iloc[i, df.columns.get_loc("Equity")] = cash + inv * mid

        # Set today's quotes for next bar fill checks
        df.iloc[i, df.columns.get_loc("BidQuote")] = bid
        df.iloc[i, df.columns.get_loc("AskQuote")] = ask

        max_inv_abs = max(max_inv_abs, abs(inv))

    metrics = {
        "Trades": int(trades),
        "Turnover": float(turnover),
        "MaxInv": int(max_inv_abs),
        "FinalEquity": float(df["Equity"].iloc[-1]),
        "FinalInventory": int(inv),
    }
    return df, metrics
