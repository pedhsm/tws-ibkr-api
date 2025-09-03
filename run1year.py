import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from teste_api_conx import TradingApp

# =========================
# Logging config (file + console)
# =========================
file_handler = logging.FileHandler("historical_fetch.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True
)
logger = logging.getLogger(__name__)

# -------------------------------------------
# 1) Instantiate and connect to TWS
# -------------------------------------------
app = TradingApp()

try:
    logger.info("Connecting to IB TWS/Gateway...")
    app.connect_and_run("127.0.0.1", 7497, clientId=65656)
    logger.info("Connected successfully!")
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    raise

# -------------------------------------------
# 2) Weekly historical data collection
# -------------------------------------------
logger.info("Starting historical data collection for AAPL (52 weekly chunks).")
aapl_contract = app.get_stock_contract("AAPL")

end_date = datetime(2025, 9, 2, 0, 0, 0)  # fixed and safe
total_weeks = 52
weeks_done = 0
all_historical_data = []

for i in range(total_weeks):
    start_date = end_date - timedelta(days=7)
    current_end_date_str = end_date.strftime("%Y%m%d %H:%M:%S US/Eastern")
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    req_id = 10000 + i

    print(f"🔹 Week {i+1:02d}/{total_weeks} | Requested period: {start_date_str} → {end_date_str}")
    logger.info(f"[Week {i+1}/{total_weeks}] Requesting data from {start_date_str} to {end_date_str} (TWS: {current_end_date_str})")

    try:
        historical_data_df_chunk = app.get_historical_data(
            reqId=req_id,
            contract=aapl_contract,
            endDateTime=current_end_date_str,
            durationStr="7 D",
            barSizeSetting="1 min",
            whatToShow="TRADES"
        )

        if not historical_data_df_chunk.empty:
            all_historical_data.append(historical_data_df_chunk)
            weeks_done += 1
            logger.info(f"✔ Week {i+1} received with {len(historical_data_df_chunk)} rows.")
        else:
            logger.warning(f"No data received for week {i+1}.")

    except Exception as e:
        logger.exception(f"❌ Error while requesting data for week {i+1}: {e}")

    end_date -= timedelta(days=7)
    time.sleep(10)

    if weeks_done > 0 and weeks_done % 52 == 0:
        try:
            partial_df = pd.concat(all_historical_data).sort_index()
            partial_df = partial_df[~partial_df.index.duplicated(keep='first')]
            filename = f"aapl_1min_checkpoint_{weeks_done}weeks.csv"
            partial_df.to_csv(filename)
            logger.info(f"[Checkpoint] Saved {filename} with {len(partial_df)} rows.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

# -------------------------------------------
# 3) Save final result
# -------------------------------------------
if all_historical_data:
    full_df = pd.concat(all_historical_data).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]

    print(f"\n✅ Collection finished! Total weeks: {weeks_done}")
    print(f"🔢 Total rows: {len(full_df)}")

    filename_final = "aapl_1min_historical_data_full.csv"
    full_df.to_csv(filename_final)
    logger.info(f"Full data saved to {filename_final}")

    if weeks_done % 52 != 0:
        try:
            checkpoint_name = f"aapl_1min_checkpoint_final_{weeks_done}weeks.csv"
            full_df.to_csv(checkpoint_name)
            logger.info(f"[Final Checkpoint] Saved {checkpoint_name} with {len(full_df)} rows.")
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}")
else:
    logger.warning("No data was collected.")

# -------------------------------------------
# 4) Disconnect from TWS
# -------------------------------------------
logger.info("Disconnecting from IB TWS/Gateway...")
app.disconnect()
logger.info("Disconnected.")
