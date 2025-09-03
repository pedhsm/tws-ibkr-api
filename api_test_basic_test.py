import time
import threading
from datetime import datetime
from typing import Dict, List
import pandas as pd 
import logging
import numpy as np 

# =========================
# Logging config (file + terminal)
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
# ========== BASIC LIBS ============
#%% 
from ibapi.client import EClient 
from ibapi.wrapper import EWrapper 
from ibapi.contract import Contract 
from ibapi.order import Order
from ibapi.common import BarData

class TradingApp(EWrapper, EClient):
    def __init__(self, apply_us_stock_volume_x100: bool = False):
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self._done = threading.Event()
        self.apply_us_stock_volume_x100 = apply_us_stock_volume_x100

        # signals/buffers
        self._connected = threading.Event()
        self._cd_event = threading.Event()
        self._cd_results: Dict[int, List] = {}
        self._historical_data_events: Dict[int, threading.Event] = {}

    # -------------------- CONNECTION --------------------
    def nextValidId(self, orderId: int):
        logging.info(f"nextValidId={orderId} (connected)")
        self._connected.set()

    def connect_and_run(self, host: str, port: int, clientId: int):
        self.connect(host, port, clientId)
        thread = threading.Thread(target=self.run)
        thread.start()
        self._connected.wait(timeout=10) # Wait for connection to be established
        if not self._connected.is_set():
            logging.error("Failed to connect to IB TWS/Gateway.")
            raise ConnectionError("Could not connect to IB TWS/Gateway.")

    def disconnect(self):
        # Corrected: Call the disconnect method from the parent class (EClient)
        super().disconnect()
        logging.info("Disconnected from IB TWS/Gateway.")

# -------------------- HISTORICAL DATA --------------------
    def get_historical_data(self, reqId: int, contract: Contract, durationStr: str = "1 D", barSizeSetting: str = "1 min", whatToShow: str = "TRADES", endDateTime:str = None) -> pd.DataFrame:
        if endDateTime is None:
            endDateTime = datetime.now().strftime("%Y%m%d %H:%M:%S US/Eastern")
        self.data[reqId] = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        self.data[reqId].set_index("time", inplace=True)
        self._historical_data_events[reqId] = threading.Event()

        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            endDateTime=endDateTime,             # "" = now
            useRTH=0,
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )

        # Wait for historical data to be received
        self._historical_data_events[reqId].wait(timeout=30) # Increased timeout
        if not self._historical_data_events[reqId].is_set():
            logging.warning(f"Timeout waiting for historical data for reqId {reqId}")

        return self.data[reqId]

    def historicalData(self, reqId: int, bar: BarData) -> None:
        df = self.data[reqId]
        # Try to read it as a timestamp string w Hour
        ts = pd.to_datetime(bar.date, format="%Y%m%d %H:%M:%S", errors="coerce")
        if pd.isna(ts):
            # Try to read it as a timestamp string w Data (only)
            ts = pd.to_datetime(bar.date, format="%Y%m%d", errors="coerce")
        # If it is numerical (epoch seconds)
        if pd.isna(ts):
            try:
                ts = pd.to_datetime(int(bar.date), unit="s", utc=True, errors="coerce")
            except ValueError:
                pass  # It is Nat if it's not a string or epoch 

        vol = bar.volume * (100 if self.apply_us_stock_volume_x100 else 1)
        df.loc[ts, ["open", "high", "low", "close", "volume"]] = [
            bar.open, bar.high, bar.low, bar.close, vol
        ]

        self.data[reqId] = df.astype({
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })

        
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        logging.info(f"Historical data for reqId {reqId} ended. Start: {start}, End: {end}")
        if reqId in self._historical_data_events:
            self._historical_data_events[reqId].set()

    @staticmethod
    def get_stock_contract(symbol:str) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract 
# ========== IBKR API LIB ============
#%% 
app = TradingApp() # Create your application 
app.connect("your_IP","socket_port","client_id") # I.e: ("127.0.0.1",7497,1) 
threading.Thread(target=app.run,daemon=True).start()
nvda = TradingApp.get_stock_contract("NVDA") # Use the function to get the choosen ticket data 
data = app.get_historical_data(99,nvda) # Pass the df["high","low","close"]
data # Print the Dataframe

# U SHOULD GET THOSE IN A JUPYTER NOTEBOOK FOR FURTHER IMPROVEMNTS AND BRING THEM TO THE WPP GROUP -- Pedro