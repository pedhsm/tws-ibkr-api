import time
import threading
from datetime import datetime
from typing import Dict, Optional
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

class TradingApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self,self)
        self.data: Dict[int,pd.DataFrame] = {}


    def get_historical_data(self,reqId:int, contract: Contract) -> pd.DataFrame:
        self.data[reqId] = pd.DataFrame(columns=["time","high","low","close"])
        self.data[reqId].set_index("time",inplace=True)
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="MIDPOINT", #It's the avg between the bid(highest value sold) and the ask(lowest value sold)
            useRTH=0, #RTH = Regular Trading Hours
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )
        time.sleep(3)
        return self.data[reqId]
    
    def historicalData(self, reqId:int, bar: BarData) -> None:
        df = self.data[reqId]
        df.loc[
            pd.to_datetime(bar.date, unit="s"),
            ["high","low","close"]
        ] = [bar.high, bar.low, bar.close]
        df = df.astype(float)
        self.data[reqId] = df

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