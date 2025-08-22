import pandas as pd
import numpy as np

def trend_following_strategy(data, short_window=50, long_window=200):
    """
    Implementa uma estratégia básica de Seguimento de Tendências (cruzamento de médias móveis).

    Args:
        data (pd.DataFrame): DataFrame com os dados de preço, deve conter uma coluna 'Close'.
        short_window (int): Período da média móvel curta.
        long_window (int): Período da média móvel longa.

    Returns:
        pd.DataFrame: DataFrame com os sinais de compra/venda.
    """
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()

    data['Signal'] = 0
    # Sinal de compra: SMA curta cruza acima da SMA longa
    data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1
    # Sinal de venda: SMA curta cruza abaixo da SMA longa
    data.loc[data['SMA_Short'] < data['SMA_Long'], 'Signal'] = -1

    return data

if __name__ == '__main__':
    # Exemplo de uso com dados fictícios
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=300))
    prices = np.random.normal(loc=100, scale=5, size=300).cumsum() + 100
    sample_data = pd.DataFrame({'Date': dates, 'Close': prices})
    sample_data.set_index('Date', inplace=True)

    strategy_results = trend_following_strategy(sample_data.copy())
    print(strategy_results[['Close', 'SMA_Short', 'SMA_Long', 'Signal']].tail())
