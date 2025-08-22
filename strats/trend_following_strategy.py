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
