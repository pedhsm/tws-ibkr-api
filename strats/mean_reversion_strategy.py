def mean_reversion_strategy(data, window=20, num_std=2):
    """
    Implementa uma estratégia básica de Reversão à Média.

    Args:
        data (pd.DataFrame): DataFrame com os dados de preço, deve conter uma coluna 'Close'.
        window (int): Período da média móvel.
        num_std (int): Número de desvios padrão para as bandas de Bollinger.

    Returns:
        pd.DataFrame: DataFrame com os sinais de compra/venda.
    """
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['StdDev'] = data['Close'].rolling(window=window).std()
    data['UpperBand'] = data['SMA'] + (data['StdDev'] * num_std)
    data['LowerBand'] = data['SMA'] - (data['StdDev'] * num_std)

    data['Signal'] = 0
    # Sinal de compra: preço abaixo da banda inferior
    data.loc[data['Close'] < data['LowerBand'], 'Signal'] = 1
    # Sinal de venda: preço acima da banda superior
    data.loc[data['Close'] > data['UpperBand'], 'Signal'] = -1

    return data

