def statistical_arbitrage_strategy(data, window=60, entry_threshold=2, exit_threshold=0):
    """
    Implementa uma estratégia básica de Arbitragem Estatística (cointegração).

    Args:
        data (pd.DataFrame): DataFrame com os dados de preço, deve conter colunas para os dois ativos (e.g., 'Asset1', 'Asset2').
        window (int): Período para calcular a relação de cointegração.
        entry_threshold (float): Limiar de desvio padrão para entrar na posição.
        exit_threshold (float): Limiar de desvio padrão para sair da posição.

    Returns:
        pd.DataFrame: DataFrame com os sinais de compra/venda.
    """
    # Certifique-se de que os dados estão em ordem cronológica
    data = data.sort_index()

    # Calcula a relação de cointegração (spread) usando regressão linear
    data["HedgeRatio"] = np.nan
    data["Spread"] = np.nan
    data["ZScore"] = np.nan

    for i in range(window, len(data)):
        history = data.iloc[i-window:i]
        # Adiciona uma constante para a regressão
        X = sm.add_constant(history["Asset2"])
        model = sm.OLS(history["Asset1"], X)
        results = model.fit()
        hedge_ratio = results.params[1]
        data.loc[data.index[i], "HedgeRatio"] = hedge_ratio
        data.loc[data.index[i], "Spread"] = data["Asset1"].iloc[i] - hedge_ratio * data["Asset2"].iloc[i]

    # Calcula o Z-Score do spread
    data["RollingMeanSpread"] = data["Spread"].rolling(window=window).mean()
    data["RollingStdSpread"] = data["Spread"].rolling(window=window).std()
    data["ZScore"] = (data["Spread"] - data["RollingMeanSpread"]) / data["RollingStdSpread"]

    data["Signal"] = 0
    # Sinal de compra (long Asset1, short Asset2): Z-Score abaixo do limiar negativo
    data.loc[data["ZScore"] < -entry_threshold, "Signal"] = 1
    # Sinal de venda (short Asset1, long Asset2): Z-Score acima do limiar positivo
    data.loc[data["ZScore"] > entry_threshold, "Signal"] = -1
    # Fechar posição: Z-Score entre os limiares de saída
    data.loc[(data["ZScore"] > -exit_threshold) & (data["ZScore"] < exit_threshold), "Signal"] = 0

    return data

if __name__ == '__main__':
    # Exemplo de uso com dados fictícios
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200))
    # Ativos correlacionados com algum ruído
    asset2_prices = np.random.normal(loc=50, scale=2, size=200).cumsum() + 100
    asset1_prices = asset2_prices * 1.05 + np.random.normal(loc=0, scale=1, size=200)

    sample_data = pd.DataFrame({"Date": dates, "Asset1": asset1_prices, "Asset2": asset2_prices})
    sample_data.set_index("Date", inplace=True)

    strategy_results = statistical_arbitrage_strategy(sample_data.copy())
    print(strategy_results[["Asset1", "Asset2", "Spread", "ZScore", "Signal"]].tail())

