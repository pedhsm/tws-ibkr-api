def market_making_strategy(data, spread_percentage=0.001, inventory_limit=100):
    """
    Implementa uma estratégia básica de Criação de Mercado.

    Args:
        data (pd.DataFrame): DataFrame com os dados de preço, deve conter uma coluna 'MidPrice'.
        spread_percentage (float): Porcentagem do spread em torno do preço médio.
        inventory_limit (int): Limite de inventário para gerenciar o risco.

    Returns:
        pd.DataFrame: DataFrame com as ordens de compra/venda e inventário.
    """
    data["BidPrice"] = data["MidPrice"] * (1 - spread_percentage)
    data["AskPrice"] = data["MidPrice"] * (1 + spread_percentage)
    data["Inventory"] = 0
    data["Trades"] = 0

    current_inventory = 0

    for i in range(1, len(data)):
        # Simula a execução de ordens
        # Se o preço de mercado cair para o nosso BidPrice, compramos
        if data["MidPrice"].iloc[i] <= data["BidPrice"].iloc[i-1] and current_inventory < inventory_limit:
            current_inventory += 1
            data.loc[data.index[i], "Trades"] = 1 # Compra
        # Se o preço de mercado subir para o nosso AskPrice, vendemos
        elif data["MidPrice"].iloc[i] >= data["AskPrice"].iloc[i-1] and current_inventory > -inventory_limit:
            current_inventory -= 1
            data.loc[data.index[i], "Trades"] = -1 # Venda

        data.loc[data.index[i], "Inventory"] = current_inventory

    return data

