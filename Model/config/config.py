import financedatabase as fd

ASSET_CONFIGS = {
    "equities": {
        "fetcher": lambda: fd.Equities().select(exclude_exchanges=False),
        "columns": ['symbol', 'currency', 'sector', 'industry_group', 'country', 'market_cap', 'exchange'],
        "partitions": ['currency', 'market_cap', 'sector', 'industry_group']
    },
    "cryptos": {
        "fetcher": lambda: fd.Cryptos().select(),
        "columns": ['symbol', 'cryptocurrency', 'currency', 'exchange'],
        "partitions": ['cryptocurrency']
    },
    "currencies": {
        "fetcher": lambda: fd.Currencies().select(),
        "columns": ['symbol', 'base_currency', 'quote_currency', 'exchange'],
        "partitions": ['base_currency']
    },
    "etfs": {
        "fetcher": lambda: fd.ETFs().select(),
        "columns": ['symbol', 'currency', 'category_group', 'exchange'],
        "partitions": ['currency', 'category_group']
    },
    "indices": {
        "fetcher": lambda: fd.Indices().select(),
        "columns": ['symbol', 'currency', 'category_group', 'exchange'],
        "partitions": ['currency', 'category_group']
    },
    "funds": {
        "fetcher": lambda: fd.Funds().select(),
        "columns": ['symbol', 'currency', 'category_group', 'exchange'],
        "partitions": ['currency', 'category_group']
    },
    "moneymarkets": {
        "fetcher": lambda: fd.Moneymarkets().select(),
        "columns": ['symbol', 'currency', 'exchange'],
        "partitions": ['currency']
    }
}


