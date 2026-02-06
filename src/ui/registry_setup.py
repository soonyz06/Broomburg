from src.utils.logg import log_info, log_df
from src.io.fetch import load_apikeys
from src.io.database import DatabaseManager
from src.io.fundamental import FundamentalManager
from src.io.price import PriceManager
from src.io.sec import EdgarToolsAPI

import plotly.express as px
from dash import dcc, html
import re
import polars as pl
import dash_ag_grid as dag # Ensure you have this installed


# =============================================================================
# 1. REGISTRY
# =============================================================================

class UniversalRegistry:
    def __init__(self):
        self._workers = {}

    def register(self, names):
        if isinstance(names, str):
            names = [names]
        def decorator(fn):
            for name in names:
                self._workers[name.lower()] = fn
            return fn
        return decorator

    def get(self, name):
        return self._workers.get(name.lower())

    def parse_args(self, args_list: list) -> dict:
        result = {}
        for arg in args_list:
            parts = re.split(r'\s*[=:]\s*', arg, 1)
            key = parts[0].lower()
            value = parts[1] if len(parts) > 1 else True
            if isinstance(value, str):
                if value.lower() == "true": value = True
                elif value.lower() == "false": value = False
                elif value.lower() == "none": value = None
            result[key] = value
        return result

    def parse_command(self, cmd: str):
        if not cmd:
            return [], None, {}
            
        parts = re.split(r'(?=\s-\w+)', cmd.strip())
        main_body = parts[0].strip()
        
        tokens = [x.strip() for x in main_body.split(" ") if x.strip()]
        if not tokens:
            return [], None, {}

        if tokens[-1].lower() in self._workers:
            func_key = tokens.pop().lower()
            params = [t.upper() for t in tokens]
        elif tokens[0].lower() in self._workers:
            func_key = tokens.pop(0).lower()
            params = [t.upper() for t in tokens]
        else:
            func_key = "default"
            params = [t.upper() for t in tokens]

        settings = {}
        for p in parts[1:]:
            match = re.search(r'-(\w+)\s*(.*)', p.strip())
            if match:
                flag_name = match.group(1).lower()
                flag_content = match.group(2).strip()
                arg_tokens = re.findall(r'\S+[:=]\S+|\S+', flag_content)
                settings[flag_name] = self.parse_args(arg_tokens)
                
        return params, func_key, settings

    def execute(self, prompt: str):
        params, func_key, settings = self.parse_command(prompt)
        target_func = self.get(func_key)
        
        if target_func:
            return target_func(*params, settings=settings, func=func_key.upper())
        return None, "NOT_FOUND"

def boot_system():
    managers = {}
    api_keys = load_apikeys()
    history_start = "1980-01-01"
    managers["dm"] = DatabaseManager()
    
    sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
    managers["pm"] = PriceManager(sources=sources, history_start=history_start)
    
    sources = {"yahooquery": None}
    managers["fm"] = FundamentalManager(sources=sources, history_start=history_start, pm=managers["pm"])

    managers["edgar"] = EdgarToolsAPI(0.5)
    return managers

managers = boot_system()
registry = UniversalRegistry()


# =============================================================================
# 2. WORKERS
# =============================================================================


@registry.register(["cls", "clear"])
def clear_logic(*args, **kwargs):
    return None, "CLEAR"

@registry.register("new")
def command_new_chat(*args, **kwargs):
    return None, "NEW_CHAT"

@registry.register("default")
def description_logic(*params, settings=None, **kwargs):
    msg = " ".join(params) if params else "No input provided."
    return html.Div(f"System Echo: {msg}"), "RENDER"

@registry.register("open")
def open_logic(*symbols, settings=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    edgar = managers["edgar"]
    try:
        for symbol in symbols:
            filings = edgar.fetch_filings(symbol, **f_args)
            edgar.open_filing(filings)
        return html.Div(f"Processed {', '.join(symbols)}"), "RENDER"
    except Exception as e:
        return html.Div(f"Error: {str(e)}"), "ERROR"

@registry.register(["fa", "is", "bs", "cf", "rv"])
def open_statement(*symbols, settings=None, func=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    cmd = str(func).upper()
    
    func_map = {"IS": ["income_statement"], "BS": ["balance_sheet"], "CF": ["cash_flow"], "RV": ["relative_valuation"]}
    all_domains = ["income_statement", "balance_sheet", "cash_flow", "relative_valuation"]
    domains = all_domains if cmd == "FA" else func_map.get(cmd, [])

    dm, fm = managers["dm"], managers["fm"]
    try:
        database = dm.fetch_database(asset_type="equities", filters={"currency": ['usd']}, COLLECT=False)
        database = dm.equity_filter(database, "equities", COLLECT=False)
        df_init, _ = dm.filter_excludes(database, "equities", filters={"asset_type": "equities", "source": "yahooquery"}, limit=None, COLLECT=True)
        df_init = df_init.filter(pl.col("symbol").is_in(list(symbols)))
        if df_init.is_empty(): return html.Div(f"Symbols {symbols} not found."), "ERROR"
        
        fa, _ = fm.load_fundamentals("equities", df_init, frequency=f_args.get("frequency", "annual"), refresh_threshold_days=None, REFRESH=False)
        if fa.is_empty(): return html.Div("No data found."), "ERROR"

        rv = fm._yq.get_relative_valuation(fa)
        dfs = fm._yq.get_tables(fa, rv, domains=domains)

        output_elements = []
        for symbol, data in dfs.items():
            for key, df_table in data.items():
                label_col = "Metric"
                df_table = df_table.rename({df_table.columns[0]: label_col})
                
                columnDefs = [{"field": label_col, "headerName": "", "pinned": "left"}]
                columnDefs += [{"field": col} for col in df_table.columns[1:]]

                grid = dag.AgGrid(
                    rowData=df_table.to_dicts(),
                    columnDefs=columnDefs,
                    className="ag-theme-alpine-dark",
                    columnSize="sizeToFit",
                    style={"height": "400px", "width": "100%", "marginBottom": "20px"}
                )
                
                output_elements.extend([
                    html.H5(f"{symbol} - {key.replace('_', ' ').title()}"), 
                    grid
                ])
        return html.Div([
            html.P(f"Processed {cmd} for: {', '.join(symbols)}"), 
            html.Div(output_elements)
        ]), "RENDER"
    except Exception as e:
        return html.Div(f"FA Error: {str(e)}"), "ERROR"

@registry.register(["p", "price"])
def worker_price(*symbols, settings=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    frequency = f_args.get("frequency", "daily")
    dm, pm = managers["dm"], managers["pm"]
    
    try:
        database = dm.fetch_database(asset_type="equities", filters={"currency": ['usd']}, COLLECT=False)
        database = dm.equity_filter(database, "equities", COLLECT=False)
        df_init, _ = dm.filter_excludes(database, "equities", filters={"asset_type": "equities", "source": "tiingo"}, limit=None, COLLECT=True)
        df_init = df_init.filter(pl.col("symbol").is_in(list(symbols)))
        
        if df_init.is_empty(): return html.Div(f"Symbols {symbols} not found."), "ERROR"

        df_history, _ = pm.load_history("equities", df_init, frequency, refresh_threshold_days=None, REFRESH=False)
        if df_history.is_empty(): return html.Div("No price history found."), "ERROR"

        output_elements = []
        for symbol in list(symbols):
            sym_df = df_history.filter(pl.col("symbol") == symbol.upper()).to_pandas()
            if sym_df.empty: continue
            
            fig = px.line(sym_df, x="date", y="adjClose", title=f"{symbol.upper()} - {frequency}", template="plotly_dark")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
            output_elements.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))

        return html.Div([html.P(f"Price Chart: {', '.join(symbols)}"), html.Div(output_elements)]), "RENDER"
    except Exception as e:
        return html.Div(f"Price Error: {str(e)}"), "ERROR"








    
