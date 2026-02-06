from src.utils.logg import log_info, log_df
from src.io.fetch import load_apikeys
from src.io.manager import ManagerManager
from src.io.parquet import ParquetManager
from src.io.database import DatabaseManager
from src.io.fundamental import FundamentalManager
from src.io.price import PriceManager
from src.io.sec import EdgarToolsAPI
from dash import html
import re
import polars as pl
import dash_ag_grid as dag # Ensure you have this installed


# =============================================================================
# 1. REGISTRY
# =============================================================================


class UniversalRegistry:
    def __init__(self):
        # Renamed to _workers to match your request
        self._workers = {}

    def register(self, names):
        """Allows registering multiple keywords for one function."""
        if isinstance(names, str):
            names = [names]
        def decorator(fn):
            for name in names:
                self._workers[name.lower()] = fn
            return fn
        return decorator

    def get(self, name):
        return self._workers.get(name.lower())

    def parse_args(self, args) -> dict:
        result = {}
        for arg in args:
            # Splits on '=' or ':'
            parts = re.split(r'\s*[=:]\s*', arg, 1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else None
            if value == "None":
                value = None
            result[key] = value
        return result

    def parse_command(self, cmd: str):
        if not cmd:
            return None, None, None
            
        cmd = cmd.lower()
        parts = re.split(r'(?=-\w+)', cmd.strip())
        main_body = parts[0].strip().replace("_", "-")
        
        tokens = [x.strip().upper() for x in main_body.split(" ") if x.strip()]
        
        if not tokens:
            return None, None, None

        if tokens[-1].lower() in self._workers:
            func_key = tokens.pop().lower()
            params = tokens
        else:
            func_key = "default" 
            params = tokens

        settings = {}
        for p in parts[1:]:
            matches = re.findall(r'-(\w+)\s+([^-]*)', p.strip())
            if matches:
                flag_name = matches[0][0]
                flag_content = matches[0][1].strip()                
                arg_tokens = re.findall(r'\S+\s*=\s*\S+|\S+', flag_content)
                settings[flag_name] = self.parse_args(arg_tokens)
                
        return params, func_key, settings

    def execute(self, prompt: str):
        params, func_key, settings = self.parse_command(prompt)
        
        if not func_key:
            return None, "EMPTY"
        
        target_func = self.get(func_key)
        
        if target_func:
            # Note: params are UPPERCASE list, settings is dict of dicts
            return target_func(*params, settings=settings)
        
        return None, "NOT_FOUND"

def boot_system():
    managers = {}
    io = ParquetManager() 
    mm = ManagerManager(io)
    managers["dm"] = DatabaseManager(io)
    api_keys = load_apikeys()
    history_start = "1980-01-01"
    
    sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
    managers["pm"] = PriceManager(mm=mm, sources=sources, history_start=history_start)
    
    sources = {"yahooquery": None}
    managers["fm"] = FundamentalManager(mm=mm, sources=sources, history_start=history_start, pm=managers["pm"])

    managers["edgar"] = EdgarToolsAPI(0.5)
    return managers

managers = boot_system()
registry = UniversalRegistry()


# =============================================================================
# 2. WORKERS
# =============================================================================


"""
@registry.register("open")
def open_filing():
   
"""

@registry.register("cls")
@registry.register("clear")
def clear_logic(*args, **kwargs):
    return None, "CLEAR"

@registry.register("new")
def command_new_chat(*args, **kwargs):
    return None, "NEW_CHAT"

@registry.register("default")
def description_logic(*params, settings=None):
    return html.Div(f"System Echo: {' '.join(params)}"), "RENDER"

@registry.register("open")
def open_logic(*symbols, settings=None):
    # If settings is None or 'f' key doesn't exist, f_args is an empty dict
    f_args = (settings or {}).get("f", {})
    
    edgar = managers["edgar"]
    try:
        # **f_args unpacks keys like 'form' or 'start' directly into the method
        for symbol in list(symbols):
            filings = edgar.fetch_filings(symbol, **f_args)
            edgar.open_filing(filings)
        return html.Div(f"Processed {', '.join(symbols)}"), "RENDER"
    except Exception as e:
        return html.Div(f"Error: {str(e)}"), "ERROR"

@registry.register("fa")
def open_statement(*symbols, settings=None):
    params = {"frequency": "annual"}
    
    if settings and "f" in settings:
        params.update(settings["f"])
    
    frequency = params["frequency"]
    dm = managers["dm"]
    fm = managers["fm"]
    
    database = dm.fetch_database(
        asset_type="equities", 
        filters={"currency": ['usd']}, 
        COLLECT=False
    )    
    
    df, _ = dm.filter_excludes(
        database, 
        "equities", 
        filters={"asset_type": "equities"}, 
        COLLECT=True
    )
    
    target_symbols = [s.upper() for s in symbols]
    df = df.filter(pl.col("symbol").is_in(target_symbols))
    
    if df.is_empty():
        return html.Div(f"Symbols {target_symbols} not found in database."), "ERROR"

    df, failed_df = fm.load_fundamentals(
        "equities", 
        df, 
        frequency=frequency, 
        REFRESH=False
    )

    if df.is_empty():
        return html.Div("No fundamental data records found."), "ERROR"

    df_display = df.transpose(include_header=True, header_name="Metric")
    
    grid = dag.AgGrid(
        rowData=df_display.to_dicts(),
        columnDefs=[{"field": i} for i in df_display.columns],
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        columnSize="autoSize",
        className="ag-theme-alpine-dark",
        style={"height": "400px", "width": "100%", "marginTop": "10px"}
    )

    return html.Div([
        html.P(f"Loaded {frequency} fundamentals for {', '.join(target_symbols)}:"),
        grid
    ]), "RENDER"
