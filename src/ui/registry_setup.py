from src.utils.logg import log_info, log_df
from src.io.api import load_apikeys
from src.io.database import DatabaseManager
from src.io.fundamental import FundamentalManager
from src.io.price import PriceManager
from src.io.sec import EdgarToolsAPI
from src.io.news import fetch_financial_data

from dash import html, dcc
from dash import dcc, html, callback
import dash_bootstrap_components as dbc
import re
import dash_mantine_components as dmc
import dash_ag_grid as dag
import polars as pl
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from src.utils.df import transpose_df
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# 1. REGISTRY
# =============================================================================


class UniversalRegistry:    
    def __init__(self):
        # Stores mapping of name -> {"func": callable, "desc": str}
        self._workers = {}   

    def register(self, names, desc=""):
        """Decorator to register functions under one or multiple aliases."""
        if isinstance(names, str):
            names = [names]
        def decorator(fn):
            for name in names:
                self._workers[name.lower()] = {"func": fn, "desc": desc}
            return fn
        return decorator

    def get(self, name):
        worker = self._workers.get(name.lower())
        return worker["func"] if worker else None

    def parse_args(self, args_list: list) -> dict:
        """Parses list of strings into a dict, handling key=value"""
        result = {}
        for arg in args_list:
            parts = re.split(r'\s*[=]\s*', arg, 1)
            key = parts[0].lower()
            value = parts[1] if len(parts) > 1 else True
            
            if isinstance(value, str):
                if value.lower() == "true": value = True
                elif value.lower() == "false": value = False
                elif value.lower() == "none": value = None
            result[key] = value
        return result

    def parse_command(self, cmd: str):
        """Splits a command string into (params, function_key, settings)."""
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
        """Parses and executes the command, adding \\ to result if not found."""
        params, func_key, settings = self.parse_command(prompt)
        target_func = self.get(func_key)
        
        if target_func:
            return target_func(*params, settings=settings, func=func_key.upper())        
        return None, "NOT_FOUND\\", settings

def boot_system():
    managers = {}
    api_keys = load_apikeys()
    history_start = "1980-01-01"
    managers["dm"] = DatabaseManager()
    
    sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None), "yahooquery": None}
    managers["pm"] = PriceManager(sources=sources, history_start=history_start)
    
    sources = {"yahooquery": None}
    managers["fm"] = FundamentalManager(sources=sources, history_start=history_start)

    managers["edgar"] = EdgarToolsAPI(0.5)
    return managers

managers = boot_system()
registry = UniversalRegistry()


# =============================================================================
# 2. WORKERS
# =============================================================================


@registry.register(["cls", "clear"], desc="Deletes current chat")
def clear_logic(*args, **kwargs):
    return None, "CLEAR"

@registry.register(["new"], desc="Creates new chat")
def command_new_chat(*args, **kwargs):
    return None, "NEW_CHAT"

@registry.register(["default"], desc="Echos inputs")
def description_logic(*params, settings=None, **kwargs):
    msg = " ".join(params) if params else "No input provided."
    return html.Div(f"System Echo: {msg}"), "RENDER"

@registry.register(["help"], desc="Displays all available commands and their descriptions.")
def help_menu(*args, settings=None, func=None, **kwargs):
    from collections import defaultdict
    
    func_to_names = defaultdict(list)
    func_to_desc = {}
    
    for name, data in registry._workers.items():
        f_obj = data["func"]
        func_to_names[f_obj].append(name.upper())
        func_to_desc[f_obj] = data["desc"]

    rows = []
    for f_obj, names in func_to_names.items():
        alias_str = ", ".join(sorted(names))
        rows.append(
            html.Tr([
                html.Td(html.B(alias_str), style={
                    "padding": "16px", 
                    "borderBottom": "1px solid #666",
                    "color": "#FFFFFF"
                }),
                html.Td(func_to_desc[f_obj], style={
                    "padding": "16px", 
                    "borderBottom": "1px solid #666",
                    "color": "#E0E0E0"
                })
            ])
        )

    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("COMMAND", style={"padding": "16px", "textAlign": "left"}),
                html.Th("DESCRIPTION", style={"padding": "16px", "textAlign": "left"})
            ], style={"background": "#333", "color": "#FFFFFF", "textTransform": "uppercase", "letterSpacing": "1px"})
        ),
        html.Tbody(rows)
    ], style={
        "width": "100%", 
        "borderCollapse": "collapse",
        "fontFamily": "Inter, 'Segoe UI', Roboto, sans-serif",
        "fontSize": "16px",
        "background": "#4a4a4a"
    })

    return html.Div([table], style={"borderRadius": "4px", "overflow": "hidden"}), "RENDER"



@registry.register(["sec"], desc="Opens SEC filings")
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

@registry.register(["p"], desc="Displays stock prices")
def open_price(*symbols, settings=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    frequency = f_args.get("frequency", "daily")
    dm, pm, fm = managers["dm"], managers["pm"], managers["fm"]
    
    try:
        # 1. Data fetching logic
        database = dm.fetch_database(asset_type="equities")
        database = dm._reader.filter_lazy(database, filters={"symbol": list(symbols)})
        df_init = database.collect()
        if df_init.is_empty(): 
            return dmc.Alert(f"Symbols {symbols} not found.", color="red"), "ERROR"
        
        df_history, _ = pm.load_history("equities", df_init, frequency, refresh_threshold_days=2, REFRESH=False)
        if df_history.is_empty(): 
            return dmc.Alert("No price history found.", color="yellow"), "ERROR"

        today = datetime.now()
        live = fm._yq._fetch_live_data(list(df_init["symbol"].unique(maintain_order=True)))
        live = live.with_columns(pl.lit(today).dt.date().alias("date"))
        live = live.with_columns(pl.col("Price").alias("adjClose"))
        df_history = pl.concat([df_history.select(["symbol", "date", "adjClose"]), live.select(["symbol", "date", "adjClose"])])
        
        periods = {
            "1D": "SPECIAL_1D",
            "5D": today - timedelta(days=5),
            "1M": today - timedelta(days=30),
            "3M": today - timedelta(days=90),
            "6M": today - timedelta(days=180),
            "YTD": datetime(today.year, 1, 1),
            "1Y": today - timedelta(days=365),
            "5Y": today - timedelta(days=365 * 5),
            "MAX": None
        }

        symbol_tabs_list, symbol_panels = [], []
        for symbol in list(symbols):
            sym_df = df_history.filter(pl.col("symbol") == symbol.upper()).to_pandas()
            if sym_df.empty: continue
            sym_df['date'] = pd.to_datetime(sym_df['date'])
            
            tf_tabs_list, tf_panels = [], []
            for label, start_date in periods.items():
                if label == "1D":
                    filtered_df = sym_df.tail(2)
                else:
                    filtered_df = sym_df[sym_df['date'] >= start_date] if start_date else sym_df
                
                fig = go.Figure()

                if filtered_df.empty or (label == "1D" and len(filtered_df) < 2):
                    fig_content = dmc.Center(dmc.Text("No data.", c="dimmed"), h=400)
                elif label == "1D":
                    # --- 1D DASHBOARD (JITTER-LOCKED) ---
                    curr_p, prev_p = filtered_df['adjClose'].iloc[-1], filtered_df['adjClose'].iloc[-2]
                    abs_diff = curr_p - prev_p
                    pct_diff = (abs_diff / prev_p) * 100
                    perf_color = "#32CD32" if abs_diff >= 0 else "#FF6347"

                    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.20, y0=0.25, x1=0.80, y1=0.75,
                                  line=dict(color=perf_color, width=2), fillcolor="rgba(0,0,0,0)")

                    fig.add_annotation(x=0.48, y=0.5, xref="paper", yref="paper", text=f"<b>{symbol.upper()}</b>",
                                       showarrow=False, font=dict(size=60, color="white"), xanchor="right", yanchor="middle")
                    
                    fig.add_annotation(x=0.52, y=0.5, xref="paper", yref="paper", text=f"<b>${curr_p:,.2f}</b>",
                                       showarrow=False, font=dict(size=35, color=perf_color), xanchor="left", yanchor="bottom")
                    
                    # Added timeframe sub-label here
                    fig.add_annotation(x=0.52, y=0.5, xref="paper", yref="paper", 
                                       text=f"{abs_diff:+,.2f} ({pct_diff:+,.2f}%)<br><span style='font-size:12px; color:gray;'>Today</span>",
                                       showarrow=False, font=dict(size=20, color=perf_color), xanchor="left", yanchor="top")

                    fig.update_layout(
                        plot_bgcolor="#000000", paper_bgcolor="#000000",
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1], automargin=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1], automargin=False),
                        height=400, autosize=False
                    )
                else:
                    # --- LINE PLOTS (DUAL Y-AXIS + JITTER-LOCKED) ---
                    prices = filtered_df['adjClose']
                    start_p, end_p = prices.iloc[0], prices.iloc[-1]
                    p_min, p_max = prices.min(), prices.max()
                    pct_min, pct_max = ((p_min - start_p) / start_p) * 100, ((p_max - start_p) / start_p) * 100
                    perf_color = "#32CD32" if (end_p - start_p) >= 0 else "#FF6347"

                    # Determine descriptive timeframe label
                    tf_desc = f"Past {label}" if label != "YTD" else "Year to Date"
                    if label == "MAX": tf_desc = "All Time"

                    fig.add_trace(go.Scatter(x=filtered_df['date'], y=prices, mode='lines', yaxis="y1",
                                             line=dict(color=perf_color, width=2), fill='tozeroy', 
                                             fillcolor=f"rgba({ '50,205,50' if (end_p - start_p) >= 0 else '255,99,71'}, 0.15)"))
                    
                    fig.add_trace(go.Scatter(x=filtered_df['date'], y=((prices - start_p) / start_p) * 100,
                                             mode='lines', line=dict(color='rgba(0,0,0,0)'), 
                                             yaxis="y2", hoverinfo='skip', showlegend=False))

                    # Top-Left Label with Price, Change, and Timeframe
                    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper", showarrow=False, align="left",
                                       text=(
                                           f"<b>${end_p:,.2f}</b><br>"
                                           f"<span style='color:{perf_color}; font-size:14px;'>{(end_p-start_p):+,.2f} ({(end_p-start_p)/start_p*100:.2f}%)</span><br>"
                                           f"<span style='color:white; font-size:11px;'>{tf_desc}</span>"
                                       ),
                                       xanchor="left", yanchor="top", font=dict(size=18, color="white"))

                    fig.update_layout(
                        template="plotly_dark", 
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=80, t=10, b=30), 
                        height=400, 
                        autosize=False, 
                        hovermode="x unified", 
                        showlegend=False,
                        yaxis=dict(
                            range=[p_min*0.98, p_max*1.02], 
                            fixedrange=True, side="left", gridcolor="#333", 
                            showgrid=True, zeroline=False, automargin=True, tickprefix="$"
                        ),
                        yaxis2=dict(
                            range=[pct_min-2, pct_max+2], 
                            fixedrange=True, side="right", overlaying="y", 
                            showgrid=False, zeroline=False, automargin=True, 
                            tickformat="+.1f", ticksuffix="%"
                        ),
                        xaxis=dict(
                            tickformat="%b %y" if label in ["5Y", "MAX"] else "%b %d", 
                            gridcolor="#333", showgrid=True, automargin=True
                        )
                    )

                fig_content = dcc.Graph(figure=fig, config={'displayModeBar': False, 'responsive': False}, style={"width": "100%"})
                tf_tabs_list.append(dmc.TabsTab(label, value=label))
                tf_panels.append(dmc.TabsPanel(fig_content, value=label, pt="sm"))

            symbol_tabs_list.append(dmc.TabsTab(symbol.upper(), value=symbol))
            symbol_panels.append(dmc.TabsPanel(dmc.Tabs([dmc.TabsList(tf_tabs_list), *tf_panels], value="1D"), value=symbol, pt="md"))
            
        return dmc.Tabs([dmc.TabsList(symbol_tabs_list), *symbol_panels], value=list(symbols)[0],
                        variant="pills", styles={"root": {"backgroundColor": "#000000"}}), "RENDER"
                        
    except Exception as e:
        return dmc.Alert(f"Price Error: {str(e)}", color="red"), "ERROR"
    
@registry.register(["fa", "is", "bs", "cf"], desc="Displays financial statements")
def open_statement(*symbols, settings=None, func=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    cmd = str(func).upper()
    
    func_map = {"IS": ["income_statement"], "BS": ["balance_sheet"], "CF": ["cash_flow"]}
    all_domains = ["income_statement", "balance_sheet", "cash_flow"]
    domains = all_domains if cmd == "FA" else func_map.get(cmd, [])
    dm, fm = managers["dm"], managers["fm"]
    
    # Outer Tabs (Symbols)
    symbol_styles = {
        "root": {"backgroundColor": "#000000 !important"},
        "tabsList": {
            "backgroundColor": "#ADB5BD !important",
            "width": "fit-content", 
            "borderRadius": "4px",
            "border": "1px solid #333333",
            "padding": "2px"
        },
        "tab": {
            "color": "#000000 !important", 
            "fontWeight": "400 !important", 
            "fontSize": "15px",
            "&[data-active]": {"backgroundColor": "#228BE6", "color": "#FFFFFF !important"}
        }
    }

    # Inner Tabs (Domains)
    domain_styles = {
        "root": {"backgroundColor": "#000000 !important"},
        "tabsList": {
            "backgroundColor": "#DEE2E6 !important", 
            "width": "fit-content !important",
            "border": "1px solid #000000 !important",
            "borderRadius": "4px",
            "padding": "2px",
            "marginBottom": "4px"
        },
        "tab": {
            "color": "#000000 !important", 
            "fontWeight": "400 !important", 
            "fontSize": "15px",
            "&[data-active]": {
                "color": "#228BE6 !important", 
                "backgroundColor": "#FFFFFF",
                "borderBottom": "2px solid #228BE6"
            }
        }
    }

    try:
        database = dm.fetch_database(asset_type="equities")
        database = dm._reader.filter_lazy(database, filters={"symbol": list(symbols)})
        df_init = database.collect()
        if df_init.is_empty():
            return dmc.Alert("No symbols found.", color="red"), "ERROR"
        
        fa, _ = fm.load_fundamentals("equities", df_init, frequency=f_args.get("frequency", "annual"), REFRESH=False)
        dfs, exp, highlights = fm._yq.get_tables(fa, domains=domains)
        
        UNITS = {12: "USD (in trillions)", 9: "USD (in billions)", 6: "USD (in millions)", 3: "USD (in thousands)", 0: "USD"}
        unit_label = UNITS.get(exp, "")
        js_highlights = str(list(highlights))

        symbol_tabs_list, symbol_panels = [], []
        for symbol, data in dfs.items():
            domain_tabs_list, domain_panels = [], []
            
            for domain_key, df_table in data.items():
                df_table = fm._yq._dis.format_display(df_table)
                df_table = transpose_df(df_table, "date") 
                label_col = f"{unit_label}" if unit_label and domain_key!="relative_valuation" else "metrics"
                df_table = df_table.rename({df_table.columns[0]: label_col})
                
                header_style = {
                    "backgroundColor": "#F8F9FA", 
                    "color": "#868E96 !important", 
                    "fontWeight": "300 !important", 
                    "fontSize": "14px"
                }
                
                column_defs = [{
                    "field": label_col, "pinned": "left", "width": 240,
                    "headerStyle": header_style,
                    "cellStyle": {"fontSize": "14px", "color": "#000000 !important", "backgroundColor": "inherit", "fontWeight": "inherit", "padding": "0 8px"}
                }]
                
                for col in df_table.columns[1:]:
                    column_defs.append({
                        "field": col, 
                        "flex": 1, 
                        "minWidth": 110,
                        "headerClass": "ag-right-aligned-header",
                        "headerStyle": {**header_style, "textAlign": "right"},
                        "cellStyle": {"textAlign": "right", "fontFamily": "monospace", "fontSize": "14px", "color": "#000000 !important", "backgroundColor": "inherit", "fontWeight": "inherit", "padding": "0 8px"}
                    })

                grid = dag.AgGrid(
                    rowData=df_table.to_dicts(),
                    columnDefs=column_defs,
                    className="ag-theme-alpine", 
                    style={"width": "100%", "backgroundColor": "#000000"},
                    dashGridOptions={
                        "domLayout": "autoHeight",
                        "headerHeight": 32,
                        "rowHeight": 28,
                        "animateRows": False,
                        "suppressCellFocus": True,
                        "suppressHorizontalScroll": True,
                        "getRowStyle": {
                            "styleConditions": [
                                # 1. BOLD + WHITE BACKGROUND (Even rows)
                                {
                                    "condition": f"{js_highlights}.includes(params.data['{label_col}']) && params.node.rowIndex % 2 === 0", 
                                    "style": {"fontWeight": "700", "backgroundColor": "#FFFFFF"}
                                },
                                # 2. BOLD + GRAY BACKGROUND (Odd rows)
                                {
                                    "condition": f"{js_highlights}.includes(params.data['{label_col}']) && params.node.rowIndex % 2 !== 0", 
                                    "style": {"fontWeight": "700", "backgroundColor": "#F2F2F2"}
                                },
                                # 3. NORMAL + WHITE BACKGROUND
                                {
                                    "condition": "params.node.rowIndex % 2 === 0", 
                                    "style": {"backgroundColor": "#FFFFFF"}
                                },
                                # 4. NORMAL + GRAY BACKGROUND
                                {
                                    "condition": "params.node.rowIndex % 2 !== 0", 
                                    "style": {"backgroundColor": "#F2F2F2"}
                                }
                            ]
                        }
                    }
                )
                
                tab_display_name = f"{domain_key.replace('_', ' ').title()}".strip()
                domain_tabs_list.append(dmc.TabsTab(tab_display_name, value=domain_key))
                domain_panels.append(dmc.TabsPanel(grid, value=domain_key, style={"backgroundColor": "#000000"}))

            inner_tabs = dmc.Tabs([dmc.TabsList(domain_tabs_list), *domain_panels], variant="default", value=list(data.keys())[0], styles=domain_styles)
            symbol_tabs_list.append(dmc.TabsTab(symbol, value=symbol))
            symbol_panels.append(dmc.TabsPanel(inner_tabs, value=symbol, pt="xs", style={"backgroundColor": "#000000"}))
        return dmc.Tabs([dmc.TabsList(symbol_tabs_list), *symbol_panels], variant="pills", value=list(dfs.keys())[0], styles=symbol_styles, p="md"), "RENDER"
    except Exception as e:
        return dmc.Alert(f"Error: {str(e)}", color="red"), "ERROR"

@registry.register(["save"], desc="Creates and saves an excel model")
def save_excel(*symbols, settings=None, func=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    cmd = str(func).upper()
    
    dm, fm = managers["dm"], managers["fm"]
    domains = ["income_statement", "balance_sheet", "cash_flow", "relative_valuation"]
    try:
        database = dm.fetch_database(asset_type="equities")
        database = dm._reader.filter_lazy(database, filters={"symbol": list(symbols)})
        df_init = database.collect()
        if df_init.is_empty():
            return dmc.Alert("No symbols found.", color="red"), "ERROR"
        annual, _ = fm.load_fundamentals("equities", df_init, frequency="annual", REFRESH=False)
        quarterly, _ = fm.load_fundamentals("equities", df_init, frequency="quarterly", REFRESH=False)
        fm._yq.save_FA(annual, quarterly)
        return html.Div(f"Processed {', '.join(symbols)}"), "RENDER"
    except Exception as e:
        return dmc.Alert(f"Error: {str(e)}", color="red"), "ERROR"

@registry.register(["open"], desc="Opens an excel model")
def open_excel(*symbols, settings=None, func=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    fm = managers["fm"]
    basepath = Path.cwd() / "data" / "output" / "fa_models"
    
    processed_symbols = []
    
    for symbol in symbols:
        file_path = basepath / f"{symbol}.xlsx"
        
        if not file_path.exists():
            save_func = registry.get("save") 
            if save_func:
                res, status = save_func(symbol, settings=settings, func="save", **kwargs)
                if status == "ERROR":
                    return res, "ERROR"
            else:
                return dmc.Alert("The 'save' command is not registered.", color="red"), "ERROR"

        try:
            if file_path.exists():
                fm._yq._excel.open_file(file_path)
                processed_symbols.append(symbol)
            else:
                return dmc.Alert(f"File {symbol}.xlsx not found and could not be generated.", color="red"), "ERROR"
        except Exception as e:
            return dmc.Alert(f"Error opening {symbol}: {str(e)}", color="red"), "ERROR"
    return html.Div(f"Opened: {', '.join(processed_symbols)}"), "RENDER"

@registry.register(["n"], desc="News Feed")
def news_logic(*symbols, settings=None, **kwargs):
    limit = 100
    df = fetch_financial_data(symbols, limit=limit)
    instance_id = str(uuid.uuid4())
    
    if df.is_empty():
        rows = [html.Tr(html.Td("NO DATA", colSpan=4, style={'color': '#fff', 'padding': '20px'}))]
    else:
        rows = [
            html.Tr([
                html.Td(f"{r['date']}", style={'width': '110px', 'padding': '6px 10px'}),
                html.Td(f"{r['symbol']}", style={'width': '80px', 'fontWeight': 'bold'}),
                html.Td(r['headline'], style={'wordWrap': 'break-word', 'whiteSpace': 'normal'}),
                html.Td(f"[{r.get('source', '').upper()}]", style={'width': '120px', 'textAlign': 'right', 'paddingRight': '10px'})
            ], 
            className="n-row", tabIndex=0, **{"data-url": r.get('link') or ""},
            style={'color': '#fff', 'fontSize': '13px', 'borderBottom': '1px solid #333', 'outline': 'none'}
            ) for r in df.to_dicts()
        ]

    return html.Div([
        html.Div(id={'type': 'n-container', 'index': instance_id}, style={
            'backgroundColor': '#1e1e1e', 'height': '55vh', 'overflowY': 'auto',
            'outline': 'none', 'position': 'relative', 'fontFamily': 'monospace'
        }, children=[
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("LATEST ↓", style={'width': '110px', 'color': '#888', 'borderBottom': '1px solid #444', 'textAlign': 'left', 'padding': '10px'}),
                        html.Th("SYM", style={'width': '80px', 'color': '#888', 'borderBottom': '1px solid #444', 'textAlign': 'left', 'padding': '10px'}),
                        html.Th("HEADLINE", style={'color': '#888', 'borderBottom': '1px solid #444', 'textAlign': 'left', 'padding': '10px'}),
                        html.Th("SOURCE", style={'width': '120px', 'color': '#888', 'borderBottom': '1px solid #444', 'textAlign': 'right', 'padding': '10px'}),
                    ], style={'position': 'sticky', 'top': 0, 'backgroundColor': '#1e1e1e', 'zIndex': 10})
                ]),
                html.Tbody(rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'tableLayout': 'fixed'})
        ]),
        # Unique trigger for the clientside callback
        html.Div(id={'type': 'n-nav-init', 'index': instance_id}, style={'display': 'none'})
    ]), "RENDER"

@registry.register(["txt"], desc="Opens SEC sections")
def open_sections(*symbols, settings=None, **kwargs):
    f_args = (settings or {}).get("f", {})
    edgar = managers["edgar"]
    
    # Generate one unique ID for this execution
    uid = str(uuid.uuid4())[:8]
    
    BG_DARK = "#000000"
    TEXTBOX_GRAY = "#4A4A4A"
    ACCENT_BLUE = "#228BE6"

    symbol_styles = {
        "root": {"backgroundColor": f"{BG_DARK} !important"},
        "tabsList": {"backgroundColor": "#ADB5BD !important", "width": "fit-content", "borderRadius": "4px", "padding": "2px"},
        "tab": {
            "color": "#000000 !important", "fontWeight": "500 !important", "fontSize": "15px",
            "&[data-active]": {"backgroundColor": ACCENT_BLUE, "color": "#FFFFFF !important"}
        }
    }

    section_styles = {
        "root": {"backgroundColor": f"{BG_DARK} !important"},
        "tabsList": {"backgroundColor": "#DEE2E6 !important", "width": "fit-content !important", "borderRadius": "4px", "padding": "2px", "marginBottom": "12px"},
        "tab": {
            "color": "#000000 !important", "fontSize": "14px",
            "&[data-active]": {"color": ACCENT_BLUE, "backgroundColor": "#FFFFFF", "borderBottom": f"3px solid {ACCENT_BLUE}"}
        }
    }

    def get_symbol_data(symbol):
        try:
            filings = edgar.fetch_filings(symbol, ["10-K", "20-F"], **f_args)
            df = edgar.process_obj(symbol, filings, limit=1).sort("date", descending=False)
            if df.is_empty(): return None
            return symbol, df.tail(1).select(['business', 'risk_factors', 'management_discussion'])
        except: return None

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(get_symbol_data, symbols))

    symbol_tabs_list, symbol_panels = [], []
    valid_tickers = []

    for res in results:
        if res is None: continue
        symbol, sections = res
        valid_tickers.append(symbol)
        
        domain_tabs_list, domain_panels = [], []
        for section_name in sections.columns:
            display_content = dmc.ScrollArea(
                offsetScrollbars=True,
                # CLASSNAMES INSTEAD OF IDS
                className=f"scroll-area-{uid}",
                style={"height": 700, "backgroundColor": TEXTBOX_GRAY, "borderRadius": "12px", "border": "1px solid #555555"},
                children=[
                    html.Div(
                        str(sections[section_name][0]), 
                        className=f"text-div-{uid}",
                        style={
                            "color": "#FFFFFF", "whiteSpace": "pre-wrap", "padding": "50px 80px",
                            "fontFamily": "'Georgia', serif", "fontSize": "18px", "lineHeight": "1.8",
                            "maxWidth": "1000px", "margin": "0 auto"
                        }
                    )
                ]
            )
            domain_tabs_list.append(dmc.TabsTab(section_name.replace("_", " ").title(), value=section_name))
            domain_panels.append(dmc.TabsPanel(display_content, value=section_name, pt="md"))

        symbol_tabs_list.append(dmc.TabsTab(symbol, value=symbol))
        symbol_panels.append(dmc.TabsPanel(
            dmc.Tabs([dmc.TabsList(domain_tabs_list), *domain_panels], variant="default", value=sections.columns[0], styles=section_styles), 
            value=symbol, pt="xs"
        ))

    if not symbol_tabs_list:
        return dmc.Alert("No filings found.", color="red"), "ERROR"

    return html.Div([
        dmc.Group(
            justify="flex-end", mb="sm",
            children=[
                dmc.Text("Light Mode", id=f"label-{uid}", c="white", size="xs"),
                dmc.Switch(id={"type": "sec-toggle", "index": uid}, checked=False)
            ]
        ),
        dmc.Tabs(
            [dmc.TabsList(symbol_tabs_list), *symbol_panels], 
            variant="pills", value=valid_tickers[0], styles=symbol_styles, p="md"
        ),
        dcc.Store(id={"type": "sec-store", "index": uid})
    ]), "RENDER"


@registry.register(["rec"], desc="Displays live market recap")
def open_recap(settings=None, **kwargs):
    fm = managers["fm"]
    FILTERS = ['most_actives', 'day_gainers', 'day_losers']
    
    # Matching your symbol_styles exactly
    symbol_styles = {
        "root": {"backgroundColor": "#000000 !important"},
        "tabsList": {
            "backgroundColor": "#ADB5BD !important",
            "width": "fit-content", 
            "borderRadius": "4px",
            "border": "1px solid #333333",
            "padding": "2px"
        },
        "tab": {
            "color": "#000000 !important", 
            "fontWeight": "400 !important", 
            "fontSize": "15px",
            "&[data-active]": {"backgroundColor": "#228BE6", "color": "#FFFFFF !important"}
        }
    }

    header_style = {
        "backgroundColor": "#F8F9FA", 
        "color": "#868E96 !important", 
        "fontWeight": "300 !important", 
        "fontSize": "14px"
    }

    try:
        def fetch_single_filter(f_name):
            return fm._yq.fetch_screeners(filters=[f_name])

        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            thread_results = list(executor.map(fetch_single_filter, FILTERS))
            for res in thread_results:
                results.update(res)

        if not results:
            return dmc.Alert("No data returned.", color="red"), "ERROR"

        symbol_tabs_list, symbol_panels = [], []
        
        for title, df in results.items():
            column_defs = [
                {
                    "field": "symbol", "headerName": "Symbol", "pinned": "left", "width": 240,
                    "headerStyle": header_style,
                    "cellStyle": {"fontSize": "14px", "backgroundColor": "inherit", "padding": "0 8px"}
                },
                {
                    "field": "price", "headerName": "Price", "flex": 1,
                    "headerClass": "ag-right-aligned-header",
                    "headerStyle": {**header_style, "textAlign": "right"},
                    "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"},
                    "cellStyle": {"textAlign": "right", "fontFamily": "monospace", "fontSize": "14px", "backgroundColor": "inherit", "padding": "0 8px"}
                },
                {
                    "field": "pctChange", "headerName": "% Change", "flex": 1,
                    "headerClass": "ag-right-aligned-header",
                    "headerStyle": {**header_style, "textAlign": "right"},
                    "valueFormatter": {"function": "params.value.toFixed(2) + '%'"},
                    "cellStyle": {"textAlign": "right", "fontFamily": "monospace", "fontSize": "14px", "backgroundColor": "inherit", "padding": "0 8px"}
                }
            ]

            grid = dag.AgGrid(
                rowData=df.to_dicts(),
                columnDefs=column_defs,
                className="ag-theme-alpine", 
                style={"width": "100%", "backgroundColor": "#000000"}, 
                dashGridOptions={
                    "domLayout": "autoHeight",
                    "headerHeight": 32,
                    "rowHeight": 28,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "suppressHorizontalScroll": True,
                    "getRowStyle": {
                        "styleConditions": [
                            # POSITIVE: Green Text + open_statement Zebra Backgrounds
                            {"condition": "params.data.pctChange > 0 && params.node.rowIndex % 2 === 0", "style": {"color": "#32CD32", "backgroundColor": "#FFFFFF"}},
                            {"condition": "params.data.pctChange > 0 && params.node.rowIndex % 2 !== 0", "style": {"color": "#32CD32", "backgroundColor": "#F2F2F2"}},
                            
                            # NEGATIVE: Red Text + open_statement Zebra Backgrounds
                            {"condition": "params.data.pctChange < 0 && params.node.rowIndex % 2 === 0", "style": {"color": "#FF6347", "backgroundColor": "#FFFFFF"}},
                            {"condition": "params.data.pctChange < 0 && params.node.rowIndex % 2 !== 0", "style": {"color": "#FF6347", "backgroundColor": "#F2F2F2"}},
                            
                            # NEUTRAL: Black Text + open_statement Zebra Backgrounds
                            {"condition": "params.node.rowIndex % 2 === 0", "style": {"backgroundColor": "#FFFFFF", "color": "#000000"}},
                            {"condition": "params.node.rowIndex % 2 !== 0", "style": {"backgroundColor": "#F2F2F2", "color": "#000000"}}
                        ]
                    }
                }
            )

            display_title = title.replace('_', ' ').title()
            symbol_tabs_list.append(dmc.TabsTab(display_title, value=title))
            symbol_panels.append(dmc.TabsPanel(grid, value=title, pt="xs", style={"backgroundColor": "#000000"}))

        return dmc.Tabs(
            [dmc.TabsList(symbol_tabs_list), *symbol_panels], 
            variant="pills", 
            value=list(results.keys())[0], 
            styles=symbol_styles, 
            p="md",
            style={"backgroundColor": "#000000"}
        ), "RENDER"

    except Exception as e:
        return dmc.Alert(f"Recap Error: {str(e)}", color="red"), "ERROR"
