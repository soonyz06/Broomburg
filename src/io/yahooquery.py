from pathlib import Path
import polars as pl
import pandas as pd
import time
import datetime
import numpy as np
import re
import json
from dateutil.relativedelta import relativedelta
from xlsxwriter.utility import xl_rowcol_to_cell
import os
import subprocess
import platform

from yahooquery import Ticker, Screener

from src.utils.df import replace_metric, max_metric, transpose_df
from src.feature.cross_section import MetricProcessor


class ExcelManager:
    def __init__(self):
        basepath = Path.cwd() / "config" / "fa_config"
        basepath.mkdir(parents=True, exist_ok=True)
        filepath = basepath / f"excel.json"

        with open(filepath, "r") as f:
            self.formulae = json.load(f)
        
    def format_table(self, df, frequency):
        def fmt(x, mode=",", mult=1):
            if x is None:
                return None # Fill null handles this later
            if isinstance(x, (int, float)):
                x = x * mult
                if mode == ",":
                    return f"{int(x):,}"
                elif mode == "%":
                    return f"{x*100:,.2f}%"
                elif mode == "m":
                    return f"{x:,.1f}"
            return str(x)

        q_map = {
            "Q1": [3, 4, 5],
            "Q2": [6, 7, 8],
            "Q3": [9, 10, 11],
            "Q4": [12, 1, 2]
        }

        max_q4 = q_map["Q4"][-1]

        df = df.with_columns([
            pl.col("date").dt.month().alias("_m"),
            pl.col("date").dt.year().alias("_y")
        ])

        df = df.with_columns(
            pl.when(pl.col("_m") <= max_q4)
            .then(pl.col("_y") - 1)
            .otherwise(pl.col("_y"))
            .alias("_fy")
        )

        if frequency == "a":
            df = df.with_columns(
                pl.col("_fy").cast(pl.Utf8).alias("date")
            )
        elif frequency == "q":
            df = df.with_columns(
                pl.when(pl.col("_m").is_in(q_map["Q1"])).then(pl.lit("Q1"))
                .when(pl.col("_m").is_in(q_map["Q2"])).then(pl.lit("Q2"))
                .when(pl.col("_m").is_in(q_map["Q3"])).then(pl.lit("Q3"))
                .otherwise(pl.lit("Q4")).alias("_q")
            )
            df = df.with_columns(
                (pl.lit("FY") + pl.col("_fy").cast(pl.Utf8).str.slice(-2) + pl.col("_q")).alias("date")
            )
        df = df.drop(["_m", "_y", "_fy", "_q"], strict=False)
        df = df.fill_null(0)
                    
        float_cols = [c for c, dtype in df.schema.items() if c != "date" and dtype == pl.Float64]
        int_cols = [c for c, dtype in df.schema.items() if c != "date" and dtype == pl.Int64]
        df = df.with_columns(
            [pl.col(c).map_elements(lambda x: fmt(x, "%"), return_dtype=pl.Utf8).alias(c) for c in float_cols] +
            [pl.col(c).map_elements(lambda x: fmt(x, ","), return_dtype=pl.Utf8).alias(c) for c in int_cols]
        )
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in df.columns]).fill_null("0")
        return df
    
    def _format_data(self, df):
        if df is None: return []
        df_str, formatted, prev_int = df.astype(str), [], False
        
        for i, row in enumerate(df.values):
            v_str = df_str.iloc[i, 1] if df_str.shape[1] > 1 else ""
            
            if "%" in v_str and prev_int: 
                formatted.append([None] * len(row))
                formatted.append([None] * len(row))
            
            new_row = []
            for j, cell in enumerate(row):
                if j == 0: 
                    new_row.append(cell)
                    continue
                if pd.isna(cell) or cell == "" or cell is None:
                    new_row.append(None)
                    continue
                
                s = str(cell).replace(",", "")
                if "%" in s:
                    new_row.append(float(s.replace("%", "")) / 100)
                else:
                    try: 
                        new_row.append(float(s))
                    except: 
                        new_row.append(cell)
            
            formatted.append(new_row)
            # Update prev_int: True if row has data and no '%' was detected
            has_data = not all(x is None for x in new_row[1:])
            if has_data:
                prev_int = "%" not in v_str
        return formatted

    def _apply_formula(self, label, cmp, row_map, ci, mults, fmts):
        m_fmt, pct_fmt, num_fmt = fmts['m'], fmts['pct'], fmts['num']
        use_fmt = m_fmt if label in mults else None
        
        if cmp[0] == "/" and len(cmp) >= 3:
            n, d = cmp[1], cmp[2]
            if n in row_map and d in row_map:
                f = f"={xl_rowcol_to_cell(row_map[n], ci)}/{xl_rowcol_to_cell(row_map[d], ci)}"
                return f, use_fmt or pct_fmt
        elif cmp[0] == "D" and len(cmp) >= 2:
            t = cmp[1]
            if t in row_map:
                if ci > 2:
                    cr, pr = xl_rowcol_to_cell(row_map[t], ci), xl_rowcol_to_cell(row_map[t], ci - 1)
                    return f"=IFERROR(({cr}-{pr})/{pr},0)", use_fmt or pct_fmt
                return 0, use_fmt or pct_fmt
        else:
            refs = [xl_rowcol_to_cell(row_map[x], ci) for x in cmp if x in row_map]
            if refs: return "=" + "+".join(refs), use_fmt or num_fmt
        return None, None

    def _autosize(self, ws, col_idx, data, headers):
        lens = []
        for r in data:
            if r is None or not isinstance(r, (list, tuple, np.ndarray)): continue
            if len(r) > 0 and r[0] is not None: lens.append(len(str(r[0])))
        h_len = len(str(headers[0])) if headers is not None and len(headers) > 0 else 0
        mx = max(lens + ([h_len] if h_len > 0 else [])) if lens or h_len > 0 else 10
        ws.set_column(col_idx, col_idx, mx + 2)

    def write_to_xlsx(self, filepath, sheets, unit_label, mults=None):
        mults = mults or []
        opts = {'options': {'nan_inf_to_errors': True}}
        with pd.ExcelWriter(filepath, engine="xlsxwriter", engine_kwargs=opts) as writer:
            wb = writer.book
            fmts = {
                'dt': wb.add_format({"num_format": "0", "align": "right"}),
                'pr': wb.add_format({"num_format": "#,##0.00", "align": "right"}),
                'num': wb.add_format({"num_format": "#,##0", "align": "right"}),
                'pct': wb.add_format({"num_format": "0%", "align": "right"}),
                'm': wb.add_format({"num_format": "0.0", "align": "right"}),
                'txt': wb.add_format({"align": "left"})
            }
            ws_m = wb.add_worksheet("Main")
            ws_m.write(2, 1, unit_label, fmts['txt'])
            
            s_map = {"income_statement(a)": "Income Statement(a)", "income_statement(q)": "Income Statement(q)", \
                     "cash_flow(a)": "Cash Flow(a)", "cash_flow(q)": "Cash Flow(q)", \
                     "balance_sheet(a)": "Balance Sheet(a)", "balance_sheet(q)": "Balance Sheet(q)"}
            g_row_maps, p_sheets = {}, {}
            
            for k, s_name in s_map.items():
                raw = sheets.pop(k, None)
                if raw is not None: 
                    raw = raw.to_pandas()
                    
                    ws = wb.add_worksheet(s_name)
                    data = self._format_data(raw)
                    r_map = {str(r[0]): 2 + idx for idx, r in enumerate(data) if r and r[0] is not None}
                    g_row_maps[k], p_sheets[k] = r_map, data
                    
                    for c, col in enumerate(raw.columns):
                        if c == 0: continue
                        h = str(col); y = h[:4]
                        ws.write(1, 1 + c, int(y) if y.isdigit() else col, fmts['dt'])
                    
                    cur_f = self.formulae.get(k, {})
                    for i, row in enumerate(data):
                        lbl = str(row[0]) if row[0] is not None else ""
                        ws.write(2 + i, 1, lbl, fmts['txt'])
                        
                        is_pct_row = False
                        if lbl:
                            raw_match = raw[raw.iloc[:, 0] == lbl]
                            if not raw_match.empty:
                                first_val = str(raw_match.iloc[0, 1])
                                is_pct_row = "%" in first_val

                        for j, val in enumerate(row):
                            if j == 0: continue
                            ri, ci = 2 + i, 1 + j
                            
                            if lbl in cur_f:
                                f_val, f_fmt = self._apply_formula(lbl, cur_f[lbl], r_map, ci, mults, fmts)
                                if f_val is not None:
                                    ws.write_formula(ri, ci, f_val, f_fmt) if isinstance(f_val, str) else ws.write(ri, ci, f_val, f_fmt)
                                    continue
                            
                            if val is not None:
                                if lbl in mults: f = fmts['m']
                                elif is_pct_row: f = fmts['pct']
                                else: f = fmts['num']
                                ws.write(ri, ci, val, f)
                            else:
                                ws.write_blank(ri, ci, None, fmts['num'])
                    self._autosize(ws, 1, data, raw.columns)

            sum_df = sheets.pop("summary", None)
            if sum_df is not None:
                sr, lr = {}, 0
                for i, row in enumerate(sum_df.values):
                    lb, v = row[0], row[1]
                    ri, cv = 2 + i, 9
                    lr, sr[lb] = ri, xl_rowcol_to_cell(ri, cv)
                    ws_m.write(ri, 8, lb, fmts['txt'])
                    u_f = fmts['m'] if lb in mults else None
                    if lb == "MC" and "Price" in sr and "Shares" in sr:
                        ws_m.write_formula(ri, cv, f"={sr['Price']}*{sr['Shares']}", u_f or fmts['num'])
                    elif lb == "EV" and all(x in sr for x in ["MC", "Debt", "Cash"]):
                        ws_m.write_formula(ri, cv, f"={sr['MC']}+{sr['Debt']}-{sr['Cash']}", u_f or fmts['num'])
                    else:
                        ws_m.write(ri, cv, v, u_f or (fmts['pr'] if lb == "Price" else fmts['num']))
                
                ism = g_row_maps.get("income_statement(a)")
                if ism and "Operating Income" in ism and "EV" in sr:
                    is_d = p_sheets["income_statement(a)"]
                    lc = len(is_d[0]) - 1 if is_d else 2
                    o_ref = f"'Income Statement'!{xl_rowcol_to_cell(ism['Operating Income'], lc)}"
                    ws_m.write_formula(lr + 1, 9, f"={sr['EV']}/{o_ref}", fmts['m'])
                
                self._autosize(ws_m, 1, [unit_label], None)
                self._autosize(ws_m, 8, sum_df.values, sum_df.columns)
        return self

    def open_file(self, filepath):
        if not filepath.exists():
            raise FileNotFoundError(f"The path '{filepath}' does not exist.")
        
        system = platform.system()
        if system == "Windows":
            os.startfile(filepath)
        elif system == "Darwin":  # macOS
            subprocess.call(["open", filepath])
        else:  # Linux/Unix
            subprocess.call(["xdg-open", filepath])
  
class FXManager():
    def __init__(self):
        self._basepath = Path.cwd() / "data" / "raw" / "FX"
        self._basepath.mkdir(parents=True, exist_ok=True)
        
    def _fetch_fx(self, symbol, start, end):
        print(f"Fetching exchange rate ({symbol})")
        fx = Ticker(symbol)
        start = start - relativedelta(months=1)
        end = end + relativedelta(months=1)
        df = pl.from_pandas(fx.history(start=start, end=end, interval="1d").reset_index()).select(["date", "close"])
        return df

    def _get_fx_data(self, currecy, start, end):
        if currecy == "USD":
            return None
        symbol = f"USD{currecy}=X"
        filepath =  self._basepath / f"{symbol}.parquet"
        
        try:
            df = pl.read_parquet(filepath)
            start_loaded = df["date"].min()
            end_loaded = df["date"].max()
        except FileNotFoundError:
            start_loaded=None
            end_loaded=None
            
        if start_loaded is None or end_loaded is None:
            df = self._fetch_fx(symbol, start, end)
        elif start<start_loaded or end>end_loaded:
            new_df = self._fetch_fx(symbol, start, end)
            df = pl.concat([df, new_df]).unique(subset=["date"], keep="last").sort("date")
        else:
            return df
        df.write_parquet(filepath)
        return df

    def adj_fx_table(self, df):
        if df is None or df.is_empty():
            return None
        
        fx_df = {}
        results = []
        currencies = df["currencyCode"].unique()
        for currency in currencies:
            fx_df[currency] = self._get_fx_data(currency, df["date"].min(), df["date"].max())
            temp_df = df.filter(pl.col("currencyCode")==currency).sort("date")
            if fx_df[currency] is not None:
                fx_data = fx_df[currency].with_columns(pl.col("date").cast(df["date"].dtype))
                temp_df = temp_df.join_asof(fx_data, on="date", strategy="backward").rename({"close": "fx"})
            else:
                temp_df = temp_df.with_columns(pl.lit(1, dtype=pl.Float64).alias("fx"))
            results.append(temp_df)
        df = pl.concat(results).sort("date")
        
        df = df.with_columns(
            (pl.col(col) / pl.col("fx")).cast(pl.Float64).alias(col)
            for col in df.select(pl.selectors.numeric()).columns
            if not any(term in col.lower() for term in ["shares", "issuance", "market"]) and col != "fx"
        )
        return df

class FADisplay:
    def compute_fa_exponent(self, df):
        subset = ["NetIncome", "FreeCashFlow", "CommonStockEquity"]
        target = next((c for c in subset if c in df.columns), None)
        min_abs_val = df.group_by("symbol").agg(pl.col(target).abs().median()).sort(target).row(0)[1]
        exp = self.get_exponent(min_abs_val)
        return exp

    def _standardise_subsets(self, df):
        subsets = {
            "SellingGeneralAndAdministration": ["OperatingExpense"],
            "OtherOpEx": ["OperatingExpense"],
            "OtherIncome": ["PretaxIncome", "OperatingIncome"],
            "NetInterestIncome": ["PretaxIncome", "OperatingIncome"],
            "After-TaxAdj": ["PretaxIncome", "NetIncome"],
            "OtherStockholders": ["NetIncomeCommonStockholders"],
            "OtherCurrentAssets": ["CurrentAssets"],
            "OtherAssets": ["TotalNonCurrentAssets"],
            "OtherCurrentLiabilities": ["CurrentLiabilities"],
            "OtherLiabilities": ["TotalNonCurrentLiabilitiesNetMinorityInterest"],
            "AccumulatedOtherChanges": ["StockholdersEquity"],
            "Non-ControllingInterest": ["TotalEquityGrossMinorityInterest"],
            "OtherAdj": ["CashFlowFromContinuingOperatingActivities", "NetIncome"],
            "OtherInvesting": ["CashFlowFromContinuingInvestingActivities"], 
            "OtherFinancing": ["CashFlowFromContinuingFinancingActivities"],
            "SupplementaryAdj": ["EndCashPosition", "ChangesInCash"]
        }

        for parent, children in subsets.items():
            if parent not in df.columns:
                continue

            for child in children:
                if child not in df.columns:
                    df = df.with_columns(None).alias(parent)
                    break
                
                df = df.with_columns(
                    pl.when(df[child].is_null())
                      .then(None)
                      .otherwise(df[parent])
                      .alias(parent)
                )
        return df
           
    def get_exponent(self, min_abs_val):
        for exp in range(12, -1, -3):
            divisor = 10 ** exp
            if abs(min_abs_val)>=divisor:
                if exp>3:
                    exp-=3
                return exp

    def format_display(self, df):
        def fmt(x, mode=",", mult=1):
            if x is None:
                return ""
            if isinstance(x, (int, float)):
                x = x*mult
                if mode==",":
                    if x<0:
                        return f"({np.abs(x):,})"
                    return f"{x:,}"
                elif mode=="%":
                    return f"{(x*100):,.2f}%"
                elif mode=="m":
                    return f"{x:,.1f}"
                else:
                    return f"{x}{mode}"
            return str(x)

        mult_cols = [c for c in df.columns if c in ["EVE", "EVS", "EVB", "PE", "PS", "PB", "EPS", "Beta", "Close"]]
        float_cols = [c for c, dtype in df.schema.items()if c != "date" and dtype == pl.Float64 and c not in mult_cols]
        int_cols = [c for c, dtype in df.schema.items() if c != "date" and dtype == pl.Int64 and c not in mult_cols]
        df = df.with_columns(
            [pl.col(c).map_elements(lambda x: fmt(x, "m")).alias(c) for c in mult_cols] +
            [pl.col(c).map_elements(lambda x: fmt(x, "%")).alias(c) for c in float_cols] +
            [pl.col(c).map_elements(lambda x: fmt(x, ",")).alias(c) for c in int_cols]
        )
        df = df.cast(pl.Utf8).with_columns([
            pl.when(pl.col(c) == "0")
              .then(pl.lit("--"))
              .otherwise(pl.col(c))
              .alias(c)
            for c in df.columns
        ]).fill_null("--")
        return df

            
class YahooQueryAPI:
    def __init__(self, sleep_s, SIZE=10):
        self._sleep_s = sleep_s
        self._renames, self._metrics, self._standard_cols = self._load_configs()
        self._SIZE = SIZE

        self._fx = FXManager()
        self._pro = MetricProcessor()
        self._dis = FADisplay()
        self._excel = ExcelManager()

    def _load_configs(self):
        basepath = Path.cwd() / "config" / "fa_config"
        basepath.mkdir(parents=True, exist_ok=True)

        renames = {}
        metrics = {}
        standard_cols = {}
        for domain in ["income_statement", "cash_flow", "balance_sheet"]:
            filepath = basepath / f"{domain}_rename.json"
            with open(filepath, "r") as f:
                renames[domain] = json.load(f)
             
            filepath = basepath / f"{domain}_add.json"
            with open(filepath, "r") as f:
                metrics[domain] = json.load(f)

            cols = ["symbol", "date"]
            for name, session_dict in metrics[domain].items():
                if "hidden" not in name:
                    cols.extend(session_dict.keys())
            standard_cols[domain] = cols
        return renames, metrics, standard_cols

    def fetch_screeners(self, filters=['most_actives', 'day_gainers', 'day_losers']):
        screeners = {}
        print(f"Fetching Live Screeners -> {len(filters)}")
        for f in filters:
            data = Screener().get_screeners(f, count=30)
            rows = [{"symbol": f"{item.get('symbol')} ({item.get('shortName', '')})", "price": item.get('regularMarketPrice'), "pctChange": round(item.get('regularMarketChangePercent'), 2)}
                    for item in data[f]['quotes']]
            df = pl.DataFrame(rows)
            time.sleep(0.5)
            screeners[data[f].get("title", f)] = df
        return screeners

    def fetch_history(self, symbols, start, end, interval):
        time.sleep(self._sleep_s)
        ticker = Ticker(symbols)
        retrieved = ticker.history(start=start, end=end, interval=interval)
        df = pl.from_pandas(retrieved.reset_index()).select(["symbol", "date", "open", "high", "low", "close", "volume", "adjclose", "dividends"])
        df = df.with_columns([pl.col(c) for c in df.columns if c not in ["symbol", "date"]])
        return df if not df.is_empty() else None

    def _fetch_live_data(self, symbols, exp=None):
        if not isinstance(symbols, list): symbols = [symbols]
        print("[INFO]Getting live price data...")
        ticker = Ticker(symbols)
        rows = []
        for symbol in symbols:
            price_data = ticker.price[symbol]
            price = price_data["regularMarketPrice"]
            row = {"symbol": symbol, "Price": price,}
            if exp is not None:
                mc = price_data["marketCap"]
                divisor = 10 ** exp
                row["mc"] = mc / divisor
            rows.append(row)
        return pl.DataFrame(rows)

    def _fetch_financial(self, symbols, domain, frequency="annual"):
        time.sleep(self._sleep_s)
        ticker = Ticker(symbols, asynchronous=True)
        print(f"Fetching {domain.title()}({frequency[0]}) -> {len(symbols)}")
            
        if domain == "income_statement":
            retrieved = ticker.income_statement(frequency=frequency, trailing=False)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="NetIncome")
        elif domain == "cash_flow":
            retrieved = ticker.cash_flow(frequency=frequency, trailing=False)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="OperatingCashFlow")
        elif domain == "balance_sheet":
            retrieved = ticker.balance_sheet(frequency=frequency)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="CommonStockEquity")
            df = df.with_columns(pl.col("OrdinarySharesNumber").forward_fill().over("symbol"))
        else:
            print("Invalid domain")
            return None
        
        df = df.with_columns(pl.col("asOfDate").dt.date().alias("date")).drop("asOfDate")
        df = df.select(["date"] + [c for c in df.columns if c != "date"])
        return df if not df.is_empty() else None

    def get_combined_financials(self, symbols, frequency): 
        domains = ["income_statement", "cash_flow", "balance_sheet"]
        identifiers = ["symbol", "date", "currencyCode"]
        for i, domain in enumerate(domains):
            df = self._fetch_financial(symbols, domain, frequency)
            if not isinstance(df, pl.DataFrame):
                return None
        
            if i==0:
                combined_df = df
            else:
                combined_df = combined_df.join(df, on=identifiers, how="inner")
                combined_df = combined_df.drop([c for c in combined_df.columns if c.endswith("_right")])
        df = combined_df.select(identifiers+[c for c in combined_df.columns if c not in identifiers])
        df = max_metric(df, des="DilutedAverageShares", src="BasicAverageShares")
        df = max_metric(df, des="DilutedAverageShares", src="OrdinarySharesNumber")
        df = replace_metric(df, des="CashFlowFromContinuingOperatingActivities", src="OperatingCashFlow")
        df = replace_metric(df, des="CashFlowFromContinuingInvestingActivities", src="InvestingCashFlow")
        df = replace_metric(df, des="CashFlowFromContinuingFinancingActivities", src="FinancingCashFlow")
        df = replace_metric(df, des="OperatingCashFlow", src="CashFlowFromContinuingOperatingActivities")
        df = replace_metric(df, des="CashCashEquivalentsAndShortTermInvestments", src="CashAndCashEquivalents")

        df = self._fx.adj_fx_table(df)
        if df is None or df.is_empty():
            return None

        standard_cols = []
        for domain in domains:
            df = self._pro.add_metrics(df, self._metrics[domain], NEW=False)
        standard_cols = list(dict.fromkeys(col for domain in domains for col in self._standard_cols[domain]))
        df = df.with_columns([pl.lit(0).alias(c) for c in standard_cols if c not in df.columns])
        df = df.select(standard_cols) #determinstic schema
        return df.sort(["symbol", "date"])
    
    def fetch_batch(self, missing_symbols, function, params=None):
        if not missing_symbols:
            return None, []
        if params is None:
            params = {}

        all_frames = []
        failed_symbols = {}
        def process(batch, idx): #recursively
            while batch:
                try:
                    df = function(batch, **params)
                    if df is not None:
                        all_frames.append(df)
                        print(df.head(3))
                    return
                except Exception as e:
                    print(f"[WARNING] Batch {idx} failed: {batch} due to {e}")
                    if len(batch) == 1:
                        failed_symbols[batch[0]] = e
                        print(f"[ERROR] Single symbol failed: {batch[0]}")
                        return
                    mid = len(batch) // 2
                    process(batch[:mid], f"{idx}a")
                    process(batch[mid:], f"{idx}b")
                    return

        batches = [missing_symbols[i:i+self._SIZE] for i in range(0, len(missing_symbols), self._SIZE)]
        for i, batch in enumerate(batches):
            time.sleep(int(len(batch) * 0.5))
            process(batch, i)
            
        df = pl.concat(all_frames) if all_frames else None
        return df, failed_symbols

    def get_tables(self, fa, rv=None, domains=["income_statement", "cash_flow", "balance_sheet"]):
        highlights = [
            "Revenue", "Gross Profit", "Operating Income", "Operating Expense", "Pretax Income", "Net Income", "NI Common", \
            "CFO", "CFI", "CFF", "Changes In Cash", "End Cash Position", "FCF", \
            "Current Assets", "Non-Current Assets", "Current Liabilities", "Non-Current Liabilities", "Stockholders Equity", "Total Assets", "Total Liabilities", "Total Equity", "NCAV", \
            "Beta", "EVE", "PE", "PB", "PS", "EPS", "ROTA", "ROIC", "ROCE", "ROE"        
            ]
        
        dfs = {}
        exp = self._dis.compute_fa_exponent(fa)
        divisor = 10**exp
        costs = ["COGS", "SG&A", "R&D", "Others", "Operating Expense", "Taxes", "After-Tax Adj"]
        symbols = fa["symbol"].unique(maintain_order=True).to_list()
        for symbol in symbols:
            data = {}
            for domain in domains:
                temp_df = fa.filter(pl.col("symbol") == symbol)
                temp_df = temp_df.select(self._standard_cols[domain]).drop("symbol")
                temp_df = self._dis._standardise_subsets(temp_df) 
                temp_df = temp_df.rename(self._renames[domain])
                temp_df = temp_df.rename({col: (lambda name: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name))(col) for col in temp_df.columns})
                temp_df = temp_df.with_columns((pl.col(c)*-1).alias(c) for c in costs if c in temp_df.columns)
                temp_df = temp_df.with_columns((pl.col(pl.Int64) / divisor).round(0).cast(pl.Int64)) 
                data[domain] = temp_df.sort("date")
            dfs[symbol] = data
        return dfs, exp, highlights
    
    def save_FA(self, annual, quarterly):
        basepath = Path.cwd() / "data"/ "output" / "fa_models"
        basepath.mkdir(parents=True, exist_ok=True)

        a_dfs, _, _ = self.get_tables(annual, domains=["income_statement", "cash_flow", "balance_sheet"])
        q_dfs, exp, highlights = self.get_tables(quarterly, domains=["income_statement", "cash_flow", "balance_sheet"])        
        UNITS = {12: "USD (in trillions)", 9: "USD (in billions)", 6: "USD (in millions)", 3: "USD (in thousands)", 0: "USD"}
        unit_label = UNITS.get(exp, "")

        for symbol in a_dfs.keys():
            data = {
                **{f"{k}(a)": v for k, v in a_dfs[symbol].items()},
                **{f"{k}(q)": v for k, v, in q_dfs[symbol].items()}
            }
            live = self._fetch_live_data(symbol, exp) 
            latest_bs = data["balance_sheet(q)"].tail(1)
            cash = latest_bs["Cash & Short Inv"]
            debt = latest_bs["Net Debt"] + cash
            
            live = live.with_columns([
                ((pl.col("mc") / pl.col("Price"))).alias("Shares"),
                (pl.col("mc")).alias("MC"),
                (pl.lit(cash)).alias("Cash"),
                (pl.lit(debt)).alias("Debt")
                ])
            live = live.with_columns([
                (pl.col("mc") + pl.col("Cash") - pl.col("Debt")).alias("EV")
                ]).select(["Price", "Shares", "MC", "Cash", "Debt", "EV"])
            summary = live.to_pandas()
            summary = pd.DataFrame(summary.iloc[0]).reset_index()
            summary.columns = ["",""]    
            
            for domain_key, df_table in data.items():
                data[domain_key] = self._excel.format_table(data[domain_key], frequency=domain_key[-2])
                data[domain_key] = transpose_df(data[domain_key], "date")
            data["summary"] = summary
            self._excel.write_to_xlsx(basepath / f"{symbol}.xlsx", data, unit_label, mults=["EPS"])
        return self






