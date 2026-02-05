from pathlib import Path
import polars as pl
import pandas as pd

from src.utils.logg import log_info, log_df
from src.io.fetch import load_apikeys
from src.io.sec import EdgarToolsAPI
import dash
from dash import dcc, html, Input, Output, State


# ======================== DATA ========================


def load_data():
    edgar = EdgarToolsAPI(0.5)
    history_start = "2023-01-01"
        
    symbol = "AAPL"
    filings = edgar.fetch_filings(symbol, ["10-K", "20-F"], startDate=history_start)
    df = edgar.process_obj(symbol, filings, limit=1)

    symbol = df["symbol"][0]
    sections = df.select(['business', 'risk_factors', 'management_discussion'])
    texts = sections.row(0, named=True)
    columns = list(texts.keys())
    log_info("Fetch", t0)
    return {
        "symbol": symbol,
        "texts": texts,
        "columns": columns,
    }


# ======================== DASH ========================


app = dash.Dash(__name__)


# ======================== Layout ========================


def serve_layout():
    return html.Div(
        [
            html.H1(id="page-title"),
            html.Div(id="section-title"),
            html.Div(id="text-display"),
            dcc.Store(id="data-store"),
            dcc.Store(id="current-index", data=0),
            dcc.Store(id="key-press"),
        ]
    )

app.layout = serve_layout


# ======================== Callbacks ========================


@app.callback(
    Output("data-store", "data"),
    Input("page-title", "id"),
    prevent_initial_call=False,
)
def init_app(_):
    return load_data()

app.clientside_callback(
    """
    function(data) {
        // Don't attach listener until data is loaded
        if (!data) {
            return null;
        }

        if (!window._dashKeyListenerAttached) {
            window.addEventListener("keydown", function(e) {
                if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                    e.preventDefault();

                    window.dash_clientside.set_props(
                        "key-press",
                        { data: e.key === "ArrowLeft" ? "prev" : "next" }
                    );
                }
            });

            window._dashKeyListenerAttached = true;
        }
        return null;

        if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
            e.preventDefault();
        }

        if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
            return;
        }
    }
    """,
    Output("key-press", "data"),
    Input("data-store", "data"),   
)


@app.callback(
    Output("page-title", "children"),
    Output("section-title", "children"),
    Output("text-display", "children"),
    Output("current-index", "data"),
    Input("key-press", "data"),  
    State("current-index", "data"),
    State("data-store", "data"),
)
def update_text(key, current_index, data):
    if not data:
        raise dash.exceptions.PreventUpdate

    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if key == "prev":
            current_index = (current_index - 1) % len(data["columns"])
        elif key == "next":
            current_index = (current_index + 1) % len(data["columns"])

    col = data["columns"][current_index]

    return (
        f"{data['symbol']} – Filing Sections",
        col.replace("_", " ").title(),
        data["texts"][col],
        current_index,
    )


# ======================== APP ========================


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    # C:\Users\User\Desktop\Model

##nlp/rag, load sections on demand+cache, vector embedding, semantic chunking,, create vector lookup table
#Abstraction, Coupling, Locality


