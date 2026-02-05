import dash
from dash import dcc, html, Input, Output, State, Patch, ALL
import plotly.express as px
import uuid
import json

# =============================================================================
# 1. REGISTRY & LOGIC
# =============================================================================
class UniversalRegistry:
    def __init__(self):
        self.commands = {}
    def register(self, keyword):
        def wrapper(func):
            self.commands[keyword.lower()] = func
            return func
        return wrapper
    def execute(self, prompt: str):
        if not prompt: return None, "EMPTY"
        p = prompt.strip().lower()
        return self.commands[p]() if p in self.commands else self.commands.get("default")()

registry = UniversalRegistry()
iris_df = px.data.iris()

@registry.register(":cls")
@registry.register("clear")
def clear_logic(): return None, "CLEAR"

@registry.register("default")
def default_logic():
    fig = px.scatter(iris_df, x="sepal_width", y="sepal_length", color="species", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", margin=dict(l=20, r=20, t=20, b=20))
    return dcc.Graph(figure=fig, style={"height": "350px"}, config={'displayModeBar': False}), "RENDER"

# =============================================================================
# 2. APP UI & CSS (THE INDEPENDENT LAYER FIX)
# =============================================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}<title>Broomburg</title>{%favicon%}{%css%}
        <style>
            ._dash-undo-redo { display: none !important; }
            *::-webkit-scrollbar { display: none !important; }
            
            html, body {
                margin: 0; padding: 0;
                background-color: black;
                overflow: hidden !important; 
                width: 100vw; height: 100vh;
                position: fixed;
            }

            * { 
                -ms-overflow-style: none !important; 
                scrollbar-width: none !important; 
                outline: none !important; 
                box-shadow: none !important;
            }
        </style>
    </head>
    <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''

app.layout = html.Div([
    dcc.Store(id="store-sessions", data=[]),        
    dcc.Store(id="active-session-id", data=None),   
    dcc.Store(id="sidebar-state", data=False),

    # 1. FIXED SIDEBAR (Independent Layer)
    html.Div([
        html.Div(id="sidebar-top-controls", children=[
            html.Div("☰", id="btn-expand", n_clicks=0, style={
                "cursor": "pointer", "fontSize": "18px", "height": "40px", "width": "40px", 
                "display": "flex", "alignItems": "center", "justifyContent": "center"
            }),
            html.Div("＋", id="btn-new-chat", n_clicks=0, style={
                "cursor": "pointer", "fontSize": "22px", "height": "40px", "width": "40px", 
                "display": "flex", "alignItems": "center", "justifyContent": "center"
            }),
        ], style={"display": "flex", "flexDirection": "column", "paddingTop": "10px"}),

        html.Div(id="sidebar-history-list", style={
            "marginTop": "15px", "overflowY": "auto", "width": "100%", 
            "display": "flex", "flexDirection": "column", "boxSizing": "border-box"
        }),
        
        html.Div(id="sidebar-bottom-controls", children=[
             html.Div("⚙", id="btn-settings", style={
                "cursor": "pointer", "fontSize": "18px", "opacity": "0.6",
                "height": "40px", "width": "40px",
                "display": "flex", "alignItems": "center", "justifyContent": "center"
            })
        ], style={"marginTop": "auto", "paddingBottom": "20px"})
    ], id="sidebar-container", style={
        "position": "fixed", "top": 0, "left": 0, "zIndex": "100",
        "width": "60px", "backgroundColor": "#1e1e1e", "display": "flex",
        "flexDirection": "column", "color": "white", "height": "100vh",
        "transition": "width 0.3s ease", "overflow": "hidden", "whiteSpace": "nowrap"
    }),

   # 2. MAIN CONTENT AREA
    html.Div([
        # Chat History
        html.Div(id="chat-history", children=[], style={
            "flex": "1", 
            "overflowY": "auto", 
            "padding": "40px 10%", 
            "display": "flex", 
            "flexDirection": "column", 
            "gap": "20px"
        }),
        
        # Input Area
        html.Div([
            dcc.Input(
                id="chat-input", 
                type="text", 
                placeholder="Ask something...",
                autoComplete="off", 
                style={
                    "width": "100%", 
                    "padding": "25px 30px", 
                    "borderRadius": "50px",
                    "background": "#2c2c2c", 
                    "color": "white", 
                    "fontSize": "18px",
                    "border": "none",           
                    "outline": "none",           
                    "boxShadow": "none",         
                    "appearance": "none",       
                    "WebkitAppearance": "none",  
                    "resize": "none",            
                }
            )
        ], style={
            "padding": "20px 10% 40px 10%", 
            "width": "100%", 
            "boxSizing": "border-box",
            "background": "black"
        })
        
    ], id="main-content", style={
        "marginLeft": "60px",
        "width": "calc(100vw - 60px)", 
        "display": "flex", 
        "flexDirection": "column", 
        "backgroundColor": "black", 
        "height": "100vh",
        "transition": "margin-left 0.3s ease, width 0.3s ease",
        "overflowX": "hidden"
    })
], style={"backgroundColor": "black", "width": "100vw", "height": "100vh", "overflow": "hidden"})

# =============================================================================
# 3. CALLBACKS
# =============================================================================

@app.callback(
    Output("sidebar-state", "data"),
    Input("btn-expand", "n_clicks"),
    State("sidebar-state", "data"),
    prevent_initial_call=True
)
def toggle_sidebar(n, state): return not state

@app.callback(
    Output("sidebar-container", "style"),
    Output("main-content", "style"),
    Output("sidebar-history-list", "children"),
    Output("sidebar-top-controls", "style"),
    Output("sidebar-bottom-controls", "style"),
    Output("active-session-id", "data", allow_duplicate=True),
    Output("chat-history", "children", allow_duplicate=True),
    Input("sidebar-state", "data"),
    Input("store-sessions", "data"),
    Input({'type': 'sidebar-item', 'index': ALL}, 'n_clicks'),
    State("sidebar-container", "style"),
    State("main-content", "style"),
    State("active-session-id", "data"),
    prevent_initial_call=True
)
def unified_sidebar_manager(is_expanded, sessions, n_clicks, cur_side_style, cur_main_style, active_id):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id']
    
    new_active_id = active_id
    new_chat_display = dash.no_update

    if 'sidebar-item' in triggered_id:
        new_active_id = json.loads(triggered_id.split('.')[0])['index']
        session = next((s for s in sessions if s['id'] == new_active_id), None)
        if session:
            new_chat_display = []
            for msg in session['messages']:
                content, _ = registry.execute(msg)
                new_chat_display.append(html.Div(msg, style={"alignSelf": "flex-end", "background": "#4a4a4a", "color": "white", "padding": "14px 22px", "borderRadius": "25px 25px 4px 25px", "maxWidth": "70%"}))
                new_chat_display.append(html.Div(content, style={"alignSelf": "flex-start", "width": "100%", "background": "#2c2c2c", "padding": "20px", "borderRadius": "20px", "border": "1px solid #444"}))

    # Update Styles
    side_w = "260px" if is_expanded else "60px"

    new_side_style = cur_side_style.copy()
    new_side_style["width"] = side_w

    new_main_style = cur_main_style.copy()
    new_main_style["marginLeft"] = side_w
    new_main_style["width"] = f"calc(100vw - {side_w})" 
    new_main_style["overflowX"] = "hidden"              

    icon_style = {"display": "flex", "flexDirection": "column", "paddingTop": "10px", "paddingLeft": "10px", "alignItems": "flex-start", "transition": "padding-left 0.3s ease"}
    bottom_style = icon_style.copy()
    bottom_style["marginTop"] = "auto"; bottom_style["paddingBottom"] = "20px"

    history_items = []
    if is_expanded:
        history_items = [
            html.Div(s['name'], id={'type': 'sidebar-item', 'index': s['id']}, n_clicks=0,
                style={
                    "padding": "10px 15px", "margin": "5px 8px", "borderRadius": "12px", 
                    "backgroundColor": "#3a3a3a" if s['id'] == new_active_id else "transparent",
                    "fontSize": "13px", "cursor": "pointer", "whiteSpace": "nowrap",
                    "overflow": "hidden", "textOverflow": "ellipsis", "boxSizing": "border-box",
                    "border": "1px solid #444" if s['id'] == new_active_id else "1px solid transparent"
                }) for s in sessions
        ]

    return new_side_style, new_main_style, history_items, icon_style, bottom_style, new_active_id, new_chat_display

@app.callback(
    Output("store-sessions", "data"),
    Output("active-session-id", "data"),
    Output("chat-input", "value"),
    Output("chat-history", "children", allow_duplicate=True),
    Input("chat-input", "n_submit"),
    Input("btn-new-chat", "n_clicks"),
    State("chat-input", "value"),
    State("store-sessions", "data"),
    State("active-session-id", "data"),
    prevent_initial_call=True
)
def handle_interaction(n_submit, n_new, prompt, sessions, active_id):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == "btn-new-chat": return sessions, str(uuid.uuid4()), "", []
    if not active_id: active_id = str(uuid.uuid4())
    if not prompt: return sessions, active_id, "", dash.no_update
    _, signal = registry.execute(prompt)
    if signal == "CLEAR":
        sessions = [s for s in sessions if s['id'] != active_id]
        return sessions, str(uuid.uuid4()), "", []
    curr = next((s for s in sessions if s['id'] == active_id), None)
    if not curr:
        curr = {"id": active_id, "name": prompt[:25], "messages": []}
        sessions.append(curr)
    curr['messages'].append(prompt)
    return sessions, active_id, "", dash.no_update

@app.callback(
    Output("chat-history", "children"),
    Input("store-sessions", "data"),
    State("active-session-id", "data"),
    prevent_initial_call=True
)
def update_chat_ui(sessions, active_id):
    if not active_id or not sessions: return []
    session = next((s for s in sessions if s['id'] == active_id), None)
    if not session or not session['messages']: return dash.no_update
    msg = session['messages'][-1]; content, _ = registry.execute(msg)
    p = Patch()
    p.append(html.Div(msg, style={"alignSelf": "flex-end", "background": "#4a4a4a", "color": "white", "padding": "14px 22px", "borderRadius": "25px 25px 4px 25px", "maxWidth": "70%"}))
    p.append(html.Div(content, style={"alignSelf": "flex-start", "width": "100%", "background": "#2c2c2c", "padding": "20px", "borderRadius": "20px", "border": "1px solid #444"}))
    return p

app.clientside_callback(
    """
    function(id){
        document.addEventListener('keydown', function(e){
            if(e.key==='~' || e.key==='`'){
                var i = document.getElementById('chat-input');
                if(i){ e.preventDefault(); i.focus(); window.scrollTo(0,0); }
            }
            if(e.key==='Tab'){ setTimeout(() => window.scrollTo(0,0), 0); }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-input', 'id'), Input('chat-input', 'id')
)

app.clientside_callback(
    "function(c){var h=document.getElementById('chat-history');if(h){setTimeout(()=>h.scrollTop=h.scrollHeight,50);}return window.dash_clientside.no_update;}",
    Output('chat-history', 'id'), Input('chat-history', 'children')
)

if __name__ == "__main__":
    app.run(debug=True)
