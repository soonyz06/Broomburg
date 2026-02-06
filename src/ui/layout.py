from dash import dcc, html

USER_STYLE = {"alignSelf": "flex-end", "background": "#4a4a4a", "color": "white", "padding": "14px 22px", "borderRadius": "25px 25px 4px 25px", "maxWidth": "70%", "fontSize": "15px", "flexShrink": 0}
AI_STYLE = {"alignSelf": "flex-start", "width": "100%", "background": "rgba(0,0,0,0)", "color": "white", "padding": "20px", "borderRadius": "20px", "border": "1px solid #333 !important", "fontSize": "15px", "flexShrink": 0}

layout = html.Div([
    dcc.Store(id="store-sessions", data=[]),        
    dcc.Store(id="active-session-id", data=None),   
    dcc.Store(id="sidebar-state", data=False),

    html.Div([
        html.Div([
            html.Div(html.Div("☰", id="btn-expand", n_clicks=0, style={"cursor": "pointer", "fontSize": "18px"}), className="sidebar-icon-box"),
            html.Div(html.Div("＋", id="btn-new-chat", n_clicks=0, style={"cursor": "pointer", "fontSize": "22px"}), className="sidebar-icon-box"),
        ], style={"width": "260px"}),
        html.Div(id="sidebar-history-list", style={"marginTop": "10px", "overflowY": "auto", "overflowX": "hidden", "width": "260px", "flex": "1"}),
        html.Div([
             html.Div(html.Div("⚙", id="btn-settings", style={"cursor": "pointer", "fontSize": "18px", "opacity": "0.6"}), className="sidebar-icon-box")
        ], style={"marginTop": "auto", "paddingBottom": "10px", "width": "260px"})
    ], id="sidebar-container", style={"position": "fixed", "top": 0, "left": 0, "zIndex": "100", "width": "60px", "backgroundColor": "#111", "color": "white", "height": "100vh", "overflow": "hidden"}),

    html.Div([
        html.Div(id="chat-history", children=[], style={"flex": "1", "overflowY": "auto", "padding": "40px 10%", "display": "flex", "flexDirection": "column", "gap": "20px"}),
        html.Div([
            html.Div([
                dcc.Input(id="chat-input", type="text", placeholder="Ask something...", autoComplete="off", style={"flex": "1", "padding": "25px 30px", "borderRadius": "50px 0 0 50px", "background": "#2c2c2c", "color": "white", "fontSize": "18px"}),
                html.Button("➤", id="btn-submit", n_clicks=0, style={"width": "80px", "borderRadius": "0 50px 50px 0", "background": "#3d3d3d", "color": "white", "fontSize": "20px", "cursor": "pointer"})
            ], style={"display": "flex", "width": "100%"})
        ], style={"padding": "20px 10% 40px 10%", "width": "100%", "boxSizing": "border-box", "background": "black"})
    ], id="main-content", style={"marginLeft": "60px", "width": "calc(100vw - 60px)", "display": "flex", "flexDirection": "column", "backgroundColor": "black", "height": "100vh", "transition": "margin-left 0.3s ease, width 0.3s ease"})
], id="layout-wrapper")
