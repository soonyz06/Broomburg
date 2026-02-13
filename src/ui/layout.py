from dash import dcc, html
import dash_mantine_components as dmc 

USER_STYLE = {"alignSelf": "flex-end", "background": "#4a4a4a", "color": "white", "padding": "14px 22px", "borderRadius": "25px 25px 4px 25px", "maxWidth": "70%", "fontSize": "15px", "flexShrink": 0}
AI_STYLE = {"alignSelf": "flex-start", "width": "100%", "background": "rgba(0,0,0,0)", "color": "white", "padding": "20px", "borderRadius": "20px", "border": "1px solid #333 !important", "fontSize": "15px", "flexShrink": 0}

layout = dmc.MantineProvider(
    forceColorScheme="dark", 
    theme={
        "primaryColor": "blue",
        "fontFamily": "'Inter', sans-serif",
    },
    children=[
        html.Div([
            dcc.Store(id="store-sessions", data=[]),        
            dcc.Store(id="active-session-id", data=None),   
            dcc.Store(id="sidebar-state", data=False),

            # Sidebar
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

            # Main Content
            html.Div([
                # --- HEADER BAR ---
                html.Div([
                    dmc.Text(
                        "Broomburg", 
                        size="22px", # Slightly smaller than before for elegance
                        fw=700, 
                        c="white",
                        style={"letterSpacing": "0.5px"}
                    )
                ], style={
                    "height": "70px", # Slightly shorter height
                    "width": "100%", 
                    "backgroundColor": "#000", 
                    "borderBottom": "1px solid #222", 
                    "display": "flex", 
                    "alignItems": "center", 
                    "padding": "0 40px",
                    "flexShrink": 0
                }),

                # Chat History
                html.Div(id="chat-history", children=[], style={
                    "flex": "1", 
                    "overflowY": "auto", 
                    "padding": "30px 18%", 
                    "display": "flex", 
                    "flexDirection": "column", 
                    "gap": "25px"
                }),

                # Prompt Input Area
                html.Div([
                    html.Div([
                        dcc.Input(
                            id="chat-input", 
                            type="text", 
                            placeholder="Ask Broomburg anything...", 
                            autoComplete="off", 
                            style={
                                "flex": "1", 
                                "padding": "16px 25px", 
                                "borderRadius": "12px 0 0 12px", 
                                "background": "#1a1a1a", 
                                "color": "white", 
                                "fontSize": "15px", # MATCHES BUBBLE FONT SIZE
                                "border": "1px solid #333",
                                "borderRight": "none"
                            }
                        ),
                        html.Button(
                            "➤", 
                            id="btn-submit", 
                            n_clicks=0, 
                            style={
                                "width": "60px", 
                                "borderRadius": "0 12px 12px 0", 
                                "background": "#2c2c2c", # Reverted color to match input theme
                                "color": "white", 
                                "fontSize": "18px", 
                                "cursor": "pointer",
                                "border": "1px solid #333"
                            }
                        )
                    ], style={"display": "flex", "width": "100%", "maxWidth": "1000px", "margin": "0 auto"})
                ], style={"padding": "20px 18% 40px 18%", "width": "100%", "boxSizing": "border-box", "background": "black"})
            ], id="main-content", style={"marginLeft": "60px", "width": "calc(100vw - 60px)", "display": "flex", "flexDirection": "column", "backgroundColor": "black", "height": "100vh", "transition": "margin-left 0.3s ease, width 0.3s ease"})
        ], id="layout-wrapper")
    ]
)
