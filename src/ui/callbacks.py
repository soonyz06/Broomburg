import dash
from dash import html, Input, Output, State, Patch, ALL
import uuid
import json
from src.ui.registry_setup import registry
from src.ui.layout import USER_STYLE, AI_STYLE

def register_callbacks(app, RESULT_CACHE):
    @app.callback(Output("sidebar-state", "data"), Input("btn-expand", "n_clicks"), State("sidebar-state", "data"), prevent_initial_call=True)
    def toggle_sidebar(n, state): return not state

    @app.callback(
        Output("sidebar-container", "style"), Output("main-content", "style"),
        Output("sidebar-history-list", "children"), Output("active-session-id", "data", allow_duplicate=True),
        Output("chat-history", "children", allow_duplicate=True),
        Input("sidebar-state", "data"), Input("store-sessions", "data"), Input("active-session-id", "data"),
        Input({'type': 'sidebar-item', 'index': ALL}, 'n_clicks'),
        State("sidebar-container", "style"), State("main-content", "style"),
        prevent_initial_call=True
    )
    def unified_sidebar_manager(is_expanded, sessions, active_id, n_clicks, cur_side_style, cur_main_style):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id']
        new_active_id = active_id
        new_chat_display = dash.no_update

        if 'sidebar-item' in triggered_id:
            new_active_id = json.loads(triggered_id.split('.')[0])['index']
            session = next((s for s in sessions if s['id'] == new_active_id), None)
            if session:
                new_chat_display = []
                for i, msg in enumerate(session['messages']):
                    content = RESULT_CACHE.get(f"{new_active_id}_{i}", html.Div("..."))
                    new_chat_display.extend([html.Div(msg, style=USER_STYLE), html.Div(content, style=AI_STYLE)])

        side_w = "260px" if is_expanded else "60px"
        new_side_style = {**cur_side_style, "width": side_w}
        new_main_style = {**cur_main_style, "marginLeft": side_w, "width": f"calc(100vw - {side_w})"}

        history_items = []
        if sessions:
            for s in sessions:
                is_active = (s['id'] == new_active_id)
                history_items.append(html.Div([
                    html.Div("•", className="sidebar-icon-box", style={"fontSize": "10px", "color": "white" if is_active else "#444"}),
                    html.Div(s['name'], style={"display": "block" if is_expanded else "none", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis", "flex": "1"})
                ], id={'type': 'sidebar-item', 'index': s['id']}, n_clicks=0,
                   **{"data-active": "true" if is_active else "false"},
                   style={
                    "display": "flex", "alignItems": "center", "height": "50px", "margin": "4px 0",
                    "backgroundColor": ("#222" if is_active and is_expanded else "transparent"),
                    "cursor": "pointer", "color": "white" if is_active else "#888",
                    "borderLeft": ("3px solid white" if is_active and is_expanded else "3px solid transparent")
                }))
        return new_side_style, new_main_style, history_items, new_active_id, new_chat_display

    @app.callback(
        Output("store-sessions", "data"), Output("active-session-id", "data"),
        Output("chat-input", "value"), Output("chat-history", "children", allow_duplicate=True),
        Output("btn-submit", "children"),
        Input("chat-input", "n_submit"), Input("btn-submit", "n_clicks"), Input("btn-new-chat", "n_clicks"),
        State("chat-input", "value"), State("store-sessions", "data"),
        State("active-session-id", "data"), State("chat-history", "children"),
        prevent_initial_call=True
    )
    def handle_interaction(n_submit, n_btn, n_new, prompt, sessions, active_id, current_history):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        is_thinking = any(hasattr(e, 'get') and e.get('props', {}).get('id') == "loading-placeholder" for e in (current_history or []))

        if is_thinking:
            if trigger == "btn-submit":
                p = Patch(); del p[-1]
                p.append(html.Div("Interrupted.", style={**AI_STYLE, "color": "red"}))
                return dash.no_update, dash.no_update, dash.no_update, p, "➤"
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if trigger == "btn-new-chat" or (prompt and prompt.lower().strip() == "new"):
            return sessions, str(uuid.uuid4()), "", [], "➤"
        if prompt and prompt.lower().strip() in ["cls", "clear"]:
            return [s for s in sessions if s['id'] != active_id], str(uuid.uuid4()), "", [], "➤"
        if not prompt: return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        raw_tokens = prompt.strip().split()
        if raw_tokens[0].lower() == "tab" and len(raw_tokens) >= 2:
            try:
                idx = int(raw_tokens[1])
                if 0 <= idx < len(sessions):
                    if len(raw_tokens) >= 4 and raw_tokens[2].lower() == "rename":
                        sessions[idx]['name'] = " ".join(raw_tokens[3:]); return sessions, active_id, "", dash.no_update, "➤"
                    new_id = sessions[idx]['id']; rebuild = []
                    for i, msg in enumerate(sessions[idx]['messages']):
                        content = RESULT_CACHE.get(f"{new_id}_{i}", html.Div("..."))
                        rebuild.extend([html.Div(msg, style=USER_STYLE), html.Div(content, style=AI_STYLE)])
                    return sessions, new_id, "", rebuild, "➤"
                else:
                    p = Patch(); p.append(html.Div(prompt, style=USER_STYLE)); p.append(html.Div(f"Invalid tab {idx}", style={**AI_STYLE, "color": "orange"})); return sessions, active_id, "", p, "➤"
            except: pass

        if not active_id: active_id = str(uuid.uuid4())
        curr = next((s for s in sessions if s['id'] == active_id), None)
        if not curr:
            curr = {"id": active_id, "name": prompt[:25], "messages": []}
            sessions.append(curr)
        curr['messages'].append(prompt.strip())
        p = Patch(); p.append(html.Div(prompt, style=USER_STYLE)); p.append(html.Div(["Thinking..."], id="loading-placeholder", style=AI_STYLE))
        return sessions, active_id, "", p, "■"

    @app.callback(
        Output("chat-history", "children"), Output("btn-submit", "children", allow_duplicate=True),
        Input("store-sessions", "data"), State("active-session-id", "data"), prevent_initial_call=True
    )
    def finalize_response(sessions, active_id):
        if not active_id or not sessions: return dash.no_update, "➤"
        session = next((s for s in sessions if s['id'] == active_id), None)
        if not session or not session['messages']: return dash.no_update, "➤"
        last_msg = session['messages'][-1].lower().strip()
        if last_msg.startswith("tab ") or last_msg in ["new", "cls", "clear"]: return dash.no_update, "➤"
        cache_key = f"{active_id}_{len(session['messages'])-1}"
        if cache_key in RESULT_CACHE: return dash.no_update, "➤"
        content, _ = registry.execute(session['messages'][-1])
        RESULT_CACHE[cache_key] = content
        p = Patch(); del p[-1]; p.append(html.Div(content, style=AI_STYLE)); return p, "➤"
