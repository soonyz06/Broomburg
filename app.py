import dash
from src.ui.layout import layout
from src.ui.callbacks import register_callbacks
from dash import Output, Input

app = dash.Dash(__name__, suppress_callback_exceptions=True)
RESULT_CACHE = {}

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}<title>Broomburg</title>{%favicon%}{%css%}
        <style>
            *::-webkit-scrollbar { display: none !important; }
            * { 
                -ms-overflow-style: none !important; 
                scrollbar-width: none !important; 
                outline: none !important; 
                box-shadow: none !important;
                border: none !important;
            }
            html, body { 
                margin: 0; padding: 0; background-color: black; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden !important; width: 100vw; height: 100vh;
            }
            ._dash-undo-redo, .dash-debug-menu { display: none !important; }
            #sidebar-container { 
                transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; 
                display: flex; flex-direction: column; align-items: flex-start; 
            }
            .sidebar-icon-box {
                width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
            }
        </style>
    </head>
    <body id="body-root">{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''

app.layout = layout
register_callbacks(app, RESULT_CACHE)

app.clientside_callback(
    """
    function(id) {
        // 1. STABLE KEYBINDS
        document.removeEventListener('keydown', window.dashKeyHandler);
        window.dashKeyHandler = function(e) {
            if (e.key === 'Escape') { 
                const btn = document.getElementById('btn-expand');
                if (btn) btn.click(); 
            }
            if (e.key === '`') { 
                const inp = document.getElementById('chat-input');
                if (inp) { e.preventDefault(); inp.focus(); }
            }
        };
        document.addEventListener('keydown', window.dashKeyHandler);
        
        // 2. HEIGHT-BASED AUTO-SCROLL
        var chatH = document.getElementById('chat-history');
        if (chatH) {
            // Track the last known scroll height
            chatH.lastHeight = chatH.scrollHeight;

            new MutationObserver(() => {
                var currentHeight = chatH.scrollHeight;
                // ONLY scroll if the content actually got taller (new message or new plot rendered)
                if (currentHeight > chatH.lastHeight) {
                    chatH.scrollTo({top: currentHeight, behavior: 'smooth'});
                }
                chatH.lastHeight = currentHeight;
            }).observe(chatH, { childList: true, subtree: true });
        }

        // 3. SIDEBAR CENTERING
        var sideH = document.getElementById('sidebar-history-list');
        if (sideH) {
            new MutationObserver(() => {
                setTimeout(() => {
                   var active = sideH.querySelector('[data-active="true"]');
                   if (active) { active.scrollIntoView({behavior: 'smooth', block: 'center'}); }
                }, 50);
            }).observe(sideH, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-active'] });
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output('layout-wrapper', 'id'), Input('layout-wrapper', 'id')
)
if __name__ == "__main__":
    app.run(debug=True)
