from __future__ import annotations
import dash
from dash import html, dcc, dash_table, Output, Input
import dash_bootstrap_components as dbc
from app_instance import app, server
from router import home_layout, not_found_layout
from common import _planning_ids_skeleton
from common import header_bar, sidebar_component, global_loading_overlay, GLOBAL_LOADING_STYLE
from planning_workspace import planning_layout, register_planning_ws
from plan_detail import plan_detail_validation_layout, register_plan_detail
from plan_store import auto_lock_previous_month_plans
from pages.forecast_page import page_forecast_section, forecast_shared_stores
from callbacks_pkg import *  # registers callbacks


# ---- Main Layout (verbatim) ----
app.layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed", data=True, storage_type="session"),
    dcc.Store(id="nav-log-dummy"),
    dcc.Store(id="global-loading", data=False),
    dcc.Store(id="forecast-phase-store", storage_type="memory"),
    *forecast_shared_stores(storage_type="memory"),
    _planning_ids_skeleton(),
    html.Div(id="app-wrapper", className="sidebar-collapsed", children=[
        html.Div(id="sidebar", children=sidebar_component(False).children),
        html.Div(id="root")
    ]),
    # Global loading overlay
    global_loading_overlay(),
    dcc.Interval(id="cap-plans-refresh", interval=5000, n_intervals=0)
])


# ---- Validation Layout (verbatim) ----
# ✅ VALIDATION LAYOUT — include plan-detail skeleton
app.validation_layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed"),
    dcc.Store(id="ws-status"),
    dcc.Store(id="ws-selected-ba"),
    dcc.Store(id="ws-refresh"),
    dcc.Store(id="global-loading"),
    dcc.Store(id="forecast-phase-store", storage_type="memory"),
    *forecast_shared_stores(storage_type="memory"),
    header_bar(),
    planning_layout(),
    plan_detail_validation_layout(),
    # Forecasting workspace stubs for callback validation
    page_forecast_section("volume-summary"),
    page_forecast_section("smoothing-anomaly", validation_only=True),
    page_forecast_section("forecasting"),
    page_forecast_section("transformation-projects"),
    page_forecast_section("daily-interval"),
    dash_table.DataTable(id="tbl-projects"),
    # Placeholders for Home timeline callback targets so Dash can validate callbacks globally
    html.Div(id="timeline-body"),
    dbc.Collapse(id="timeline-collapse"),
    html.Button(id="timeline-toggle"),
    # Global loading overlay in validation context
    global_loading_overlay(),
    # Minimal placeholders for callbacks that toggle global loading (keeps validation happy)
    html.Div(
        [
            html.Button(id="vs-run-btn"),
            html.Button(id="sa-run-smoothing"),
            html.Button(id="sa-run-prophet"),
            html.Button(id="fc-run-btn"),
            html.Button(id="fc-load-phase1"),
            html.Button(id="fc-load-saved-btn"),
            html.Button(id="tp-apply-selection"),
            html.Button(id="tp-apply-transform"),
            html.Button(id="di-run-btn"),
            html.Div(id="vs-alert"),
            html.Div(id="sa-alert"),
            html.Div(id="fc-alert"),
            html.Div(id="tp-selection-status"),
            html.Div(id="tp-transform-status"),
            html.Div(id="di-run-status"),
        ],
        style={"display": "none"},
    ),
])

# Global loading overlay visibility toggler
@app.callback(Output("global-loading-overlay", "style"), Input("global-loading", "data"))
def _toggle_global_loading(is_on):
    base = GLOBAL_LOADING_STYLE.copy()
    try:
        if bool(is_on):
            base["display"] = "flex"
        else:
            base["display"] = "none"
    except Exception:
        base["display"] = "none"
    return base

# Show/hide global overlay during navigation (avoid race conditions).
@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("url-router", "pathname"),
    Input("root", "children"),
    prevent_initial_call=True,
)
def _toggle_nav_loading(_path, _children):
    triggered = [t["prop_id"].split(".")[0] for t in (dash.callback_context.triggered or [])]
    if "root" in triggered:
        return False
    if "url-router" in triggered:
        return True
    return False

# Register callbacks (planning + plan-detail)
register_planning_ws(app)
register_plan_detail(app)

# Auto-lock previous months' plans at startup (idempotent)
try:
    auto_lock_previous_month_plans()
except Exception:
    pass

# ---- Entrypoint (verbatim) ----
# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(debug=True)
