from __future__ import annotations
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from common import header_bar

MODELS = [
    {
        "title": "Random Forest",
        "icon": "🌳",
        "content": """
Aggregates many decision trees for prediction.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
f(x) = (1 / B ) Σ Tₖ(x)
</span>
"""
    },
    {
        "title": "Prophet",
        "icon": "📅",
        "content": """
Handles trend, seasonality, and holidays.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
yₜ = gₜ + sₜ + hₜ + eₜ
</span>
"""
    },
    {
        "title": "XGBoost",
        "icon": "⚡",
        "content": """
Gradient boosting framework for high performance.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
ŷᵢ = Σ fₖ(xᵢ) <br>
Obj(θ) = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)
</span>
"""
    },
    {
        "title": "ARIMA",
        "icon": "📘",
        "content": """
Combines autoregression and moving average.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
AR(p): yₜ = c + Φ₁yₜ₋₁ + ... + Φₚyₜ₋ₚ <br>
MA(q): yₜ = α + θ₁eₜ₋₁ + ... + θq eₜ₋q <br>
ARIMA(p,d,q): differencing d times then ARMA(p,q)
</span>
"""
    },
    {
        "title": "Triple Exponential Smoothing (Holt-Winters)",
        "icon": "📉",
        "content": """
Captures level, trend, and seasonality.<br><br>
<strong style="color:#000000;">Equations:</strong><br>
<span style="color:#000000; font-size:22px;">
Level: lₜ = βyₜ + (1−β)(lₜ₋₁ + bₜ₋₁)<br>
Trend: bₜ = β(lₜ − lₜ₋₁) + (1 − β)bₜ₋₁<br>
Seasonality: sₜ = γ(yₜ / lₜ) + (1 − γ)sₜ₋ₘ<br>
Forecast: yₜ₊₁ = lₜ + bₜ + sₜ₊₁
</span>
"""
    },
    {
        "title": "Double Exponential Smoothing (Holt’s)",
        "icon": "📊",
        "content": """
Captures level and trend.<br><br>
<strong style="color:#000000;">Equations:</strong><br>
<span style="color:#000000; font-size:22px;">
Level: lₜ = βyₜ + (1−β)(lₜ₋₁)<br>
Trend: bₜ = β(lₜ − lₜ₋₁) + (1 − β)bₜ₋₁<br>
Forecast: yₜ₊₁ = lₜ + bₜ
</span>
"""
    },
    {
        "title": "Single Exponential Smoothing",
        "icon": "🔹",
        "content": """
Simple smoothing method.<br><br>
<strong style="color:#000000;">Equations:</strong><br>
<span style="color:#000000; font-size:22px;">
lₜ = βyₜ + (1 − β)lₜ₋₁<br>
Forecast: yₜ₊₁ = lₜ
</span>
"""
    },
    {
        "title": "Linear Regression",
        "icon": "📐",
        "content": """
Predicts using a linear combination of features.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
ŷ = β0 + β1x₁ + β2x₂ + ... + βₖxₖ<br>
RSS = Σ (yᵢ − ŷᵢ)²
</span>
"""
    },
    {
        "title": "Weighted Moving Average",
        "icon": "📘",
        "content": """
Forecasts using average of past observations.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
ŷₜ = Σ (wᵢ × yₜ₋ᵢ), where Σ wᵢ = 1
</span>
"""
    },
]

FORECAST_NAV = [
    {"slug": "volume-summary", "label": "Volume Summary", "emoji": "📊"},
    {"slug": "transformation-projects", "label": "Transformation Projects", "emoji": "⚙️"},
    {"slug": "daily-interval", "label": "Daily Interval Forecast", "emoji": "⏱️"},
]

FORECAST_STEPS = [
    {"slug": "volume-summary", "label": "Volume Summary", "emoji": "📊", "next": "transformation-projects"},
    {"slug": "transformation-projects", "label": "Transformation Projects", "emoji": "⚙️", "next": "daily-interval"},
    {"slug": "daily-interval", "label": "Daily Interval Forecast", "emoji": "⏱️", "next": None},
]

def _model_card(title: str, content: str, icon: str):
    return html.Div(
        className="forecast-card",
        children=[
            html.Div(f"{icon} {title}", className="forecast-card-title"),
            dcc.Markdown(
                content,
                dangerously_allow_html=True,
                className="forecast-card-content"
            ),
        ],
    )

def _nav_buttons():
    return dbc.Row(
        [
            dbc.Col(
                dcc.Link(
                    dbc.Button(
                        f"{item['emoji']} {item['label']}",
                        color="secondary",
                        outline=True,
                        className="w-100",
                    ),
                    href=f"/forecast/{item['slug']}",
                    style={"textDecoration": "none"},
                ),
                xs=12,
                sm=6,
                md=12,
                lg=3,
            )
            for item in FORECAST_NAV
        ],
        className="g-2 mb-3",
    )

def page_forecast():
    return html.Div(
        dbc.Container(
            [
                header_bar(),
                html.Div(
                    children=[
                        html.H1(
                            "🔮 Power of 9 Models: A complete suite for Forecasting",
                            className="forecast-heading",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Your guided path through forecasting.",
                                    className="text-muted small",
                                ),
                                dcc.Link(
                                    dbc.Button("→ Volume Summary", color="primary", outline=False, className="float-end"),
                                    href="/forecast/volume-summary",
                                    style={"textDecoration": "none"},
                                ),
                            ],
                            className="d-flex justify-content-between align-items-center mb-3",
                        ),
                        html.Div(
                            children=[
                                _model_card(m["title"], m["content"], m["icon"])
                                for m in MODELS
                            ],
                            className="forecast-grid",
                        ),
                    ],
                    className="forecast-page",
                ),
            ],
            fluid=True,
        )
    )

def forecast_shared_stores(storage_type: str = "memory"):
    return [
        dcc.Store(id="vs-data-store", storage_type=storage_type),
        dcc.Store(id="vs-results-store", storage_type=storage_type),
        dcc.Store(id="vs-iq-store", storage_type=storage_type),
        dcc.Store(id="vs-iq-summary-store", storage_type=storage_type),
        dcc.Store(id="vs-seasonality-store", storage_type=storage_type),
        dcc.Store(id="vs-prophet-store", storage_type=storage_type),
        dcc.Store(id="vs-holiday-store", storage_type=storage_type),
        dcc.Store(id="vs-phase1-config-store", storage_type=storage_type),
        dcc.Store(id="vs-phase1-download-store", storage_type=storage_type),
        dcc.Store(id="vs-phase2-store", storage_type=storage_type),
        dcc.Store(id="vs-adjusted-store", storage_type=storage_type),
        dcc.Store(id="sa-raw-store", storage_type=storage_type),
        dcc.Store(id="sa-results-store", storage_type=storage_type),
        dcc.Store(id="sa-seasonality-store", storage_type=storage_type),
        dcc.Store(id="fc-data-store", storage_type=storage_type),
        dcc.Store(id="fc-saved-list-store", storage_type=storage_type),
        dcc.Store(id="fc-saved-selected", storage_type=storage_type),
        dcc.Store(id="fc-config-store", storage_type=storage_type),
        dcc.Store(id="tp-raw-store", storage_type=storage_type),
        dcc.Store(id="tp-filtered-store", storage_type=storage_type),
        dcc.Store(id="tp-final-store", storage_type=storage_type),
        dcc.Store(id="di-page-init", data=0, storage_type=storage_type),
        dcc.Store(id="di-forecast-dates-store", storage_type=storage_type),
        dcc.Store(id="di-holidays-store", storage_type=storage_type),
        dcc.Store(id="di-transform-store", storage_type=storage_type),
        dcc.Store(id="di-interval-store", storage_type=storage_type),
        dcc.Store(id="di-results-store", storage_type=storage_type),
        dcc.Store(id="di-distribution-store", storage_type=storage_type),
        dcc.Store(id="di-staged-run-id", storage_type=storage_type),
    ]

def page_forecast_section(slug: str, validation_only: bool = False):
    """Forecasting workspace wizard-style sections."""
    def _stepper(active_slug: str):
        items = []
        for step in FORECAST_STEPS:
            is_active = step["slug"] == active_slug
            items.append(
                html.Div(
                    [
                        dbc.Badge(f"{step['emoji']} {step['label']}", color="primary" if is_active else "secondary", pill=True, className="me-2"),
                        html.Span("→", className="text-muted me-2") if step["next"] else None,
                    ],
                    className="d-flex align-items-center",
                )
            )
        return html.Div(items, className="d-flex flex-wrap gap-2 mb-3")

    def _section_heading(step_meta):
        prev_item = None
        for i, s in enumerate(FORECAST_STEPS):
            if s["slug"] == step_meta["slug"] and i > 0:
                prev_item = FORECAST_STEPS[i - 1]
                break
        prev_href = f"/forecast/{prev_item['slug']}" if prev_item else "/forecast"
        prev_label = f"← {prev_item['label']}" if prev_item else "Back to models"
        return html.Div(
            [
                html.Div(
                    [
                        html.H2(f"{step_meta['emoji']} {step_meta['label']}", className="mb-1"),
                        html.Div("Follow the steps to keep moving forward.", className="text-muted"),
                    ],
                ),
                dcc.Link(
                    dbc.Button(prev_label, color="link", className="px-0"),
                    href=prev_href,
                ),
            ],
            className="d-flex justify-content-between align-items-start mb-3 flex-wrap gap-2",
        )

    def _table(id: str, page_size: int = 10, editable: bool = False):
        base_style = {
            "overflowX": "auto",
            "overflowY": "auto",
            "border": "1px solid #e9ecef",
            "borderRadius": "8px",
            "padding": "4px",
            "maxHeight": "360px",
        }
        base_cells = {
            "fontSize": 12,
            "padding": "6px 8px",
            "textAlign": "left",
            "border": "none",
            "minWidth": "120px",
            "width": "120px",
            
            "whiteSpace": "nowrap",
        }
        return dash_table.DataTable(
            id=id,
            data=[],
            columns=[],
            page_size=page_size,
            page_action="none",
            editable=editable,
            fixed_rows={"headers": True},
            style_table=base_style,
            style_cell=base_cells,
            style_header={
                "backgroundColor": "#f8f9fa",
                "fontWeight": "600",
                "border": "none",
            },
        )

    def _volume_summary_layout(step_meta):
        return html.Div(
            [
                _section_heading(step_meta),
                _stepper(step_meta["slug"]),
                dbc.Row(
                    [
                        dbc.Col(
                            html.H4("Upload data"),
                            className="EB21"
                        ),
                        dbc.Col(
                            dcc.Upload(
                                id="vs-upload",
                                children=html.Div(["Drag & drop or ", html.Strong("select a CSV/XLSX file")]),
                                multiple=False,
                                accept=".csv,.xls,.xlsx,.xlsm",
                                className="border border-secondary rounded p-3 mb-2 bg-light",
                            ),
                            className="EB21"
                        ), 
                        dbc.Col(
                            dbc.Button("Run Volume Summary", id="vs-run-btn", color="primary", className="mt-2"),
                            className="EB21"
                        ),
                    ],
                    className="EB2"
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Alert(id="vs-alert", color="info", is_open=False, className="mt-2"),
                    ),
                ),  
                dbc.Row(
                    dbc.Col(
                        html.Div(id="vs-upload-msg", className="small text-muted"),
                    ),
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Preview"),
                                _table("vs-preview", page_size=8),
                                html.Hr(),
                                html.H5("Summary"),
                                _table("vs-summary", page_size=10),
                                html.Hr(),
                                html.H5("IQ Summary"),
                                html.Div(
                                    "Upload an Excel file with an IQ_Data sheet to populate these tables.",
                                    className="small text-muted mb-1",
                                ),
                                _table("vs-iq-summary", page_size=8),
                                html.Hr(),
                                html.H5("Volume Summary"),
                                _table("vs-volume-summary", page_size=8),
                                html.Hr(),
                                html.H5("Contact Ratio Summary (Volume/IQ)"),
                                _table("vs-contact-summary", page_size=8),
                                html.Hr(),
                                html.Div(
                                    [
                                        html.H5("Category", className="mb-1"),
                                        dcc.Tabs(
                                            id="vs-category-tabs",
                                            value=None,
                                            children=[],
                                            className="vs-category-tabs",
                                            parent_className="vs-category-tabs-wrapper",
                                        ),
                                        html.Div(id="vs-category-heading", className="small text-muted mt-1"),
                                    ],
                                    className="mb-2",
                                ),
                                html.H5(id="vs-pivot-title", className="mb-1"),
                                _table("vs-pivot", page_size=10),
                                html.Hr(),
                                html.H5(id="vs-volume-split-title"),
                                _table("vs-volume-split", page_size=10),
                                html.Hr(),
                                html.H5("Seasonality (Based on Contact Ratio)"),
                                dcc.Graph(id="vs-seasonality-chart", figure={}, className="mb-2"),
                                _table("vs-seasonality-table", page_size=6),
                                html.Hr(),
                                html.H5("Seasonality Cap Adjustment"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Lower Cap"),
                                                    dbc.Input(id="vs-lower-cap", type="number", value=0.8, step=0.05),
                                                ]
                                            ),
                                            md=6,
                                        ),
                                        dbc.Col(
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Upper Cap"),
                                                    dbc.Input(id="vs-upper-cap", type="number", value=1.15, step=0.05),
                                                ]
                                            ),
                                            md=6,
                                        ),
                                    ],
                                    className="g-2",
                                ),
                                html.Div("Seasonality Capped Adjustment Table", className="small text-muted mt-2"),
                                dash_table.DataTable(
                                    id="vs-capped-editor",
                                    data=[],
                                    columns=[],
                                    page_size=6,
                                    page_action="none",
                                    editable=True,
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Updated Seasonality with Recalculated Avg"),
                                _table("vs-recalc-table", page_size=6),
                                dcc.Graph(id="vs-capped-chart", figure={}, className="mb-2"),
                                html.Hr(),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Custom Base Volume"),
                                        dbc.Input(id="vs-base-volume", type="number", step=1),
                                    ],
                                    className="mb-2",
                                ),
                                html.H5("Normalized Ratio"),
                                _table("vs-normalized-table", page_size=6),
                                dbc.Button("Apply Changes", id="vs-apply-seasonality", color="primary", className="mt-2"),
                                html.Div(id="vs-seasonality-status", className="small text-muted mt-2"),
                                html.Hr(),
                                html.H4("Prophet Smoothing"),
                                dbc.Button("Run Prophet Smoothing", id="vs-run-prophet", color="secondary", className="mb-2"),
                                html.Div(id="vs-prophet-status", className="small text-muted mb-2"),
                                html.H5("Data After Prophet Smoothing"),
                                dash_table.DataTable(
                                    id="vs-prophet-table",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    page_action="none",
                                    editable=True,
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Normalized Contact Ratio 2"),
                                _table("vs-norm2-table", page_size=6),
                                dcc.Graph(id="vs-norm2-chart", figure={}, className="mb-2"),
                                dcc.Graph(id="vs-prophet-line", figure={}, className="mb-2"),
                                dbc.Button("Save Changes if any else skip", id="vs-save-prophet", color="primary", className="mt-2"),
                                html.Div(id="vs-prophet-save-status", className="small text-muted mt-2"),
                                html.Hr(),
                                dbc.Button("Run Phase 1", id="vs-run-phase1", color="success", className="mt-2"),
                                html.Div(id="vs-phase1-status", className="small text-muted mt-2"),
                                html.Hr(),
                                html.H5("Phase 1 - Forecast Results"),
                                _table("vs-phase1-results", page_size=8),
                                html.Hr(),
                                html.H5("Phase 1 Accuracy Before Iterations"),
                                _table("vs-phase1-accuracy", page_size=8),
                                html.Hr(),
                                html.H5("Output Iterative Tuning"),
                                _table("vs-phase1-tuning", page_size=8),
                                html.Hr(),
                                html.H5("Final Accuracy (After Tuning)"),
                                _table("vs-final-accuracy", page_size=8),
                                html.Hr(),
                                html.H5("Current Model Configurations"),
                                html.Div(id="vs-phase1-configs"),
                                dbc.Button(
                                    "Download Configuration Summary",
                                    id="vs-config-download-btn",
                                    color="secondary",
                                    className="mt-2",
                                ),
                                dcc.Download(id="vs-config-download"),
                                dbc.Button(
                                    "Download Phase 1 Results",
                                    id="vs-phase1-download-btn",
                                    color="secondary",
                                    className="mt-2 ms-2",
                                ),
                                dcc.Download(id="vs-phase1-download"),
                                html.Hr(),
                                html.H5("Phase 2 Forecast"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.DatePickerSingle(
                                                id="vs-phase2-start",
                                                placeholder="Start date",
                                            ),
                                            md=1, style={"marginRight": "105px"},
                                        ),
                                        dbc.Col(
                                            dcc.DatePickerSingle(
                                                id="vs-phase2-end",
                                                placeholder="End date",
                                            ),
                                            md=1,
                                        ),
                                    ],
                                    className="g-2 mb-2",
                                ),
                                dbc.Button("Run Phase 2 Forecast", id="vs-run-phase2", color="primary"),
                                dbc.Button(
                                    "Clear Phase 2 Cache",
                                    id="vs-clear-phase2",
                                    color="secondary",
                                    className="ms-2",
                                ),
                                html.Div(id="vs-phase2-status", className="small text-muted mt-2"),
                                html.Hr(),
                                html.H5("Phase 2 Contact Ratio Forecast"),
                                _table("vs-phase2-forecast", page_size=8),
                                html.Hr(),
                                html.H5("Phase 2 Base Forecast with Volumes"),
                                _table("vs-phase2-base", page_size=8),
                                html.Hr(),
                                html.H5("Forecast Group Summary"),
                                _table("vs-phase2-fg-summary", page_size=8),
                                html.Hr(),
                                html.H5("Volume Split (%)"),
                                _table("vs-phase2-volume-split", page_size=8),
                                html.Hr(),
                                html.H5("Edit Volume Split Allocation (Latest Year Data)"),
                                dash_table.DataTable(
                                    id="vs-volume-split-edit",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    page_action="none",
                                    editable=True,
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"},
                                    style_cell={
                                        "fontSize": 12,
                                        "minWidth": "120px",
                                        "width": "120px",
                                        
                                        "whiteSpace": "nowrap",
                                    },
                                ),
                                html.Div(id="vs-volume-split-info", className="small text-muted mt-2"),
                                dbc.Button(
                                    "Apply Volume Split to Base Forecast",
                                    id="vs-apply-volume-split",
                                    color="primary",
                                    className="mt-2",
                                ),
                                html.Hr(),
                                html.H5("Base Forecast Adjusted by Volume Split"),
                                _table("vs-adjusted-forecast", page_size=8),
                                html.Hr(),
                                html.H5("Verification"),
                                _table("vs-adjusted-verify", page_size=8),
                                html.Div(id="vs-adjusted-status", className="small text-muted mt-2"),
                                dbc.Button(
                                    "Download Adjusted Forecast by Group",
                                    id="vs-download-adjusted",
                                    color="secondary",
                                    className="mt-2",
                                ),
                                dcc.Download(id="vs-download-adjusted-file"),
                                dcc.Download(id="vs-save-adjusted-download"),
                                dbc.Button(
                                    "Save Monthly Forecast With Adjustments",
                                    id="vs-save-adjusted",
                                    color="success",
                                    className="mt-2 ms-2",
                                ),
                                html.Div(id="vs-save-adjusted-status", className="small text-muted mt-2"),
                                html.Div(
                                    dcc.Link(
                                        dbc.Button("→ Transformation Projects", color="primary"),
                                        href="/forecast/transformation-projects",
                                        style={"textDecoration": "none"},
                                    ),
                                    id="vs-next-step",
                                    className="mt-3",
                                    style={"display": "none"},
                                ),
                            ],
                            md=12,
                        ),
                    ],
                    className="g-3",
                ),
            ],
            className="forecast-page",
        )

    def _smoothing_layout(step_meta):
        return [
            dcc.Download(id="sa-download-smoothed"),
            dcc.Download(id="sa-download-seasonality"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5("Upload raw volume data", className="mb-2"),
                                html.Div(
                                    "If you already uploaded Volume Summary (Volume/IQ), we’ll reuse it automatically and run smoothing in the background. You can tweak the settings and rerun here.",
                                    className="text-muted small mb-2",
                                ),
                                dcc.Upload(
                                    id="sa-upload",
                                    children=html.Div(["Drag & drop or ", html.Strong("select CSV/XLSX (Date, Volume, IQ_value optional)")]),
                                    multiple=False,
                                    className="border border-secondary rounded p-3 mb-2 bg-light",
                                ),
                                html.Div(id="sa-upload-msg", className="small text-muted mb-2"),
                                html.Div("Smoothing & anomaly configuration", className="fw-semibold mb-2"),
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("EWMA span"), dbc.Input(id="sa-window", type="number", value=6, min=1)]), md=12),
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("Z-score threshold"), dbc.Input(id="sa-threshold", type="number", value=2.5, step=0.1)]), md=12),
                                    ],
                                    className="g-2",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("Prophet fourier"), dbc.Input(id="sa-prophet-order", type="number", value=5, min=1)]), md=12),
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("Hold-out months"), dbc.Input(id="sa-holdout", type="number", value=0, min=0)]), md=12),
                                    ],
                                    className="g-2 mt-1",
                                ),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Run Smoothing", id="sa-run-smoothing", color="primary"),
                                        dbc.Button("Prophet smoothing", id="sa-run-prophet", color="secondary", outline=True),
                                    ],
                                    className="mt-2",
                                ),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Download", id="sa-download-btn", color="secondary", outline=True),
                                        dbc.Button("Save to disk", id="sa-save-btn", color="secondary", outline=True),
                                        dbc.Button("Send to Phase 2", id="sa-send-phase2", color="info"),
                                    ],
                                    className="mt-2",
                                ),
                                dbc.Alert(id="sa-alert", is_open=False, color="info", className="mt-2"),
                                html.Div(id="sa-save-status", className="small text-muted mt-1"),
                                html.Div(id="sa-phase-status", className="small fw-semibold text-primary mt-2"),
                            ],
                            body=True,
                        ),
                        md=12,
                    ),
                    dbc.Col(
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    label="Series & anomalies",
                                    tab_id="series",
                                    children=[
                                        dcc.Graph(id="sa-smooth-chart", figure={}, className="mb-2"),
                                        dash_table.DataTable(
                                            id="sa-anomaly-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            page_action="none",
                                            fixed_rows={"headers": True},
                                            style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                            style_cell={
                                                "fontSize": 12,
                                                "minWidth": "120px",
                                                "width": "120px",
                                                
                                                "whiteSpace": "nowrap",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            id="sa-smooth-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            page_action="none",
                                            fixed_rows={"headers": True},
                                            style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                            style_cell={
                                                "fontSize": 12,
                                                "minWidth": "120px",
                                                "width": "120px",
                                                
                                                "whiteSpace": "nowrap",
                                            },
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Seasonality & ratios",
                                    tab_id="seasonality",
                                    children=[
                                        html.Div(
                                            "Normalized ratio views. Edit the capped seasonality table to tweak the curve.",
                                            className="small text-muted mb-2",
                                        ),
                                        dcc.Graph(id="sa-norm1-chart", figure={}, className="mb-2"),
                                        dcc.Graph(id="sa-norm2-chart", figure={}, className="mb-2"),
                                        dash_table.DataTable(
                                            id="sa-ratio-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            page_action="none",
                                            fixed_rows={"headers": True},
                                            style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                            style_cell={
                                                "fontSize": 12,
                                                "minWidth": "120px",
                                                "width": "120px",
                                                
                                                "whiteSpace": "nowrap",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            id="sa-seasonality-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            editable=True,
                                            page_action="none",
                                            fixed_rows={"headers": True},
                                            style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                            style_cell={
                                                "fontSize": 12,
                                                "minWidth": "120px",
                                                "width": "120px",
                                                
                                                "whiteSpace": "nowrap",
                                            },
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Raw preview",
                                    tab_id="raw",
                                    children=[
                                        dash_table.DataTable(
                                            id="sa-preview",
                                            data=[],
                                            columns=[],
                                            page_size=8,
                                            page_action="none",
                                            fixed_rows={"headers": True},
                                            style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                            style_cell={
                                                "fontSize": 12,
                                                "minWidth": "120px",
                                                "width": "120px",
                                                
                                                "whiteSpace": "nowrap",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                            active_tab="series",
                        ),
                        md=12,
                    ),
                ],
                className="g-3",
            ),
        ]

    def _forecasting_layout(step_meta):
        return [
            dcc.Download(id="fc-download-forecast"),
            dcc.Download(id="fc-download-config"),
            dcc.Download(id="fc-download-accuracy"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Smoothed data upload"),
                            dcc.Upload(
                                id="fc-upload",
                                children=html.Div(["Drag & drop or ", html.Strong("select smoothed CSV/XLSX (ds, Final_Smoothed_Value, IQ_value)")]),
                                multiple=False,
                                className="border border-secondary rounded p-3 mb-2 bg-light",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Forecast months"),
                                    dbc.Input(id="fc-months", type="number", value=12, min=1, step=1),
                                ],
                                className="mb-2",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Run Forecast", id="fc-run-btn", color="primary"),
                                    dbc.Button("Load Phase 1 data", id="fc-load-phase1", color="secondary", outline=True),
                                ],
                                className="mb-2",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Download results", id="fc-download-btn", color="secondary", outline=True),
                                    dbc.Button("Download config", id="fc-download-config-btn", color="secondary", outline=True),
                                    dbc.Button("Save to disk", id="fc-save-btn", color="secondary", outline=True),
                                    dbc.Button("Save to workspace", id="fc-save-db-btn", color="success"),
                                ],
                                className="mb-2",
                            ),
                            html.Div(id="fc-save-status", className="small text-muted mb-1"),
                            html.Div(id="fc-save-db-status", className="small text-muted text-success mb-2"),
                            dbc.Alert(id="fc-alert", color="info", is_open=False),
                            html.Div(id="fc-errors", className="text-danger small mt-2"),
                            html.Div(id="fc-phase2-meta", className="small fw-semibold text-primary mt-2"),
                            html.Hr(),
                            html.H5("Saved forecasts"),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Refresh list", id="fc-refresh-saved", color="secondary", outline=True),
                                    dbc.Button("Load selected", id="fc-load-saved-btn", color="primary", outline=True),
                                ],
                                className="mb-2",
                            ),
                            html.Div(id="fc-saved-status", className="small text-muted mb-1"),
                            dash_table.DataTable(
                                id="fc-saved-table",
                                data=[],
                                columns=[],
                                page_size=6,
                                row_selectable="single",
                                page_action="none",
                                fixed_rows={"headers": True},
                                style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                style_cell={"fontSize": 12},
                            ),
                            html.Hr(),
                            html.H5("Model configuration"),
                            dcc.Interval(id="fc-config-loader", interval=0, n_intervals=0, max_intervals=1),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("ChangePoint Prior"), dbc.Input(id="fc-prophet-cps", type="number", step=0.01)]), md=12),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Seasonality Prior"), dbc.Input(id="fc-prophet-sps", type="number", step=0.01)]), md=12),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Holiday Prior"), dbc.Input(id="fc-prophet-hps", type="number", step=0.01)]), md=12),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Yearly Fourier Order"), dbc.Input(id="fc-prophet-fourier", type="number", step=1)]), md=12),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-prophet-holidays", label="Use holidays"), md=12),
                                                    dbc.Col(dbc.Checkbox(id="fc-prophet-iq", label="Use IQ scaled"), md=12),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="Prophet",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("n_estimators"), dbc.Input(id="fc-rf-n", type="number", step=10)]), md=12),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("max_depth"), dbc.Input(id="fc-rf-depth", type="number", step=1)]), md=12),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-rf-holidays", label="Use holidays"), md=12),
                                                    dbc.Col(dbc.Checkbox(id="fc-rf-iq", label="Use IQ scaled"), md=12),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="Random Forest",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("n_estimators"), dbc.Input(id="fc-xgb-n", type="number", step=10)]), md=12),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("learning_rate"), dbc.Input(id="fc-xgb-lr", type="number", step=0.01)]), md=12),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("max_depth"), dbc.Input(id="fc-xgb-depth", type="number", step=1)]), md=12),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-xgb-holidays", label="Use holidays"), md=12),
                                                    dbc.Col(dbc.Checkbox(id="fc-xgb-iq", label="Use IQ scaled"), md=12),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="XGBoost",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.InputGroup([dbc.InputGroupText("Lags"), dbc.Input(id="fc-var-lags", type="number", step=1)]),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-var-holidays", label="Use holidays"), md=12),
                                                    dbc.Col(dbc.Checkbox(id="fc-var-iq", label="Use IQ scaled"), md=12),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="VAR",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.InputGroup([dbc.InputGroupText("order (p,d,q)"), dbc.Input(id="fc-sarimax-order", type="text")], className="mb-2"),
                                            dbc.InputGroup([dbc.InputGroupText("seasonal (P,D,Q,s)"), dbc.Input(id="fc-sarimax-seasonal", type="text")], className="mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-sarimax-holidays", label="Use holidays"), md=12),
                                                    dbc.Col(dbc.Checkbox(id="fc-sarimax-iq", label="Use IQ scaled"), md=12),
                                                ],
                                                className="g-2",
                                            ),
                                        ],
                                        title="SARIMAX",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checkbox(id="fc-general-seasonality", label="Use seasonality"),
                                        ],
                                        title="General",
                                    ),
                                ],
                                start_collapsed=True,
                                className="mb-2",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Save Config", id="fc-config-save", color="success"),
                                    dbc.Button("Reset Defaults", id="fc-config-reset", color="secondary"),
                                ],
                                className="mb-2",
                            ),
                            dbc.Alert(id="fc-config-status", color="info", is_open=False),
                        ],
                        md=12,
                    ),
                    dbc.Col(
                        [
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Forecast tables",
                                        tab_id="tables",
                                        children=[
                                            dcc.Graph(id="fc-line-chart", figure={}, className="mb-2"),
                                            dash_table.DataTable(
                                                id="fc-combined",
                                                data=[],
                                                columns=[],
                                                page_size=8,
                                                page_action="none",
                                                fixed_rows={"headers": True},
                                                style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                                style_cell={"fontSize": 12},
                                            ),
                                            html.Hr(),
                                            dash_table.DataTable(
                                                id="fc-pivot",
                                                data=[],
                                                columns=[],
                                                page_size=8,
                                                page_action="none",
                                                fixed_rows={"headers": True},
                                                style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                                style_cell={"fontSize": 12},
                                            ),
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="Normalized ratios",
                                        tab_id="ratios",
                                        children=[
                                            dcc.Graph(id="fc-ratio1-chart", figure={}, className="mb-2"),
                                            dcc.Graph(id="fc-ratio2-chart", figure={}, className="mb-2"),
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="Accuracy",
                                        tab_id="accuracy",
                                        children=[
                                            dcc.Graph(id="fc-accuracy-chart", figure={}, className="mb-2"),
                                            dash_table.DataTable(
                                                id="fc-accuracy-table",
                                                data=[],
                                                columns=[],
                                                page_size=8,
                                                page_action="none",
                                                fixed_rows={"headers": True},
                                                style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                                style_cell={"fontSize": 12},
                                            ),
                                        ],
                                    ),
                                ],
                                active_tab="tables",
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="g-3",
            ),
        ]

    def _transformation_layout(step_meta):
        return [
            dcc.Download(id="tp-download-final"),
            dcc.Download(id="tp-download-full"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5("Load forecast data", className="mb-2"),
                                dcc.Dropdown(
                                    id="tp-saved-run",
                                    options=[],
                                    placeholder="Select saved forecast",
                                    className="mt-2",
                                ),
                                html.Div(id="tp-saved-status", className="small text-muted mt-2"),
                                html.Div(id="tp-load-status", className="small text-muted mt-2"),
                                html.Hr(),
                                html.H6("Selection criteria"),
                                dcc.Dropdown(
                                    id="tp-group",
                                    options=[],
                                    placeholder="Select forecast group",
                                    className="mb-2",
                                ),
                                dcc.Dropdown(
                                    id="tp-model",
                                    options=[],
                                    placeholder="Select model",
                                    className="mb-2",
                                ),
                                dcc.Dropdown(
                                    id="tp-year",
                                    options=[],
                                    placeholder="Select year",
                                    className="mb-2",
                                ),
                                dbc.Button(
                                    "Apply selection & load data",
                                    id="tp-apply-selection",
                                    color="primary",
                                    className="mb-2",
                                ),
                                dbc.Alert(
                                    id="tp-selection-status",
                                    color="info",
                                    is_open=False,
                                    className="mb-2",
                                ),
                                html.Hr(),
                                html.H6("Transformations"),
                                dbc.Button(
                                    "Apply transformations",
                                    id="tp-apply-transform",
                                    color="success",
                                    className="mb-2",
                                ),
                                html.Div(id="tp-transform-status", className="small text-success mb-2"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Download final forecast",
                                            id="tp-download-final-btn",
                                            color="secondary",
                                            outline=True,
                                        ),
                                        dbc.Button(
                                            "Download full detail",
                                            id="tp-download-full-btn",
                                            color="secondary",
                                            outline=True,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.Button(
                                    "Save to disk",
                                    id="tp-save-btn",
                                    color="secondary",
                                    outline=True,
                                    className="mb-2",
                                ),
                                html.Div(id="tp-save-status", className="small text-muted mb-2"),
                                dbc.Button(
                                    "Reset",
                                    id="tp-reset",
                                    color="light",
                                    className="w-100",
                                ),
                            ],
                            body=True,
                        ),
                        md=12,
                    ),
                    dbc.Col(
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    label="Raw data",
                                    tab_id="tp-raw",
                                    children=[
                                        html.H5("Raw forecast preview", className="mb-2"),
                                        _table("tp-raw-table", page_size=8),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Filtered data",
                                    tab_id="tp-filtered",
                                    children=[
                                        html.H5("Filtered preview", className="mb-2"),
                                        _table("tp-filtered-table", page_size=8),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Edit transformations",
                                    tab_id="tp-edit",
                                    children=[
                                        html.H5("Transformation editor", className="mb-2"),
                                        html.Div(
                                            "Enter % change values (e.g., 10 for +10%). Base columns are locked.",
                                            className="small text-muted mb-2",
                                        ),
                                        _table("tp-transform-table", page_size=10, editable=True),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Results",
                                    tab_id="tp-results",
                                    children=[
                                        html.H5("Transposed view", className="mb-2"),
                                        _table("tp-transposed-table", page_size=8),
                                        html.Hr(),
                                        html.H5("Final forecast", className="mb-2"),
                                        _table("tp-final-forecast-table", page_size=8),
                                        html.Hr(),
                                        html.H5("Summary", className="mb-2"),
                                        _table("tp-summary-table", page_size=6),
                                    ],
                                ),
                            ],
                            active_tab="tp-raw",
                        ),
                        md=12,
                    ),
                ],
                className="g-3",
            ),
        ]

    def _daily_interval_layout(step_meta):
        return [
            dcc.Download(id="di-download-daily"),
            dcc.Download(id="di-download-interval"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5("Forecast dates", className="mb-2"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Refresh forecast dates", id="di-refresh-forecast-dates", color="secondary", outline=True),
                                    ],
                                    className="mb-2",
                                ),
                                html.Div(id="di-forecast-dates-msg", className="small text-muted mb-2"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="di-forecast-year",
                                                options=[],
                                                placeholder="Select year",
                                            ),
                                            md=12,
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="di-forecast-month",
                                                options=[],
                                                placeholder="Select month",
                                            ),
                                            md=12,
                                        ),
                                    ],
                                    className="g-2 mb-2",
                                ),
                                html.Hr(),
                                html.H5("Use transformed forecast", className="mb-2"),
                                dcc.Dropdown(
                                    id="di-saved-run",
                                    options=[],
                                    placeholder="Select transformed forecast",
                                    className="mb-2",
                                ),
                                html.Div(id="di-saved-status", className="small text-muted mb-2"),
                                html.Div(id="di-transform-msg", className="small text-muted mb-2"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="di-transform-group",
                                                options=[],
                                                placeholder="Select forecast group",
                                            ),
                                            md=12,
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="di-transform-model",
                                                options=[],
                                                placeholder="Select model",
                                            ),
                                            md=12,
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="di-transform-month",
                                                options=[],
                                                placeholder="Select month",
                                            ),
                                            md=12,
                                        ),
                                    ],
                                    className="g-2 mb-2",
                                ),
                                html.Hr(),
                                html.H5("Interval history (last 3 months)", className="mb-2"),
                                html.Div(
                                    "Uses the Volume Summary upload if available; upload only to override.",
                                    className="small text-muted mb-2",
                                ),
                                dcc.Upload(
                                    id="di-upload",
                                    children=html.Div(["Drag & drop or ", html.Strong("select interval CSV/XLSX")]),
                                    multiple=False,
                                    accept=".csv,.xls,.xlsx",
                                    className="border border-secondary rounded p-3 mb-2 bg-light",
                                ),
                                html.Div(id="di-upload-msg", className="small text-muted"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Build daily + interval", id="di-run-btn", color="primary"),
                                        dbc.Button("Apply distribution edits", id="di-apply-distribution-btn", color="secondary", outline=True),
                                        dbc.Button("Download daily", id="di-download-daily-btn", color="secondary", outline=True),
                                        dbc.Button("Download interval", id="di-download-interval-btn", color="secondary", outline=True),
                                    ],
                                    className="mt-2",
                                ),
                                html.Div(id="di-run-status", className="small fw-semibold text-primary mt-2"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Save to disk", id="di-save-btn", color="secondary", outline=True),
                                    ],
                                    className="mt-2",
                                ),
                                html.Div(id="di-save-status", className="small text-muted mt-2"),
                                html.Div(
                                    [
                                        html.Hr(),
                                        html.H5("Push to capacity plan", className="mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Dropdown(id="di-push-ba", options=[], placeholder="Select BA"),
                                                    md=12,
                                                ),
                                                dbc.Col(
                                                    dcc.Dropdown(id="di-push-sba", options=[], placeholder="Select Sub BA"),
                                                    md=12,
                                                ),
                                                dbc.Col(
                                                    dcc.Dropdown(id="di-push-channel", options=[], placeholder="Select Channel"),
                                                    md=12,
                                                ),
                                                dbc.Col(
                                                    dcc.Dropdown(id="di-push-site", options=[], placeholder="Select Site"),
                                                    md=12,
                                                ),
                                            ],
                                            className="g-2 mb-2",
                                        ),
                                        dbc.Button("Push forecast to plan", id="di-push-btn", color="primary"),
                                        html.Div(id="di-push-status", className="small text-muted mt-2"),
                                    ],
                                    id="di-push-section",
                                    style={"display": "none"},
                                ),
                                html.Div(
                                    [
                                        html.Div("Tips:", className="fw-semibold mb-1"),
                                        html.Ul(
                                            [
                                                html.Li("Date column is auto-detected (Date/ds/timestamp)."),
                                                html.Li("Interval like 09:00-09:30 or 09:00 works."),
                                                html.Li("Volume column aliases: volume, calls, items, count."),
                                                html.Li("Optional AHT column: aht, aht_sec, avg_handle_time."),
                                            ],
                                            className="small mb-0",
                                        ),
                                    ],
                                    className="mt-3",
                                ),
                            ],
                            body=True,
                        ),
                        md=12,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5("Preview (first 200 rows)", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-transform-preview",
                                    data=[],
                                    columns=[],
                                    page_size=6,
                                    page_action="none",
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Interval history preview (first 200)", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-preview",
                                    data=[],
                                    columns=[],
                                    page_size=6,
                                    page_action="none",
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Daily distribution (editable)", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-distribution-table",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    page_action="none",
                                    fixed_rows={"headers": True},
                                    editable=True,
                                    style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Div(id="di-distribution-msg", className="small text-muted mt-2"),
                                html.Hr(),
                                html.H5("Daily forecast", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-daily-table",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    page_action="none",
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Interval forecast", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-interval-forecast-table",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    page_action="none",
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.Div(id="di-analysis-block"),
                            ],
                            body=True,
                        ),
                        md=12,
                    ),
                ],
                className="g-3",
            ),
        ]

    def _generic_step_layout(step_meta, body, button_id, next_slug):
        has_next = bool(next_slug)
        next_label = next_slug.replace("-", " ").title() if has_next else "Forecast Home"
        next_href = f"/forecast/{next_slug}" if has_next else "/forecast"
        modal_body = f"Move to {next_label}?" if has_next else "Return to the Forecast overview?"
        next_btn_text = "Next →" if has_next else "Back to Forecast"
        save_btn_text = "Save & Continue" if has_next else "Save"
        return html.Div(
            [
                _section_heading(step_meta),
                _stepper(step_meta["slug"]),
                html.Div(body, className="mb-3"),
                dbc.Button(save_btn_text, id=button_id, color="primary"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Ready for the next step"),
                        dbc.ModalBody(modal_body),
                        dbc.ModalFooter(
                            [
                                dbc.Button("Stay", id=f"{button_id}-close", className="me-2"),
                                dcc.Link(
                                    dbc.Button(next_btn_text, color="primary"),
                                    href=next_href,
                                    style={"textDecoration": "none"},
                                ),
                            ]
                        ),
                    ],
                    id=f"{button_id}-modal",
                    is_open=False,
                ),
            ],
            className="forecast-page",
        )

    if slug == "smoothing-anomaly" and validation_only:
        step_meta = {
            "slug": "smoothing-anomaly",
            "label": "Smoothing & Anomaly Detection",
            "emoji": "🧹",
            "next": "transformation-projects",
        }
        content = _generic_step_layout(step_meta, _smoothing_layout(step_meta), "sa-complete", "transformation-projects")
        return html.Div(
            dbc.Container(
                [
                    header_bar(),
                    content,
                ],
                fluid=True,
            )
        )

    step_meta = next((s for s in FORECAST_STEPS if s["slug"] == slug), None)
    if not step_meta:
        item = next((i for i in FORECAST_NAV if i["slug"] == slug), None)
        title = item["label"] if item else slug.replace("-", " ").title()
        emoji = item["emoji"] if item else "🧭"
        if slug == "smoothing-anomaly":
            return html.Div(
                dbc.Container(
                    [
                        header_bar(),
                        html.Div(
                            children=[
                                _nav_buttons(),
                                html.H1(f"{emoji} {title}", className="forecast-heading"),
                                dbc.Alert(
                                    "Smoothing & anomaly detection now runs automatically after Volume Summary upload. "
                                    "Continue directly to Transformation Projects.",
                                    color="info",
                                ),
                                dcc.Link(
                                    dbc.Button("Go to Transformation Projects →", color="primary"),
                                    href="/forecast/transformation-projects",
                                    style={"textDecoration": "none"},
                                ),
                            ],
                            className="forecast-page",
                        ),
                    ],
                    fluid=True,
                )
            )
        return html.Div(
            dbc.Container(
                [
                    header_bar(),
                    html.Div(
                        children=[
                            _nav_buttons(),
                            html.H1(f"{emoji} {title}", className="forecast-heading"),
                            dbc.Alert(
                                "This forecasting workspace page is under construction. "
                                "Once available, forecasts saved here can be pushed directly into planning.",
                                color="info",
                            ),
                        ],
                        className="forecast-page",
                    ),
                ],
                fluid=True,
            )
        )

    next_slug = step_meta.get("next")
    body_placeholder = html.Div(
        [
            html.P("This step is coming soon. You can still save and move forward.", className="text-muted"),
        ]
    )

    if slug == "volume-summary":
        content = _volume_summary_layout(step_meta)
    elif slug == "forecasting":
        content = html.Div(
            [
                _section_heading(step_meta),
                dbc.Alert(
                    "Forecasting step has been merged into Volume Summary. "
                    "Use Volume Summary to run Phase 1 and Phase 2, then continue to Transformation Projects.",
                    color="info",
                ),
                dcc.Link(
                    dbc.Button("Go to Volume Summary →", color="primary"),
                    href="/forecast/volume-summary",
                    style={"textDecoration": "none"},
                ),
            ],
            className="forecast-page",
        )
    elif slug == "transformation-projects":
        content = _generic_step_layout(step_meta, _transformation_layout(step_meta), "tp-complete", next_slug)
    elif slug == "daily-interval":
        content = _generic_step_layout(step_meta, _daily_interval_layout(step_meta), "di-complete", None)
    else:
        content = body_placeholder

    return html.Div(
        dbc.Container(
            [
                header_bar(),
                content,
            ],
            fluid=True,
        )
    )
