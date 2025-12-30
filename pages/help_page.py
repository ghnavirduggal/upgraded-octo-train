from __future__ import annotations
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from app_instance import app
from common import header_bar


SECTIONS = [
    dict(id="flow-overview", title="End-to-End Flow (Start with Forecasting)",
         search="forecast transformation daily interval capacity planning scheduling roster",
         body=html.Div([
             html.P("CAP CONNECT is designed to move from forecast to plan to schedule."),
             html.Ul([
                 html.Li("Forecasting Workspace: build monthly forecasts and select best models."),
                 html.Li("Transformation Projects: apply sequential adjustments and publish final forecast."),
                 html.Li("Daily and Interval Forecast: split monthly totals into daily and interval targets."),
                 html.Li("Capacity Planning: create plans, validate staffing vs demand, and track plan history."),
                 html.Li("Scheduling and Staffing: manage rosters and hiring inputs for supply FTE."),
             ]),
         ])),
    dict(id="forecast-overview", title="Forecasting Workflow (Start Here)",
         search="forecast volume summary seasonality smoothing prophet phase 1 phase 2 transformation daily interval",
         body=html.Div([
             html.P("Use the forecasting workspace to move from monthly volume to actionable plans."),
             html.Ul([
                 html.Li("Volume Summary: upload data, review IQ/Volume summaries, and generate seasonality."),
                 html.Li("Normalized Ratio 1: adjust caps and base volume, then apply changes."),
                 html.Li("Prophet Smoothing: smooth the series and confirm Normalized Ratio 2."),
                 html.Li("Phase 1: run multi-model forecasts and review accuracy."),
                 html.Li("Phase 2: apply best configs and generate final forecast outputs."),
                 html.Li("Transformation Projects: apply sequential adjustments and save final outputs."),
                 html.Li("Daily and Interval Forecast: split monthly totals into daily and interval plans."),
             ]),
         ])),
    dict(id="forecast-inputs", title="Forecasting Inputs",
         search="iq_data volume holidays xlsx csv date volume category forecast_group",
         body=html.Ul([
             html.Li("Volume upload: CSV/XLSX with date and volume; optional category and forecast_group."),
             html.Li("IQ_Data sheet: required for IQ summary and contact ratio calculations."),
             html.Li("Holidays sheet: optional, used for Prophet regressors and seasonality."),
             html.Li("Interval history: date, interval, volume (AHT optional) for daily/interval split."),
         ])),
    dict(id="transformation", title="Transformation Projects",
         search="transformations adjustments ia marketing sequential base forecast final",
         body=html.Div([
             html.P("Applies sequential adjustments to the base forecast for a selected group, model, and year."),
             html.Ul([
                 html.Li("Fields: Transformation 1-3, IA 1-3, Marketing Campaign 1-3 (percent changes)."),
                 html.Li("Forecast columns are generated sequentially from Base_Forecast_for_Forecast_Group."),
                 html.Li("Outputs: Final_Forecast_Post_Transformations and forecast tables."),
                 html.Li("Writes forecast_dates.csv to drive Daily and Interval Forecast selection."),
             ]),
         ])),
    dict(id="daily-interval", title="Daily and Interval Forecast",
         search="daily interval forecast split distribution history forecast_dates",
         body=html.Ul([
             html.Li("Uses forecast_dates.csv, holidays_list.csv, and the final transformation output."),
             html.Li("Builds daily distribution from recent history and lets you edit the split."),
             html.Li("Generates daily totals and interval forecasts for the selected month."),
             html.Li("Saves outputs alongside transformation files for downstream planning."),
         ])),
    dict(id="capacity", title="Capacity Planning Overview",
         search="planning workspace plan detail ba rollup validation",
         body=html.Div([
             html.P("Planning Workspace manages plans across business areas and tracks history."),
             html.Ul([
                 html.Li("Create or duplicate plans, then open Plan Detail."),
                 html.Li("Plan Detail includes weekly tables, notes, and validation views."),
                 html.Li("BA rollups provide summarized capacity views."),
             ]),
             html.Img(src="/assets/help/screen-1.png", style={"maxWidth":"100%","marginTop":"8px"}),
         ])),
    dict(id="scheduling", title="Scheduling and Staffing",
         search="scheduling roster schedule upload normalized new hire staffing",
         body=html.Ul([
             html.Li("Roster page: download template, upload schedules, and preview normalized schedule."),
             html.Li("Bulk edits: clear ranges or mark leave/off patterns."),
             html.Li("New Hire page: manage class start dates, levels, and production starts."),
             html.Li("Roster and hiring feed supply FTE calculations."),
         ])),
    dict(id="data-ops", title="Dataset and Ops Views",
         search="dataset ops voice back office kpi requirements supply",
         body=html.Ul([
             html.Li("Dataset page: snapshot of inputs and scope filters for planners."),
             html.Li("Ops page: requirements vs supply, voice and back office metrics."),
             html.Li("Voice uses volume + AHT; back office uses items + SUT."),
         ])),
    dict(id="budget-shrink", title="Budget and Shrink",
         search="budget shrink attrition weekly",
         body=html.Ul([
             html.Li("Shrink and attrition uploads are normalized to weekly series."),
             html.Li("Budget inputs support planning scenarios and validation."),
         ])),
    dict(id="settings", title="Settings and Effective-Dated Config",
         search="settings scope effective dated monday week",
         body=html.Div([
             html.P("Settings store an Effective Week (Monday). Computations use the latest settings where effective_week <= target date."),
             html.Ul([
                 html.Li("Save today -> applies to this and future weeks."),
                 html.Li("Change next week -> applies from next week onward; past weeks keep older settings."),
             ]),
             html.Img(src="/assets/help/screen-2.png", style={"maxWidth":"100%","marginTop":"8px"}),
         ])),
    dict(id="uploads", title="Uploads and Storage",
         search="headcount roster voice back office forecast actual timeseries storage keys",
         body=html.Ul([
             html.Li("Headcount: upsert by BRID; provides hierarchy and manager mapping."),
             html.Li("Roster: WIDE/LONG; stored as roster_wide and roster_long (dedupe by BRID,date)."),
             html.Li("Forecasts/Actuals: Voice (volume + AHT by date/interval), BO (items + SUT by date)."),
             html.Li("Shrinkage and Attrition: raw -> weekly series for KPIs."),
             html.Li("Forecast outputs: saved paths in latest_forecast_full_path.txt and latest_forecast_base_dir.txt."),
         ])),
    dict(id="templates", title="Templates and Normalizers",
         search="templates normalizers",
         body=html.Ul([
             html.Li("Headcount: BRID, Full Name, Line Manager BRID/Name, Journey, Level 3, Location, Group."),
             html.Li("Voice/BO: Date (+ Interval for Voice), Volume/Items, AHT/SUT."),
             html.Li("Shrinkage/Attrition: Raw -> normalized + weekly series."),
         ])),
    dict(id="roles", title="Roles and Permissions",
         search="roles admin planner viewer permissions",
         body=html.Ul([
             html.Li("Admin/Planner can save settings; Admin can delete plans."),
             html.Li("Viewer is read-only."),
         ])),
    dict(id="dupes", title="Duplicate Uploads",
         search="duplicate overwrite upsert append",
         body=html.Ul([
             html.Li("Headcount: upsert by BRID (last wins)."),
             html.Li("Timeseries: append by date/week; overlapping dates are replaced."),
             html.Li("Roster snapshots: overwrite; long dedupes by (BRID,date)."),
             html.Li("Plan bulk roster: upsert per BRID within a plan."),
             html.Li("Shrinkage weekly: merged; Attrition weekly: overwrite."),
         ])),
    dict(id="media", title="Quickstart Video and Screenshots",
         search="video screenshots quickstart",
         body=html.Div([
             html.Video(src="/assets/help/quickstart.mp4", controls=True, style={"maxWidth":"100%"}),
             html.P("Place /assets/help/quickstart.mp4 and /assets/help/screen-*.png to enable media."),
         ])),
    dict(id="troubleshoot", title="Troubleshooting",
         search="kpi not updating role settings",
         body=html.Ul([
             html.Li("If KPIs do not reflect uploads, ensure saves are complete and labels match."),
             html.Li("Effective settings: refresh and re-run for target dates to pick the correct version."),
             html.Li("Role errors on save: verify Admin/Planner permissions."),
         ])),
]


def _toc():
    return html.Ul([html.Li(html.A(s["title"], href=f"#{s['id']}")) for s in SECTIONS])


def _section_card(sec):
    return dbc.Card(dbc.CardBody([
        html.H4(sec["title"], id=sec["id"]),
        sec["body"],
        html.Div(html.A("Back to top", href="#help-top"), className="mt-2")
    ]), className="mb-3")


def page_help():
    return html.Div(dbc.Container([
        header_bar(),
        # Anchor for "Back to top" links (kept out of flex row so it doesn't affect alignment)
        html.Div(id="help-top"),
        html.Div([
            html.H3("Help & Documentation", style={"margin": "12px", "color":"#3a5166"}),
            html.Div([
                html.Button(
                    html.I(className="bi bi-search"),
                    id="help-search-toggle",
                    className="btn btn-link p-0 search-btn",
                    title="Search",
                    style={"fontSize": "1.25rem"},
                ),
                dbc.Input(id="help-search", placeholder="Search help...", type="text", debounce=True, className="search-input")
            ], id="help-search-wrap", className="help-search d-flex align-items-center ms-auto")
        ], className="d-flex align-items-center mb-2", style={"border":"rgba(0,0,0,0.175) 1px solid","borderRadius": "5px", "background": "#fff", "boxShadow": "rgba(0, 0, 0, 0.06) 0px 2px 8px", "marginLeft":"12px", "marginRight":"12px"}),
        dcc.Store(id="help-search-open", data=False),
        dbc.Card(dbc.CardBody([html.H5("Table of Contents"), _toc()]), className="mb-3"),
        html.Div(id="help-sections", children=[_section_card(s) for s in SECTIONS])
    ], fluid=True, className="help-page"), className="loading-page")


@app.callback(Output("help-sections", "children"), Input("help-search", "value"))
def _filter_sections(q):
    q = (q or "").strip().lower()
    if not q:
        return [_section_card(s) for s in SECTIONS]
    out = []
    for s in SECTIONS:
        hay = f"{s['title']} {s.get('search','')}".lower()
        if q in hay:
            out.append(s)
    if not out:
        return [dbc.Alert(f"No sections match '{q}'.", color="warning")]
    return [_section_card(s) for s in out]


@app.callback(
    Output("help-search-open", "data"),
    Input("help-search-toggle", "n_clicks"),
    State("help-search-open", "data"),
    prevent_initial_call=True,
)
def _toggle_search(n, is_open):
    try:
        return not bool(is_open)
    except Exception:
        return True


@app.callback(
    Output("help-search-wrap", "className"),
    Input("help-search-open", "data"),
)
def _set_search_class(open_now):
    base = "help-search d-flex align-items-center ms-auto"
    return f"{base} open" if open_now else base
