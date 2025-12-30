# CAP CONNECT

CAP CONNECT is a Dash/Flask platform for forecasting, transformation, capacity planning, and scheduling. The platform is designed to start with forecasting, then progressively refine forecasts into daily and interval plans, and finally use those plans for capacity planning and scheduling.

## End-to-End Flow (Start with Forecasting)
1. **Forecasting Workspace**: Upload data, generate seasonality and normalized ratios, run Prophet smoothing, and produce Phase 1/Phase 2 multi-model forecasts.
2. **Transformation Projects**: Apply sequential adjustments (Transform, IA, Marketing) and publish final forecast output.
3. **Daily and Interval Forecast**: Split monthly forecast into daily and interval targets using recent interval history and holiday context.
4. **Capacity Planning**: Build plans, validate staffing vs demand, and track plan history by Business Area and channel.
5. **Scheduling and Staffing**: Maintain rosters, normalize schedules, and layer hiring/class data onto supply.

## Modules and Routes
- **Forecasting**: `/forecast`, `/forecast/volume-summary`, `/forecast/transformation-projects`, `/forecast/daily-interval`
- **Capacity Planning**: `/planning`, `/plan/<id>`, `/plan/ba/<BA>`
- **Scheduling and Staffing**: `/roster`, `/newhire`
- **Operations and Monitoring**: `/ops`
- **Data and Settings**: `/dataset`, `/settings`, `/shrink`, `/budget`
- **Help**: `/help`

## Forecasting Workspace (What Happens in Order)
- **Volume Summary**: Uploads and data quality checks, seasonality and contact ratio setup.
- **Normalized Ratio 1**: Cap seasonality, adjust base volume, and apply changes.
- **Prophet Smoothing**: Generate Normalized Ratio 2 and preview smoothing results.
- **Phase 1**: Run multi-model forecasts (Prophet, RF, XGB, SARIMAX, VAR) with accuracy review.
- **Phase 2**: Apply best configs to generate final forecast outputs.
- **Transformation Projects**: Apply sequential adjustments to produce a final forecast.
- **Daily/Interval**: Convert the final forecast into daily and interval targets.

## Capacity Planning
- **Planning Workspace**: Create and manage plans by Business Area, Sub BA, channel, location, and site.
- **Plan Detail**: Weekly tables, plan notes, validation views, and staffing vs demand checks.
- **BA Rollups**: Summarized views across business areas for higher-level reporting.

## Scheduling and Staffing
- **Roster**: Download a template, upload schedules, and preview normalized schedules.
- **New Hire**: Maintain class start dates, levels, and ramp plans.
- **Supply Calculations**: Roster and hiring feed supply FTE used in capacity planning and ops views.

## Data Inputs (Key Files and Columns)
- **Volume upload** (CSV/XLSX): must include `date` and `volume`. Optional `category`, `forecast_group`, IQ, and holidays data.
- **IQ + Volume workbook** (XLSX): `Volume` and `IQ_Data` sheets, optional `Holidays` sheet.
- **Interval history** (CSV/XLSX): `date`, `interval` (e.g., `09:00` or `09:00-09:30`), and `volume` (AHT optional).
- **Roster**: use `/roster` to download a template and upload filled schedules.
- **Headcount/Hiring**: maintained via datasets and plan pages; used for supply FTE.
- **Shrinkage/Attrition**: uploaded and normalized to weekly series.
- **Ops data**: voice (volume + AHT by interval) and back office (items + SUT daily).

## Outputs and Storage
- **Forecast outputs**: seasonality tables, normalized ratios, Phase 1/2 results, accuracy, and configuration snapshots.
- **Transformation outputs**: sequentially adjusted forecasts and final forecast tables.
- **Daily/Interval outputs**: daily totals and interval forecasts.
- **File locations**:
  - `exports/` default output folder.
  - `latest_forecast_full_path.txt` and `latest_forecast_base_dir.txt` for transformation outputs.
  - `forecast_dates.csv` written on selection in Transformation Projects.
  - `holidays_list.csv` stored during Volume Summary/seasonality workflows.
  - `latest_forecast_path.txt` stored after saving daily forecasts.

## Configuration and Permissions
- Model hyperparameters and general settings are persisted via `config_manager.py`.
- Settings are effective-dated and applied by week to maintain historical consistency.
- Role gating: Admin and Planner can save settings; Viewer is read-only.

## Getting Started
1. **Install**: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. **Run**: `python main.py` (Dash server on localhost)
3. **Start**: go to `/forecast` and follow the guided flow.

## Help and Docs
- In-app documentation lives at `/help` and mirrors the forecasting to planning workflow.
- Use `/help` for step-by-step guidance across forecasting, capacity planning, and scheduling.

## Notes
- Global loading overlay guards navigation.
- The repo may contain user-edited work; avoid resetting unrelated changes.
- Network access can be restricted; install dependencies locally if needed.
