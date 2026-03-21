# ==========================================================
# Electric Bus Depot Load Curve Tool
# ==========================================================
# This version uses fixed charging sessions.
#
# Main logic:
# - The charging window is split into fixed-length sessions
# - Each bus is assigned to exactly ONE session
# - A bus cannot continue charging in later sessions
# - The tool distributes buses across sessions to minimize
#   the maximum depot power needed
#
# This version includes two modes:
# 1. Estimate minimum depot capacity needed
# 2. Check against a known depot power limit
#
# User-facing improvements in this version:
# - Short user guide under the title
# - Traffic-light style status
# - Clear comparison between required and available depot capacity
# - Smaller font in the KPI/results line
# - Bigger font in summary/session tables
# - More intuitive labels
# - Advanced details hidden in an expander
# - Charging window usage shown clearly
# - Charging window shaded in the 24-hour chart
# - Help text on key inputs
# - Assumptions box moved to bottom of page
# - PDF export of summary + session table
#
# IMPORTANT:
# Add "reportlab" to your requirements.txt for PDF export:
# streamlit
# pandas
# plotly
# reportlab
# ==========================================================

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


# ==========================================================
# PAGE / STYLE SETTINGS
# ==========================================================
st.set_page_config(page_title="Electric Bus Depot Load Curve Tool", layout="wide")

st.markdown(
    """
    <style>
    /* Bigger table text for readability */
    [data-testid="stDataFrame"] div {
        font-size: 15px;
    }

    /* Slightly bigger regular text in app */
    .stMarkdown p {
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# HELPER: KPI BOXES
# ==========================================================
def kpi_box(title: str, value: str, highlight: bool = False):
    """
    Custom result box with smaller font than st.metric.
    """
    if highlight:
        bg = "#eef6ff"
        border = "#cce0ff"
        title_color = "#4a6fa5"
        value_color = "#1f4e79"
    else:
        bg = "#f8f9fa"
        border = "#e6e6e6"
        title_color = "#666666"
        value_color = "#222222"

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border:1px solid {border};
            border-radius:10px;
            padding:10px 8px;
            text-align:center;
            min-height:76px;
            display:flex;
            flex-direction:column;
            justify-content:center;
        ">
            <div style="font-size:12px; color:{title_color}; margin-bottom:4px;">
                {title}
            </div>
            <div style="font-size:18px; font-weight:700; color:{value_color}; line-height:1.15;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# INPUT DATA STRUCTURE
# ==========================================================
@dataclass
class DepotInputsSession:
    n_12m: int
    n_18m: int
    battery_12m_kwh: float
    battery_18m_kwh: float
    charger_power_kw: float
    window_start: str
    window_end: str
    session_length_h: float
    efficiency: float
    min_soc: float
    target_soc: float
    timestep_min: int
    mode: str  # "estimate_capacity" or "check_known_limit"
    grid_cap_kw: Optional[float]


# ==========================================================
# HELPER: BUILD TIME WINDOW
# ==========================================================
def build_time_window(window_start: str, window_end: str):
    """
    Convert the user-defined charging window into datetimes.
    A dummy date is used only for internal time handling.
    """
    base_date = datetime(2026, 3, 20)

    start_dt = datetime.strptime(window_start, "%H:%M").replace(
        year=base_date.year, month=base_date.month, day=base_date.day
    )
    end_dt = datetime.strptime(window_end, "%H:%M").replace(
        year=base_date.year, month=base_date.month, day=base_date.day
    )

    # Handle overnight charging windows such as 23:00 -> 05:00
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)

    return start_dt, end_dt


# ==========================================================
# HELPER: SPLIT INTEGER EVENLY ACROSS SESSIONS
# ==========================================================
def split_evenly(total_items: int, n_bins: int) -> List[int]:
    """
    Split an integer as evenly as possible across n bins.
    Example: total_items = 10, n_bins = 3 -> [4, 3, 3]
    """
    base = total_items // n_bins
    remainder = total_items % n_bins
    result = [base] * n_bins

    for i in range(remainder):
        result[i] += 1

    return result


# ==========================================================
# HELPER: BUILD STATUS CATEGORY
# ==========================================================
def get_status_category(
    mode: str, capacity_gap_kw: float, required_kw: float, available_kw: Optional[float]
):
    """
    Return a simple traffic-light status category.
    """
    if mode == "estimate_capacity":
        return "info"

    if available_kw is None:
        return "info"

    if capacity_gap_kw > 1e-9:
        return "red"

    margin_ratio = (available_kw - required_kw) / required_kw if required_kw > 0 else 0.0

    if margin_ratio <= 0.10:
        return "yellow"
    return "green"


# ==========================================================
# HELPER: PDF EXPORT
# ==========================================================
def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    """
    Wrap text to fit within a maximum width in the PDF.
    """
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current_line = words[0]

    for word in words[1:]:
        trial_line = current_line + " " + word
        if stringWidth(trial_line, font_name, font_size) <= max_width:
            current_line = trial_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def draw_key_value_section(
    c: canvas.Canvas,
    title: str,
    items: List[tuple],
    x_left: float,
    y_top: float,
    page_width: float,
    page_height: float,
    bottom_margin: float,
):
    """
    Draw a section with key-value pairs.
    Returns updated y position.
    """
    y = y_top

    if y < bottom_margin + 40:
        c.showPage()
        y = page_height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_left, y, title)
    y -= 18

    label_width = 220
    value_width = page_width - x_left * 2 - label_width - 10

    for key, value in items:
        value_lines = wrap_text(str(value), "Helvetica", 9, value_width)
        row_height = max(14, 12 * len(value_lines))

        if y - row_height < bottom_margin:
            c.showPage()
            y = page_height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_left, y, title + " (cont.)")
            y -= 18

        c.setFont("Helvetica-Bold", 9)
        c.drawString(x_left, y, str(key))

        c.setFont("Helvetica", 9)
        value_y = y
        for line in value_lines:
            c.drawString(x_left + label_width, value_y, line)
            value_y -= 11

        y -= row_height

    return y - 10


def draw_dataframe_section(
    c: canvas.Canvas,
    title: str,
    df: pd.DataFrame,
    x_left: float,
    y_top: float,
    page_width: float,
    page_height: float,
    bottom_margin: float,
):
    """
    Draw a simple tabular section for a dataframe in the PDF.
    Returns updated y position.
    """
    y = y_top
    max_table_width = page_width - 2 * x_left

    if y < bottom_margin + 50:
        c.showPage()
        y = page_height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_left, y, title)
    y -= 18

    cols = list(df.columns)
    n_cols = max(1, len(cols))
    col_width = max_table_width / n_cols

    # Header row
    if y < bottom_margin + 30:
        c.showPage()
        y = page_height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_left, y, title + " (cont.)")
        y -= 18

    c.setFont("Helvetica-Bold", 8)
    for i, col in enumerate(cols):
        c.drawString(x_left + i * col_width, y, str(col)[:22])
    y -= 14

    c.setFont("Helvetica", 8)
    for _, row in df.iterrows():
        if y < bottom_margin + 20:
            c.showPage()
            y = page_height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_left, y, title + " (cont.)")
            y -= 18
            c.setFont("Helvetica-Bold", 8)
            for i, col in enumerate(cols):
                c.drawString(x_left + i * col_width, y, str(col)[:22])
            y -= 14
            c.setFont("Helvetica", 8)

        for i, col in enumerate(cols):
            text = str(row[col])
            c.drawString(x_left + i * col_width, y, text[:22])
        y -= 12

    return y - 10


def create_summary_pdf(
    inputs: DepotInputsSession,
    summary: dict,
    main_results_df: pd.DataFrame,
    advanced_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
) -> bytes:
    """
    Create a downloadable PDF with:
    - app title
    - input settings
    - main results
    - advanced details
    - session table
    - scenario comparison
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    page_width, page_height = A4
    x_left = 40
    y = page_height - 50
    bottom_margin = 40

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_left, y, "Electric Bus Depot Load Curve Tool")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(x_left, y, "Summary export")
    y -= 24

    # Inputs section
    inputs_items = [
        ("Mode", "Estimate minimum depot capacity" if inputs.mode == "estimate_capacity" else "Check against known depot power limit"),
        ("12 m buses", inputs.n_12m),
        ("18 m buses", inputs.n_18m),
        ("Battery size of 12 m bus (kWh)", inputs.battery_12m_kwh),
        ("Battery size of 18 m bus (kWh)", inputs.battery_18m_kwh),
        ("Charger power per bus (kW)", inputs.charger_power_kw),
        ("Depot power limit (kW)", inputs.grid_cap_kw if inputs.grid_cap_kw is not None else "Not entered"),
        ("Charging start time", inputs.window_start),
        ("Charging end time", inputs.window_end),
        ("Fixed session length (hours)", inputs.session_length_h),
        ("Charging efficiency", inputs.efficiency),
        ("Minimum state of charge", inputs.min_soc),
        ("Target state of charge", inputs.target_soc),
    ]
    y = draw_key_value_section(c, "Inputs", inputs_items, x_left, y, page_width, page_height, bottom_margin)

    # Main results section
    main_items = list(zip(main_results_df["Metric"], main_results_df["Value"]))
    y = draw_key_value_section(c, "Main results", main_items, x_left, y, page_width, page_height, bottom_margin)

    # Advanced details
    if not advanced_df.empty:
        advanced_items = list(zip(advanced_df["Metric"], advanced_df["Value"]))
        y = draw_key_value_section(c, "Advanced details", advanced_items, x_left, y, page_width, page_height, bottom_margin)

    # Session table
    y = draw_dataframe_section(c, "Session allocation table", sessions_df, x_left, y, page_width, page_height, bottom_margin)

    # Scenario comparison
    y = draw_dataframe_section(c, "Quick scenario comparison", scenario_df, x_left, y, page_width, page_height, bottom_margin)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ==========================================================
# CORE MODEL FUNCTION
# ==========================================================
def build_fixed_session_model(inputs: DepotInputsSession):
    """
    Build the fixed-session charging model.

    Steps:
    1. Calculate energy required per bus type
    2. Check whether one bus can finish within one session
    3. Determine how many sessions fit in the charging window
    4. Spread buses across sessions as evenly as possible
    5. Compute session power and energy
    6. Build a real stepwise 24-hour load curve
    7. Compare against user grid cap if provided
    """

    # ------------------------------------------------------
    # 1. BUILD TIME WINDOW
    # ------------------------------------------------------
    start_dt, end_dt = build_time_window(inputs.window_start, inputs.window_end)
    window_h = (end_dt - start_dt).total_seconds() / 3600

    if window_h <= 0:
        raise ValueError("Charging window must be greater than zero.")

    if inputs.session_length_h <= 0:
        raise ValueError("Session length must be greater than zero.")

    # Number of complete sessions that fit in the charging window
    n_sessions = int(window_h // inputs.session_length_h)

    if n_sessions < 1:
        raise ValueError(
            "The charging window is shorter than one session. "
            "Please reduce session length or increase the charging window."
        )

    used_session_time_h = n_sessions * inputs.session_length_h
    unused_time_h = window_h - used_session_time_h

    # ------------------------------------------------------
    # 2. ENERGY REQUIRED PER BUS
    # ------------------------------------------------------
    soc_fraction = max(0.0, inputs.target_soc - inputs.min_soc)

    # Battery energy added per bus
    e_12m_battery_kwh = soc_fraction * inputs.battery_12m_kwh
    e_18m_battery_kwh = soc_fraction * inputs.battery_18m_kwh

    # Grid energy required per bus (includes charging losses)
    e_12m_grid_kwh = e_12m_battery_kwh / inputs.efficiency
    e_18m_grid_kwh = e_18m_battery_kwh / inputs.efficiency

    # ------------------------------------------------------
    # 3. FEASIBILITY OF ONE BUS WITHIN ONE SESSION
    # ------------------------------------------------------
    p_12m_required_kw = e_12m_grid_kwh / inputs.session_length_h
    p_18m_required_kw = e_18m_grid_kwh / inputs.session_length_h

    feasible_12m = p_12m_required_kw <= inputs.charger_power_kw + 1e-9
    feasible_18m = p_18m_required_kw <= inputs.charger_power_kw + 1e-9

    if not feasible_12m or not feasible_18m:
        messages = []
        if not feasible_12m:
            messages.append(
                f"12 m buses require {p_12m_required_kw:.1f} kW per bus, above the charger power of {inputs.charger_power_kw:.1f} kW."
            )
        if not feasible_18m:
            messages.append(
                f"18 m buses require {p_18m_required_kw:.1f} kW per bus, above the charger power of {inputs.charger_power_kw:.1f} kW."
            )
        raise ValueError(
            "A bus cannot finish charging within one fixed session under the current assumptions. "
            + " ".join(messages)
        )

    # ------------------------------------------------------
    # 4. CHECK TOTAL BUSES
    # ------------------------------------------------------
    total_buses = inputs.n_12m + inputs.n_18m
    if total_buses == 0:
        raise ValueError("Please enter at least one bus.")

    # ------------------------------------------------------
    # 5. ASSIGN BUSES TO SESSIONS TO MINIMIZE PEAK
    # ------------------------------------------------------
    split_12m = split_evenly(inputs.n_12m, n_sessions)
    split_18m = split_evenly(inputs.n_18m, n_sessions)

    session_rows = []

    for s in range(n_sessions):
        session_start = start_dt + timedelta(hours=s * inputs.session_length_h)
        session_end = session_start + timedelta(hours=inputs.session_length_h)

        n_12 = split_12m[s]
        n_18 = split_18m[s]

        session_power_kw = n_12 * p_12m_required_kw + n_18 * p_18m_required_kw
        session_energy_kwh = n_12 * e_12m_grid_kwh + n_18 * e_18m_grid_kwh

        session_rows.append(
            {
                "Session": s + 1,
                "Start": session_start,
                "End": session_end,
                "12 m buses": n_12,
                "18 m buses": n_18,
                "Total buses": n_12 + n_18,
                "Session power (MW)": round(session_power_kw / 1000, 3),
                "Energy delivered (MWh)": round(session_energy_kwh / 1000, 3),
            }
        )

    sessions_df = pd.DataFrame(session_rows)

    # Minimum depot grid capacity needed = peak session power
    min_required_grid_kw = (
        sessions_df["Session power (MW)"].max() * 1000 if len(sessions_df) > 0 else 0.0
    )

    # ------------------------------------------------------
    # 6. CHECK AGAINST USER-ENTERED GRID LIMIT IF PROVIDED
    # ------------------------------------------------------
    available_limit_kw = None
    if inputs.mode == "check_known_limit":
        if inputs.grid_cap_kw is None:
            raise ValueError("Please enter the depot power limit.")

        available_limit_kw = inputs.grid_cap_kw

        if min_required_grid_kw <= inputs.grid_cap_kw + 1e-9:
            status_label = "Within depot power limit"
            status_text = (
                "The session schedule fits within the depot power limit you entered."
            )
            capacity_gap_kw = 0.0
            feasible_against_cap = True
        else:
            status_label = "Above depot power limit"
            status_text = (
                "The session schedule exceeds the depot power limit you entered."
            )
            capacity_gap_kw = min_required_grid_kw - inputs.grid_cap_kw
            feasible_against_cap = False
    else:
        status_label = "Minimum capacity estimated"
        status_text = (
            "The tool has estimated the minimum depot power needed under the fixed-session rule."
        )
        capacity_gap_kw = 0.0
        feasible_against_cap = True

    status_category = get_status_category(
        inputs.mode, capacity_gap_kw, min_required_grid_kw, available_limit_kw
    )

    # ------------------------------------------------------
    # 7. BUILD 24-HOUR STEP LOAD CURVE
    # ------------------------------------------------------
    # Start at 12:00 so overnight charging appears near the middle.
    display_start = start_dt.replace(hour=12, minute=0)
    if display_start > start_dt:
        display_start -= timedelta(days=1)

    full_steps = int(24 * 60 // inputs.timestep_min)

    session_power_map: Dict[str, float] = {}

    for _, row in sessions_df.iterrows():
        session_start = row["Start"]
        session_end = row["End"]
        session_power_kw = row["Session power (MW)"] * 1000

        current = session_start
        while current < session_end:
            key = current.strftime("%Y-%m-%d %H:%M")
            session_power_map[key] = session_power_kw
            current += timedelta(minutes=inputs.timestep_min)

    plot_rows = []
    for step in range(full_steps):
        ts = display_start + timedelta(minutes=step * inputs.timestep_min)
        key = ts.strftime("%Y-%m-%d %H:%M")

        plot_rows.append(
            {
                "time": ts,
                "time_label": ts.strftime("%H:%M"),
                "depot_load_kw": session_power_map.get(key, 0.0),
            }
        )

    plot_df = pd.DataFrame(plot_rows)

    # ------------------------------------------------------
    # 8. SUMMARY
    # ------------------------------------------------------
    total_grid_energy_kwh = inputs.n_12m * e_12m_grid_kwh + inputs.n_18m * e_18m_grid_kwh
    max_charger_system_kw = total_buses * inputs.charger_power_kw

    summary = {
        "Mode": (
            "Estimate minimum depot capacity"
            if inputs.mode == "estimate_capacity"
            else "Check against known depot power limit"
        ),
        "Number of sessions": n_sessions,
        "Session length (h)": inputs.session_length_h,
        "Charging window (h)": round(window_h, 2),
        "Session time used (h)": round(used_session_time_h, 2),
        "Unused time in charging window (h)": round(unused_time_h, 2),
        "12 m buses": inputs.n_12m,
        "18 m buses": inputs.n_18m,
        "Total buses / chargers": total_buses,
        "Battery size 12 m (kWh)": inputs.battery_12m_kwh,
        "Battery size 18 m (kWh)": inputs.battery_18m_kwh,
        "Minimum SOC": inputs.min_soc,
        "Target SOC": inputs.target_soc,
        "Electricity needed per 12 m bus (kWh)": round(e_12m_grid_kwh, 1),
        "Electricity needed per 18 m bus (kWh)": round(e_18m_grid_kwh, 1),
        "Required power per 12 m bus within one session (kW)": round(p_12m_required_kw, 1),
        "Required power per 18 m bus within one session (kW)": round(p_18m_required_kw, 1),
        "Total electricity needed (MWh)": round(total_grid_energy_kwh / 1000, 3),
        "Depot power needed (MW)": round(min_required_grid_kw / 1000, 3),
        "Maximum charger system power (MW)": round(max_charger_system_kw / 1000, 3),
        "Status": status_label,
    }

    if inputs.mode == "check_known_limit":
        summary["Depot power limit entered by user (MW)"] = round(inputs.grid_cap_kw / 1000, 3)
        summary["Capacity gap above limit (MW)"] = round(capacity_gap_kw / 1000, 3)
        summary["Can the fleet be charged overnight?"] = "Yes" if feasible_against_cap else "No"

    return (
        sessions_df,
        plot_df,
        summary,
        status_label,
        status_text,
        capacity_gap_kw,
        status_category,
        min_required_grid_kw,
    )


# ==========================================================
# STREAMLIT USER INTERFACE
# ==========================================================
st.title("Electric Bus Depot Load Curve Tool")
st.caption(
    "Quick guide: fill in the inputs in the left panel. "
    "The tool estimates depot power needs for the selected fleet, shows whether the depot limit is sufficient, "
    "displays the 24-hour charging profile, and summarizes how buses are allocated across sessions."
)

with st.sidebar:
    st.header("Inputs")

    mode_display = st.radio(
        "What do you want the tool to do?",
        options=[
            "Estimate minimum depot capacity",
            "Check against a known depot power limit",
        ],
        index=0,
    )

    mode = (
        "estimate_capacity"
        if mode_display == "Estimate minimum depot capacity"
        else "check_known_limit"
    )

    n_12m = st.number_input(
        "Number of 12 m buses",
        min_value=0,
        max_value=1000,
        value=20,
        step=1,
        help="How many 12-meter buses are in the fleet.",
    )
    n_18m = st.number_input(
        "Number of 18 m buses",
        min_value=0,
        max_value=1000,
        value=10,
        step=1,
        help="How many 18-meter buses are in the fleet.",
    )

    battery_12m = st.number_input(
        "Battery size of 12 m bus (kWh)",
        min_value=100.0,
        max_value=2000.0,
        value=350.0,
        step=10.0,
        help="Typical battery size for a 12-meter bus.",
    )
    battery_18m = st.number_input(
        "Battery size of 18 m bus (kWh)",
        min_value=100.0,
        max_value=2000.0,
        value=500.0,
        step=10.0,
        help="Typical battery size for an 18-meter bus.",
    )

    charger_power_kw = st.number_input(
        "Charger power per bus (kW)",
        min_value=10.0,
        max_value=1000.0,
        value=150.0,
        step=10.0,
        help="Maximum charging power available to each bus.",
    )

    grid_cap_kw = None
    if mode == "check_known_limit":
        grid_cap_kw = st.number_input(
            "Depot power limit (kW)",
            min_value=100.0,
            max_value=20000.0,
            value=1800.0,
            step=50.0,
            help="The maximum depot power available from the grid connection.",
        )

    window_start = st.text_input(
        "Charging start time (HH:MM)",
        "23:00",
        help="Time when overnight charging can begin.",
    )
    window_end = st.text_input(
        "Charging end time (HH:MM)",
        "05:00",
        help="Time by which charging must be completed.",
    )

    session_length_h = st.number_input(
        "Fixed session length (hours)",
        min_value=0.5,
        max_value=12.0,
        value=3.0,
        step=0.5,
        help="Length of each charging session. Each bus must finish within one session.",
    )

    efficiency = st.slider(
        "Charging efficiency",
        min_value=0.70,
        max_value=1.00,
        value=0.92,
        step=0.01,
        help="Includes charging losses. Typical values are around 0.90 to 0.95.",
    )
    min_soc = st.slider(
        "Minimum state of charge",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Battery level when buses return to the depot.",
    )
    target_soc = st.slider(
        "Target state of charge",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Battery level you want before buses leave again.",
    )

    timestep_min = st.selectbox(
        "Timestep for chart (minutes)",
        options=[5, 10, 15, 30, 60],
        index=2,
        help="Only affects how smooth the chart looks.",
    )

inputs = DepotInputsSession(
    n_12m=n_12m,
    n_18m=n_18m,
    battery_12m_kwh=battery_12m,
    battery_18m_kwh=battery_18m,
    charger_power_kw=charger_power_kw,
    window_start=window_start,
    window_end=window_end,
    session_length_h=session_length_h,
    efficiency=efficiency,
    min_soc=min_soc,
    target_soc=target_soc,
    timestep_min=timestep_min,
    mode=mode,
    grid_cap_kw=grid_cap_kw,
)

# ==========================================================
# RUN MODEL
# ==========================================================
try:
    (
        sessions_df,
        plot_df,
        summary,
        status_label,
        status_text,
        capacity_gap_kw,
        status_category,
        required_kw,
    ) = build_fixed_session_model(inputs)

    # ------------------------------------------------------
    # KEY RESULTS LINE
    # ------------------------------------------------------
    required_mw = required_kw / 1000
    total_mwh = summary["Total electricity needed (MWh)"]
    n_sessions = summary["Number of sessions"]
    session_len = summary["Session length (h)"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_box("Required depot power", f"{required_mw:.3f} MW", highlight=True)

    with c2:
        kpi_box("Total electricity needed", f"{total_mwh:.3f} MWh")

    with c3:
        if mode == "check_known_limit":
            kpi_box("Available depot power", f"{inputs.grid_cap_kw / 1000:.3f} MW")
        else:
            kpi_box("Number of sessions", f"{n_sessions}")

    with c4:
        if mode == "check_known_limit":
            if capacity_gap_kw > 1e-9:
                kpi_box("Capacity gap", f"{capacity_gap_kw / 1000:.3f} MW")
            else:
                kpi_box("Capacity gap", "0.000 MW")
        else:
            kpi_box("Session length", f"{session_len:.1f} h")

    # ------------------------------------------------------
    # REQUIRED VS AVAILABLE COMPARISON
    # ------------------------------------------------------
    st.subheader("Capacity check")

    comp1, comp2, comp3, comp4 = st.columns(4)

    with comp1:
        kpi_box("Required depot power", f"{required_mw:.3f} MW", highlight=True)

    with comp2:
        if mode == "check_known_limit":
            kpi_box("Available depot power", f"{inputs.grid_cap_kw / 1000:.3f} MW")
        else:
            kpi_box("Available depot power", "Not entered")

    with comp3:
        if mode == "check_known_limit":
            if capacity_gap_kw > 1e-9:
                kpi_box("Gap", f"+{capacity_gap_kw / 1000:.3f} MW")
            else:
                spare_kw = inputs.grid_cap_kw - required_kw
                kpi_box("Spare margin", f"{spare_kw / 1000:.3f} MW")
        else:
            kpi_box("Gap", "—")

    with comp4:
        if mode == "estimate_capacity":
            kpi_box("Status", "Estimated")
        elif status_category == "green":
            kpi_box("Status", "Feasible")
        elif status_category == "yellow":
            kpi_box("Status", "Tight")
        else:
            kpi_box("Status", "Not feasible")

    # ------------------------------------------------------
    # TRAFFIC-LIGHT MESSAGE
    # ------------------------------------------------------
    if status_category == "red":
        st.error(
            f"{status_text} The schedule exceeds the entered depot limit by {capacity_gap_kw / 1000:.3f} MW."
        )
    elif status_category == "yellow":
        st.warning(
            "The fleet can be charged overnight, but the depot power limit is tight with little spare margin."
        )
    elif status_category == "green":
        st.success(status_text)
    else:
        st.info(status_text)

    # ------------------------------------------------------
    # MAIN LAYOUT
    # ------------------------------------------------------
    left_col, right_col = st.columns([1.1, 1.9])

    with left_col:
        st.subheader("Main results")
        main_results = {
            "Depot power needed (MW)": summary["Depot power needed (MW)"],
            "Total electricity needed (MWh)": summary["Total electricity needed (MWh)"],
            "Number of sessions": summary["Number of sessions"],
            "Session length (h)": summary["Session length (h)"],
            "Charging window (h)": summary["Charging window (h)"],
            "Session time used (h)": summary["Session time used (h)"],
            "Unused time in charging window (h)": summary["Unused time in charging window (h)"],
            "Status": summary["Status"],
        }

        if mode == "check_known_limit":
            main_results["Available depot power (MW)"] = summary["Depot power limit entered by user (MW)"]
            main_results["Capacity gap above limit (MW)"] = summary["Capacity gap above limit (MW)"]
            main_results["Can the fleet be charged overnight?"] = summary["Can the fleet be charged overnight?"]

        main_results_df = pd.DataFrame(
            [{"Metric": key, "Value": value} for key, value in main_results.items()]
        )
        st.dataframe(main_results_df, use_container_width=True, hide_index=True)

        with st.expander("Advanced details"):
            advanced_items = dict(summary)
            for key in list(main_results.keys()):
                advanced_items.pop(key, None)

            advanced_df = pd.DataFrame(
                [{"Metric": key, "Value": value} for key, value in advanced_items.items()]
            )
            st.dataframe(advanced_df, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("24-hour load curve")

        plot_df = plot_df.copy()
        plot_df["depot_load_mw"] = plot_df["depot_load_kw"] / 1000

        fig = go.Figure()

        # Add shaded charging window
        start_dt, end_dt = build_time_window(inputs.window_start, inputs.window_end)
        display_start = plot_df["time"].iloc[0]
        display_end = plot_df["time"].iloc[-1] + timedelta(minutes=inputs.timestep_min)

        shade_start = max(start_dt, display_start)
        shade_end = min(end_dt, display_end)

        if shade_end > shade_start:
            fig.add_vrect(
                x0=shade_start.strftime("%H:%M"),
                x1=shade_end.strftime("%H:%M"),
                fillcolor="lightgray",
                opacity=0.15,
                line_width=0,
                layer="below",
            )

        fig.add_trace(
            go.Scatter(
                x=plot_df["time_label"],
                y=plot_df["depot_load_mw"],
                mode="lines",
                line=dict(width=3, shape="hv"),
                name="Depot load",
            )
        )

        fig.update_layout(
            title="Step load curve over 24 hours",
            xaxis_title="Time of day",
            yaxis_title="Depot load (MW)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------
    # SESSION TABLE
    # ------------------------------------------------------
    st.subheader("Session allocation table")
    st.caption(
        "How the sessions are built: the tool spreads 12 m and 18 m buses as evenly as possible across sessions "
        "so that the required depot power is kept as low as possible."
    )

    display_sessions_df = sessions_df.copy()
    display_sessions_df["Start"] = display_sessions_df["Start"].dt.strftime("%H:%M")
    display_sessions_df["End"] = display_sessions_df["End"].dt.strftime("%H:%M")

    st.dataframe(display_sessions_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # SIMPLE SCENARIO TABLE
    # ------------------------------------------------------
    st.subheader("Quick scenario comparison")

    scenario_fleet_increases = [0, 10, 20]
    scenario_rows = []

    for extra_buses in scenario_fleet_increases:
        scenario_inputs = DepotInputsSession(
            n_12m=inputs.n_12m + extra_buses,
            n_18m=inputs.n_18m,
            battery_12m_kwh=inputs.battery_12m_kwh,
            battery_18m_kwh=inputs.battery_18m_kwh,
            charger_power_kw=inputs.charger_power_kw,
            window_start=inputs.window_start,
            window_end=inputs.window_end,
            session_length_h=inputs.session_length_h,
            efficiency=inputs.efficiency,
            min_soc=inputs.min_soc,
            target_soc=inputs.target_soc,
            timestep_min=inputs.timestep_min,
            mode="estimate_capacity",
            grid_cap_kw=None,
        )

        try:
            (
                _sessions_df,
                _plot_df,
                scenario_summary,
                _status_label,
                _status_text,
                _capacity_gap_kw,
                _status_category,
                _required_kw,
            ) = build_fixed_session_model(scenario_inputs)

            scenario_rows.append(
                {
                    "Scenario": f"+{extra_buses} extra 12 m buses" if extra_buses > 0 else "Current fleet",
                    "12 m buses": scenario_inputs.n_12m,
                    "18 m buses": scenario_inputs.n_18m,
                    "Depot power needed (MW)": scenario_summary["Depot power needed (MW)"],
                    "Total electricity needed (MWh)": scenario_summary["Total electricity needed (MWh)"],
                }
            )
        except Exception:
            scenario_rows.append(
                {
                    "Scenario": f"+{extra_buses} extra 12 m buses" if extra_buses > 0 else "Current fleet",
                    "12 m buses": scenario_inputs.n_12m,
                    "18 m buses": scenario_inputs.n_18m,
                    "Depot power needed (MW)": "Not feasible",
                    "Total electricity needed (MWh)": "Not feasible",
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # DOWNLOADS
    # ------------------------------------------------------
    download_col1, download_col2 = st.columns(2)

    with download_col1:
        csv_bytes = display_sessions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download session table CSV",
            data=csv_bytes,
            file_name="electric_bus_depot_sessions.csv",
            mime="text/csv",
        )

    with download_col2:
        pdf_bytes = create_summary_pdf(
            inputs=inputs,
            summary=summary,
            main_results_df=main_results_df,
            advanced_df=advanced_df,
            sessions_df=display_sessions_df,
            scenario_df=scenario_df,
        )
        st.download_button(
            "Download summary PDF",
            data=pdf_bytes,
            file_name="electric_bus_depot_summary.pdf",
            mime="application/pdf",
        )

    # ------------------------------------------------------
    # ASSUMPTIONS BOX AT BOTTOM
    # ------------------------------------------------------
    st.info(
        "How this version works:\n"
        "- The charging window is split into fixed-length sessions.\n"
        "- Each bus must complete its charging within one session only.\n"
        "- A bus cannot continue charging in later sessions.\n"
        "- All buses are assumed to arrive and depart within the same charging window.\n"
        "- All buses are assumed to arrive at the same minimum state of charge and charge to the same target state of charge.\n"
        "- One charger is assumed per bus.\n"
        "- Charging power is assumed constant in this version.\n"
        "- The tool spreads buses across sessions to minimize the maximum depot power needed.\n"
        "- In the depot-limit mode, the tool checks whether the resulting session schedule fits under the entered depot power limit."
    )

except Exception as e:
    st.error(str(e))