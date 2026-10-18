"""
excel_report.py
===============
Libro de Excel con los resultados de la evaluación (baselines_drl_ruteo_estocastico.md):
  · Resumen            — por (K, método): margen medio, gaps contra UB-DP y UB-PI,
                         factibilidad y métricas operativas; mundos simulados y replay
                         (etiquetado "replay, no independiente del simulador"); tiempos
                         fuera de línea.
  · Mundos simulados   — una fila por (K, mundo, método)
  · Replay             — ídem sobre los días reales de evaluación
stats_analysis.append_stats_sheets_to_excel agrega después las hojas estadísticas.
"""

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from evaluation import resumen_metodos
from stats_analysis import NOTA_REPLAY, NOTA_UB_DP_FLOJA, NOTA_UB_DP_FLOJA_LARGA

# ── Palette ───────────────────────────────────────────────────
_BLUE_DARK    = "2E5395"
_BLUE_LIGHT   = "DCE6F1"
_PURPLE_DARK  = "7030A0"
_PURPLE_LIGHT = "EBDDF0"
_GREEN_DARK   = "548235"
_GREEN_LIGHT  = "E2EFDA"
_YELLOW       = "FFF2CC"


# ── Style helpers ─────────────────────────────────────────────

def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _font(bold=False, color="000000", size=11):
    return Font(bold=bold, color=color, size=size)

def _center():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def _left():
    return Alignment(horizontal="left", vertical="center", wrap_text=True)

def _border():
    s = Side(style="thin")
    return Border(left=s, right=s, top=s, bottom=s)

def _clean(v):
    """Convert nan/inf to None for Excel cells."""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return v

def _header_cell(cell, text, fill_hex, font_color="FFFFFF", bold=True):
    cell.value = text
    cell.fill = _fill(fill_hex)
    cell.font = _font(bold=bold, color=font_color)
    cell.alignment = _center()
    cell.border = _border()

def _data_cell(cell, value, num_format=None, fill_hex=None):
    cell.value = value
    cell.alignment = _left()
    cell.border = _border()
    if fill_hex:
        cell.fill = _fill(fill_hex)
    if value is not None and num_format:
        cell.number_format = num_format


# ── Public entry point ────────────────────────────────────────

def build_formatted_excel(
    sim_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    summary_rows: list,
    output_path: str,
) -> None:
    wb = Workbook()
    _build_summary(wb, sim_df, replay_df, summary_rows)
    _build_raw(wb, "Mundos simulados", sim_df)
    if replay_df is not None and not replay_df.empty:
        _build_raw(wb, "Replay", replay_df.assign(nota=NOTA_REPLAY))
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    wb.save(output_path)


# ═════════════════════════════════════════════════════════════
#  SUMMARY SHEET
# ═════════════════════════════════════════════════════════════

_COLS_RESUMEN = [
    ("Método",              "metodo",              None),
    ("Mundos",              "n_mundos",            "0"),
    ("Margen medio (USD)",  "margen_medio",        "#,##0"),
    ("IC95 inf (USD)",      "margen_ic95_low",     "#,##0"),
    ("IC95 sup (USD)",      "margen_ic95_high",    "#,##0"),
    ("Gap vs UB-DP (%)",    "gap_ub_dp_pct",       '0.00"%"'),
    ("Gap vs UB-PI (%)",    "gap_ub_pi_pct",       '0.00"%"'),
    ("Violación de tiempo (%)", "violacion_pct",   '0.0"%"'),
    ("Millas cargadas (%)", "millas_cargadas_pct", '0.0"%"'),
    ("Cargas por ruta",     "cargas_medias",       "0.00"),
    ("Duración (h)",        "duracion_media_h",    "0.0"),
    ("ms por decisión",     "ms_por_decision",     "0.00"),
    ("Nota",                "nota",                None),
]


def _bloque_resumen(ws, row, titulo, color_oscuro, color_claro, resumen):
    """Escribe una tabla por nivel de K; devuelve la siguiente fila libre."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
    _header_cell(ws.cell(row, 1), titulo, color_oscuro)
    row += 1
    for K, g in resumen.groupby("K", sort=False):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
        _header_cell(ws.cell(row, 1), f"K = {K}", color_claro, font_color="000000")
        row += 1
        for c, (etiqueta, _, _) in enumerate(_COLS_RESUMEN, 1):
            _header_cell(ws.cell(row, c), etiqueta, color_claro, font_color="000000")
        row += 1
        for _, r in g.iterrows():
            es_cota = r["metodo"].startswith("UB-")
            for c, (_, col, fmt) in enumerate(_COLS_RESUMEN, 1):
                _data_cell(ws.cell(row, c), _clean(r.get(col)), fmt,
                           fill_hex=_YELLOW if es_cota else None)
            row += 1
        if (g["nota"] == NOTA_UB_DP_FLOJA).any():
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
            _data_cell(ws.cell(row, 1), f"* K = {K}: {NOTA_UB_DP_FLOJA_LARGA}", fill_hex=_YELLOW)
            ws.row_dimensions[row].height = 48
            row += 1
        row += 1
    return row


def _build_summary(wb, sim_df, replay_df, summary_rows):
    ws = wb.create_sheet("Resumen")
    sr = summary_rows[0] if summary_rows else {}
    row = 1

    row = _bloque_resumen(ws, row, "MUNDOS SIMULADOS (proceso exógeno, semillas de evaluación)",
                          _PURPLE_DARK, _PURPLE_LIGHT, resumen_metodos(sim_df))
    if replay_df is not None and not replay_df.empty:
        row = _bloque_resumen(ws, row, f"REPLAY DE DÍAS REALES — {NOTA_REPLAY}",
                              _BLUE_DARK, _BLUE_LIGHT, resumen_metodos(replay_df))

    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
    _header_cell(ws.cell(row, 1), "TIEMPO FUERA DE LÍNEA", _GREEN_DARK)
    row += 1
    for metrica, clave in (("Entrenamiento del DRL (s)", "DRL Training Time (s)"),
                           ("Cálculo de la tabla V de UB-DP (s)", "UB-DP Table Time (s)")):
        _data_cell(ws.cell(row, 1), metrica)
        _data_cell(ws.cell(row, 2), _clean(sr.get(clave)), "0.0")
        row += 1

    ws.column_dimensions["A"].width = 30
    for c in range(2, len(_COLS_RESUMEN) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 16
    ws.column_dimensions[get_column_letter(len(_COLS_RESUMEN))].width = 26


# ═════════════════════════════════════════════════════════════
#  RAW SHEETS
# ═════════════════════════════════════════════════════════════

def _build_raw(wb, nombre, df):
    ws = wb.create_sheet(nombre)
    cols = list(df.columns)
    for c, col in enumerate(cols, 1):
        _header_cell(ws.cell(1, c), col, _BLUE_DARK)
    for r, (_, fila) in enumerate(df.iterrows(), 2):
        for c, col in enumerate(cols, 1):
            v = fila[col]
            if isinstance(v, (list, tuple)):
                v = str(list(v))
            elif isinstance(v, (np.bool_, bool)):
                v = bool(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            else:
                v = _clean(v) if not isinstance(v, str) else v
            ws.cell(r, c).value = v
    for c, col in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(c)].width = 34 if col in ("ruta", "nota") else 14
    ws.freeze_panes = "A2"
