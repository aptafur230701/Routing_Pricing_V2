"""
excel_report.py
===============
Libro de Excel con los resultados de la evaluación (baselines_drl_ruteo_estocastico.md):
  · Resumen            — por (K, método), en tres grupos (DRL y comparación B, baselines, cotas), con
                         qué decide cada método (destino y precio): margen medio, diferencia pareada
                         contra el DRL, gaps contra UB-DP (política que ignora el mercado, no es cota) y
                         UB-PI (cota), factibilidad, métricas operativas y de cotización; mundos simulados y replay
                         (días reales desde TRAIN_DAYS, fuera de muestra; etiquetado con
                         stats_analysis.NOTA_REPLAY, calculado desde FECHA_INICIO/TRAIN_DAYS); tiempos
                         fuera de línea.
  · Mundos simulados   — una fila por (K, mundo, método), con la ruta y su precio por arco en texto (ruta_precios)
  · Replay             — ídem sobre los días reales de evaluación
  · Arcos simulados / Arcos replay — una fila por paso de cada ruta: tramo, cotización (nivel, percentil u,
                         precio cotizado, mediana, P(ganar), precio del competidor), si se ganó, millas y reward
stats_analysis.append_stats_sheets_to_excel agrega después las hojas estadísticas.
"""

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from evaluation import resumen_metodos, NOTA_NO_TRATABLE
from stats_analysis import NOTA_REPLAY, NOTA_UB_DP, NOTA_UB_DP_FLOJA, NOTA_UB_DP_FLOJA_LARGA

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
    _build_raw(wb, "Arcos simulados", tabla_arcos(sim_df))
    if replay_df is not None and not replay_df.empty:
        _build_raw(wb, "Arcos replay", tabla_arcos(replay_df))
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    wb.save(output_path)


# ═════════════════════════════════════════════════════════════
#  SUMMARY SHEET
# ═════════════════════════════════════════════════════════════

_COLS_RESUMEN = [
    ("Método",              "metodo",              None),
    ("Destino",             "destino",             None),
    ("Precio",              "precio",              None),
    ("Mundos",              "n_mundos",            "0"),
    ("Margen medio (USD)",  "margen_medio",        "#,##0"),
    ("IC95 inf (USD)",      "margen_ic95_low",     "#,##0"),
    ("IC95 sup (USD)",      "margen_ic95_high",    "#,##0"),
    ("Δ margen vs DRL (USD, pareado)", "dif_vs_drl_usd", '+#,##0;-#,##0;0'),
    ("Gap vs UB-DP, ignora el mercado (%)", "gap_ub_dp_pct", '0.00"%"'),
    ("Gap vs UB-PI (%)",    "gap_ub_pi_pct",       '0.00"%"'),
    ("Violación de tiempo (%)", "violacion_pct",   '0.0"%"'),
    ("Millas cargadas (%)", "millas_cargadas_pct", '0.0"%"'),
    ("Cargas por ruta",     "cargas_medias",       "0.00"),
    ("Duración (h)",        "duracion_media_h",    "0.0"),
    ("ms por decisión",     "ms_por_decision",     "0.00"),
    ("Aceptación (%)",      "tasa_aceptacion_pct", '0.0"%"'),
    ("Cotizaciones por arco", "cotizaciones_por_arco", "0.00"),
    ("Horas perdidas en rechazos", "horas_rechazo", "0.00"),
    ("Percentil cotizado medio (u)", "u_medio",   "0.00"),
    ("USD/mi sobre la mediana", "usd_mi_sobre_mediana", "0.000"),
    ("Dinero en la mesa (USD)", "dinero_en_la_mesa", "#,##0"),
    ("Nota",                "nota",                None),
]


_COMO_LEER = (
    "Cada método decide un destino y un precio de cotización por nodo; la carga se gana con probabilidad "
    "(1 − u)^m frente al mejor competidor (u = percentil cotizado, m según el L/T del hub). Margen = Σ (precio "
    "cobrado · millas − costo) de las cargas ganadas − costo de los viajes vacíos. Comparación B: los mismos "
    "destinos del DRL con una regla fija de precio; su Δ vs DRL es lo que aporta la cabeza de precio. "
    "Dinero en la mesa = Σ (precio del competidor − precio cobrado) · millas en las cargas ganadas.")


def _bloque_resumen(ws, row, titulo, color_oscuro, color_claro, resumen):
    """Escribe una tabla por nivel de K (con subtítulos por grupo); devuelve la siguiente fila libre."""
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
        grupo_actual = None
        for _, r in g.iterrows():
            if r.get("grupo") != grupo_actual:
                grupo_actual = r.get("grupo")
                ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
                c = ws.cell(row, 1)
                _data_cell(c, grupo_actual)
                c.font = _font(bold=True)
                row += 1
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
        if (g["metodo"] == "UB-DP").any():
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
            _data_cell(ws.cell(row, 1), NOTA_UB_DP)
            ws.row_dimensions[row].height = 32
            row += 1
        row += 1
    return row


def _build_summary(wb, sim_df, replay_df, summary_rows):
    ws = wb.create_sheet("Resumen")
    sr = summary_rows[0] if summary_rows else {}
    row = 1
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
    _data_cell(ws.cell(row, 1), _COMO_LEER)
    ws.row_dimensions[row].height = 48
    row += 2

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
        valor = _clean(sr.get(clave))
        if valor is None and clave == "UB-DP Table Time (s)":
            valor = "no tratable"
        _data_cell(ws.cell(row, 2), valor, "0.0")
        row += 1

    ws.column_dimensions["A"].width = 30
    for c in range(2, len(_COLS_RESUMEN) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 16
    for c in (2, 3):                                                   # destino y precio
        ws.column_dimensions[get_column_letter(c)].width = 30
    ws.column_dimensions[get_column_letter(len(_COLS_RESUMEN))].width = 26


# ═════════════════════════════════════════════════════════════
#  RAW SHEETS
# ═════════════════════════════════════════════════════════════

_COLS_ARCOS = ("K", "mundo", "metodo", "nodo_inicio", "dia_semana", "paso", "desde", "hacia", "accion", "u",
               "precio_cotizado_usd_mi", "mediana_usd_mi", "p_ganar", "precio_competidor_usd_mi", "millas",
               "reward_usd", "horas_transcurridas")


def tabla_arcos(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por paso de cada ruta (columna `pasos` de evaluation.ejecutar_politica)."""
    if df is None or df.empty or "pasos" not in df:
        return pd.DataFrame(columns=_COLS_ARCOS)
    filas = []
    for _, r in df.iterrows():
        for p in r["pasos"]:
            filas.append({**{c: r[c] for c in ("K", "mundo", "metodo", "nodo_inicio", "dia_semana")}, **p})
    return pd.DataFrame(filas, columns=_COLS_ARCOS)


def _build_raw(wb, nombre, df):
    ws = wb.create_sheet(nombre)
    df = df.drop(columns=[c for c in ("pasos",) if c in df.columns])     # el detalle va en la hoja de arcos
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
        ws.column_dimensions[get_column_letter(c)].width = (90 if col == "ruta_precios" else
                                                            34 if col in ("ruta", "nota", "accion") else 14)
    ws.freeze_panes = "A2"
