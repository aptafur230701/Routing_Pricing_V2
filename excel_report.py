"""
excel_report.py
===============
Libro de Excel con los resultados de la evaluación (baselines_drl_ruteo_estocastico.md). Cuatro hojas:
  · Resumen     — por (K, método), en tres grupos (DRL y sus rutas con precio fijo, baselines, cotas): margen medio con
                  IC95, diferencia pareada contra el DRL, gap contra UB-PI, millas cargadas y aceptación; mundos
                  simulados y replay (días reales desde TRAIN_DAYS, fuera de muestra); tiempos fuera de línea.
  · Comparación — la agrega stats_analysis.append_comparacion_to_excel (DRL − cada método, Wilcoxon + Holm).
  · Rutas y precios — agregar_rutas_y_precios: una fila por tramo de cada ciclo del DRL con sus millas y su rango
                  de precio (evaluation.banda_precio sobre la cabeza de precio), más una fila de total por ciclo.
Los datos por mundo van aparte en un CSV (guardar_mundos), no en el Excel.
"""

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from evaluation import resumen_metodos, NOTA_NO_TRATABLE
from stats_analysis import NOTA_UB_DP, NOTA_UB_DP_FLOJA, NOTA_UB_DP_FLOJA_LARGA

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
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    wb.save(output_path)


# ═════════════════════════════════════════════════════════════
#  SUMMARY SHEET
# ═════════════════════════════════════════════════════════════

_COLS_RESUMEN = [
    ("Método",                              "nombre",              None),
    ("Cómo elige el destino",               "destino",             None),
    ("Cómo elige el precio",                "precio",              None),
    ("Escenarios",                          "n_mundos",            "#,##0"),
    ("Margen medio por ciclo (USD)",        "margen_medio",        "#,##0"),
    ("IC 95% inferior (USD)",               "margen_ic95_low",     "#,##0"),
    ("IC 95% superior (USD)",               "margen_ic95_high",    "#,##0"),
    ("Diferencia vs DRL (USD por ciclo)",   "dif_vs_drl_usd",      '+#,##0;-#,##0;"—"'),
    ("Distancia al máximo teórico (%)",     "gap_ub_pi_pct",       '0.0"%"'),
    ("Ciclos que no vuelven a tiempo (%)",  "violacion_pct",       '0.0"%"'),
    ("Millas totales por ciclo",            "millas_totales_medias", "#,##0"),
    ("Millas con carga (%)",                "millas_cargadas_pct", '0.0"%"'),
    ("Ofertas aceptadas (%)",               "tasa_aceptacion_pct", '0.0"%"'),
    ("Esperas por ciclo",                   "esperas_por_ruta",    "0.00"),
    ("Viajes vacíos a otra ciudad por ciclo", "reposiciones_por_ruta", "0.00"),
    ("Cargas por debajo del costo (%)",     "cargas_bajo_be_pct",  '0.0"%"'),
    ("Percentil de precio medio (u)",       "u_medio",             "0.00"),
    ("Precio sobre la mediana (USD/mi)",    "usd_mi_sobre_mediana", '+0.00;-0.00;0.00'),
    ("Nota",                                "nota",                None),
]


_COMO_LEER = (
    "CÓMO LEER ESTA HOJA. Cada fila es un método evaluado en los mismos escenarios. En cada ciclo un camión sale de su "
    "ciudad base y debe volver en 77 horas; en cada ciudad el método decide a dónde ir y qué precio ofrecer por la "
    "carga. La carga se gana si el precio ofrecido es menor que el del mejor competidor: un precio más alto deja más "
    "margen, pero se gana menos veces. Margen por ciclo = ingresos de las cargas ganadas − costo de todos los viajes "
    "(con y sin carga); si el camión no vuelve a tiempo se descuentan 3,000 USD. Diferencia vs DRL = margen del método "
    "− margen del DRL en los mismos escenarios (negativa: el DRL gana más). IC 95%: intervalo de confianza del margen "
    "medio. Distancia al máximo teórico: cuánto margen le falta al método para llegar al máximo posible con "
    "información perfecta. Percentil de precio medio: promedio del nivel (0.1 … 0.9) con que el método cotizó; más alto = "
    "más caro. Precio sobre la mediana: precio cobrado − tarifa mediana publicada de la lane, promedio sobre las "
    "cargas ganadas. Cargas por debajo del costo: ganadas a un precio menor que el costo por milla. La hoja Comparación indica qué diferencias son estadísticamente significativas.")

_TITULO_REAL = "DÍAS REALES DE OCTUBRE 2026 (no se usaron para entrenar): resultado principal"
_TITULO_SIM = "ESCENARIOS SIMULADOS (mercados generados con el modelo de mercado ajustado a los datos)"


def _bloque_resumen(ws, row, titulo, color_oscuro, color_claro, resumen):
    """Escribe una tabla por nivel de K (con subtítulos por grupo); devuelve la siguiente fila libre. El nivel de K
    solo se rotula si hay más de uno (el caso de siempre es K libre, sin límite de tramos)."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
    _header_cell(ws.cell(row, 1), titulo, color_oscuro)
    row += 1
    varios_k = resumen["K"].nunique() > 1
    for K, g in resumen.groupby("K", sort=False):
        if varios_k:
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(_COLS_RESUMEN))
            _header_cell(ws.cell(row, 1), f"Máximo de tramos por ciclo: {K}", color_claro, font_color="000000")
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
            _data_cell(ws.cell(row, 1), f"* {NOTA_UB_DP_FLOJA_LARGA}", fill_hex=_YELLOW)
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
    ws.row_dimensions[row].height = 110
    row += 2

    if replay_df is not None and not replay_df.empty:
        row = _bloque_resumen(ws, row, _TITULO_REAL, _BLUE_DARK, _BLUE_LIGHT, resumen_metodos(replay_df))
    row = _bloque_resumen(ws, row, _TITULO_SIM, _PURPLE_DARK, _PURPLE_LIGHT, resumen_metodos(sim_df))

    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
    _header_cell(ws.cell(row, 1), "TIEMPO DE CÓMPUTO (una sola vez, antes de usar el modelo)", _GREEN_DARK)
    row += 1
    for metrica, clave in (("Entrenamiento del DRL (horas)", "DRL Training Time (s)"),
                           ("Cálculo del «Óptimo con mercado promedio» y de B3 (horas)", "UB-DP Table Time (s)")):
        _data_cell(ws.cell(row, 1), metrica)
        valor = _clean(sr.get(clave))
        valor = "no se calcula a este tamaño" if valor is None and clave == "UB-DP Table Time (s)" else (
            None if valor is None else valor / 3600.0)
        _data_cell(ws.cell(row, 2), valor, "0.0")
        row += 1

    ws.column_dimensions["A"].width = 34
    for c in range(2, len(_COLS_RESUMEN) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 15
    for c in (2, 3):                                                   # destino y precio
        ws.column_dimensions[get_column_letter(c)].width = 40
    ws.column_dimensions[get_column_letter(len(_COLS_RESUMEN))].width = 26


def guardar_mundos(sim_df: pd.DataFrame, replay_df: pd.DataFrame, path: str) -> None:
    """Una fila por (modo, K, mundo, método) en CSV: los datos crudos de la evaluación (resumen_semillas.py los lee)."""
    partes = [sim_df] if replay_df is None or replay_df.empty else [sim_df, replay_df]
    df = pd.concat(partes, ignore_index=True)
    df.drop(columns=[c for c in ("pasos", "cotizaciones") if c in df.columns]).to_csv(path, index=False)


# ═════════════════════════════════════════════════════════════
#  RUTAS Y PRECIOS POR ARCO (DRL)
# ═════════════════════════════════════════════════════════════

_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")

_ETIQUETA_REAL, _ETIQUETA_SIM = "Días reales", "Simulado"

_COMO_LEER_RUTAS = (
    "CÓMO LEER ESTA HOJA. Una fila por tramo de cada ciclo del DRL, en el orden en que se recorre, con una fila de "
    "total al cierre de cada ciclo. Escenario: el caso de prueba, el mismo para todos los métodos. En días reales, un "
    "camión con base en una ciudad que sale un día de octubre de 2026, con las cargas y tarifas publicadas ese día; "
    "en simulado, un mercado completo generado por el modelo de mercado. Precios (USD por milla): la referencia es el "
    "precio al que el DRL ganó la carga, el mismo con que se calcula el margen del ciclo; el rango son los precios "
    "que el DRL consideraba casi igual de buenos en esa decisión (los que juntan el 80% de su preferencia). Todos "
    "los precios son percentiles (p10 … p90) de las tarifas publicadas ese día en esa ruta. Costo por milla: diésel "
    "+ costo operativo (el precio mínimo para no perder dinero). Margen del tramo = ingreso − costo (con carga) o "
    "− costo (vacío); el total del ciclo incluye, si no vuelve a tiempo, la penalización de 3,000 USD. "
    "Ofertas perdidas: cotizaciones rechazadas en ese tramo antes de ganar la carga.")


def _hubs(num_nodes: int) -> list:
    """Identificador de hub de cada nodo (columnas de datos/distance.csv, en el orden del modelo)."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "distance.csv")
    try:
        return list(pd.read_csv(path, index_col=0, nrows=0).columns[:num_nodes])
    except OSError:
        return [str(n) for n in range(num_nodes)]


def _filas_drl(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["metodo"] == "DRL"]
    return d[d["K"] == "libre"] if (d["K"] == "libre").any() else d


def _tabla(ws, row, columnas, filas, color):
    """Encabezado + filas; columnas = [(título, ancho, formato)]; devuelve la siguiente fila libre."""
    for c, (titulo, ancho, _) in enumerate(columnas, 1):
        _header_cell(ws.cell(row, c), titulo, color)
        ws.column_dimensions[get_column_letter(c)].width = ancho
    row += 1
    for valores in filas:
        lineas = 1
        for c, ((_, _, fmt), v) in enumerate(zip(columnas, valores), 1):
            _data_cell(ws.cell(row, c), _clean(v) if not isinstance(v, str) else v, fmt)
            if isinstance(v, str):
                lineas = max(lineas, v.count("\n") + 1)
        ws.row_dimensions[row].height = 15 * lineas + 2
        row += 1
    return row


def agregar_rutas_y_precios(excel_path: str, sim_df: pd.DataFrame, replay_df: pd.DataFrame = None) -> None:
    """Agrega al Excel la hoja 'Rutas y precios': una fila por tramo recorrido de cada ciclo del DRL (con su rango
    de precio, referencia, mediana, costo y millas) y una fila de total por ciclo. Requiere la columna `pasos`."""
    from openpyxl import load_workbook
    from config import FECHA_INICIO
    partes = [(_ETIQUETA_REAL, replay_df), (_ETIQUETA_SIM, sim_df)]
    partes = [(n, _filas_drl(d)) for n, d in partes if d is not None and not d.empty and "pasos" in d]
    if not partes:
        return
    N = int(max(max(r) for _, d in partes for r in d["ruta"] if len(r)) + 1)
    hubs = _hubs(N)
    wb = load_workbook(excel_path)
    for nombre in ("Rutas", "Precios por arco", "Rutas y precios"):
        if nombre in wb.sheetnames:
            del wb[nombre]
    ws = wb.create_sheet("Rutas y precios")
    cols = [("Datos", 13, None), ("Escenario", 10, "0"), ("Ciudad base", 9, "0"), ("Salida", 22, None),
            ("Tramo", 9, None), ("Tipo", 11, None), ("Millas", 9, "#,##0"), ("Ofertas perdidas", 9, "0"),
            ("Rango: desde (USD/mi)", 10, "0.00"), ("Rango: hasta (USD/mi)", 10, "0.00"),
            ("Referencia (USD/mi)", 10, "0.00"), ("Tarifa mediana publicada (USD/mi)", 12, "0.00"),
            ("Costo por milla (USD/mi)", 10, "0.00"), ("Margen del tramo (USD)", 12, "#,##0;-#,##0;0"),
            ("¿Volvió a tiempo?", 10, None)]
    ncol = len(cols)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
    _data_cell(ws.cell(1, 1), _COMO_LEER_RUTAS)
    ws.row_dimensions[1].height = 120
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)
    _data_cell(ws.cell(2, 1), "Ciudades (nodos): " + " · ".join(f"{n} = hub {h}" for n, h in enumerate(hubs)))
    row = 4
    for c, (titulo, ancho, _) in enumerate(cols, 1):
        _header_cell(ws.cell(row, c), titulo, _BLUE_DARK)
        ws.column_dimensions[get_column_letter(c)].width = ancho
    ws.freeze_panes = ws.cell(row + 1, 1)
    row += 1
    for nombre, d in partes:
        for _, r in d.sort_values("mundo").iterrows():
            dia = _DIAS[int(r["dia_semana"])]
            salida = (f"{dia} {(np.datetime64(FECHA_INICIO) + int(r['dia_abs'])).astype(str)}"
                      if nombre == _ETIQUETA_REAL else dia)
            perdidas = {}
            for p in r["pasos"]:
                tramo = (p["desde"], p["hacia"])
                if p["accion"] == "cotiza y pierde":
                    perdidas[tramo] = perdidas.get(tramo, 0) + 1
                    continue
                if p["accion"] == "espera":
                    continue
                gana = p["accion"] == "cotiza y gana"
                rg = p.get("rango") or {}
                valores = (nombre, r["mundo"], r["nodo_inicio"], salida, f"{p['desde']}-{p['hacia']}",
                           "Con carga" if gana else "Vacío", p["millas"], perdidas.pop(tramo, 0) if gana else None,
                           rg.get("piso"), rg.get("tope"),
                           rg.get("referencia", p["precio_cotizado_usd_mi"] if gana else None),
                           p["mediana_usd_mi"] if gana else None, rg.get("break_even"), p["reward_usd"], None)
                for c, ((_, _, fmt), v) in enumerate(zip(cols, valores), 1):
                    _data_cell(ws.cell(row, c), _clean(v) if not isinstance(v, str) else v, fmt)
                row += 1
            ruta = "-".join(str(n) for n in r["ruta"]) if len(r["ruta"]) > 1 else "no sale"
            total = (nombre, r["mundo"], r["nodo_inicio"], salida, f"Ciclo {ruta}", f"{r['n_cargas']} cargas",
                     r["millas_totales"], None, None, None, None, None, None, r["margen"],
                     "no" if r["violacion"] else "sí")
            for c, ((_, _, fmt), v) in enumerate(zip(cols, total), 1):
                cel = ws.cell(row, c)
                _data_cell(cel, _clean(v) if not isinstance(v, str) else v, fmt, fill_hex=_BLUE_LIGHT)
                cel.font = _font(bold=True)
            row += 1
    wb.save(excel_path)


# ═════════════════════════════════════════════════════════════
#  COMPARACIÓN (significancia de las diferencias)
# ═════════════════════════════════════════════════════════════

_COMO_LEER_COMPARACION = (
    "CÓMO LEER ESTA HOJA. Cada fila compara el margen por ciclo del DRL con el de otro método en los mismos "
    "escenarios (diferencia = DRL − método; positiva: el DRL gana más). IC 95%: intervalo de confianza de la "
    "diferencia media. p-valor: probabilidad de ver una diferencia así si en realidad no hubiera ninguna (prueba de "
    "Wilcoxon, ajustada por comparaciones múltiples con Holm); por debajo de 0.05 la diferencia es significativa. "
    "Las filas «mismas rutas del DRL» aíslan el aporte del precio: misma ruta, distinto precio.")

_PREGUNTA = {"ruteo": "DRL frente a los métodos de comparación",
             "precio": "Cuánto aporta el precio del DRL (mismas rutas, otra regla de precio)"}


def escribir_comparacion(excel_path: str, pareada: pd.DataFrame) -> None:
    """Hoja 'Comparación' con nombres legibles: una fila por (datos, comparación), días reales primero."""
    from openpyxl import load_workbook
    from evaluation import NOMBRE
    sig = [c for c in pareada.columns if c.startswith("significativo")]
    d = pareada.copy()
    d["_real"] = ~d["modo"].astype(str).str.startswith("simulado")
    d = d.sort_values("_real", ascending=False, kind="stable")
    wb = load_workbook(excel_path)
    if "Comparación" in wb.sheetnames:
        del wb["Comparación"]
    ws = wb.create_sheet("Comparación", index=1)
    ncol = 9
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
    _data_cell(ws.cell(1, 1), _COMO_LEER_COMPARACION)
    ws.row_dimensions[1].height = 62
    cols = [("Datos", 13, None), ("Pregunta", 34, None), ("DRL comparado con", 34, None), ("Escenarios", 10, "#,##0"),
            ("Diferencia media (USD por ciclo)", 13, '+#,##0;-#,##0;0'), ("IC 95% inferior", 11, '+#,##0;-#,##0;0'),
            ("IC 95% superior", 11, '+#,##0;-#,##0;0'), ("p-valor (Holm)", 11, '[<0.0001]"< 0.0001";0.0000'), ("Conclusión", 34, None)]
    filas = []
    varios_k = d["K"].nunique() > 1
    for _, r in d.iterrows():
        otro = str(r["comparacion"]).split(" − ", 1)[-1]
        nombre = NOMBRE.get(otro, otro) + (f" (máx. {r['K']} tramos)" if varios_k else "")
        signif = bool(r[sig[0]]) if sig else False
        # La prueba (Wilcoxon, una cola) solo detecta que el DRL gana; si el IC bootstrap queda
        # entero por debajo de 0, el método de comparación es mejor aunque p ≈ 1.
        if signif:
            conclusion = "El DRL gana más" if r["dif_media_usd"] > 0 else "El DRL gana menos"
        elif r["ic95_sup"] < 0:
            conclusion = "El DRL gana menos (el IC 95% queda por debajo de 0)"
        else:
            conclusion = "Sin diferencia significativa"
        filas.append((_ETIQUETA_REAL if r["_real"] else _ETIQUETA_SIM, _PREGUNTA.get(r["familia"], r["familia"]),
                      nombre, r["n_mundos"], r["dif_media_usd"], r["ic95_inf"], r["ic95_sup"], r["p_holm"],
                      conclusion))
    _tabla(ws, 3, cols, filas, _GREEN_DARK)
    wb.save(excel_path)
