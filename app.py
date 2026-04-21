"""
Outil de dimensionnement ACC – Autoconsommation Collective
Croise courbes de charge producteur / consommateur(s) et calcule les taux d'ACC.
Saisons tarifaires : Été = 1 avr – 31 oct | Hiver = 1 nov – 31 mars
"""

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Outil ACC – Autoconsommation Collective",
    layout="wide",
    page_icon="⚡",
)

st.markdown("""
<style>
    .main-title { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0; }
    .sub-title  { font-size: 0.95rem; color: #64748b; margin-top: 0; }
    .kpi-box    { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                  padding: 14px 18px; text-align: center; }
    .kpi-value  { font-size: 1.6rem; font-weight: 700; }
    .kpi-label  { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
    .kpi-green  { color: #16a34a; }
    .kpi-blue   { color: #2563eb; }
    .kpi-orange { color: #ea580c; }
    .kpi-red    { color: #dc2626; }
    .kpi-purple { color: #7c3aed; }
    div[data-testid="stTabs"] button { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers – chargement CSV ──────────────────────────────────────────────────

def _detect_delimiter(text: str) -> str:
    first = text.split("\n")[0]
    return ";" if first.count(";") >= first.count(",") else ","

# Colonnes caractéristiques du format Enedis API (soutirage / injection)
_ENEDIS_COLS = {"start_time", "end_time", "prm", "volume", "unit"}

# Colonnes à ignorer lors de la détection générique (identifiants, codes…)
_SKIP_COLS = {"prm", "pdl", "pce", "site_id", "meter_id", "id", "identifiant",
              "provider", "model", "variant", "data_type", "unit"}


@st.cache_data(show_spinner=False)
def load_curve(file_bytes: bytes, filename: str) -> tuple[pd.Series, dict]:
    """
    Charge un CSV et retourne (Series puissance en kW, métadonnées).

    Formats supportés
    -----------------
    • Format Enedis API : colonnes start_time / volume / unit
      – unit = kWh → P(kW) = volume / dt_heures
      – unit = Wh  → P(kW) = volume / dt_heures / 1000
    • Format générique : première colonne datetime + première colonne numérique
      valide hors identifiants connus.
      – Si colonne "unit" présente, même conversion que ci-dessus.
      – Sinon, valeurs supposées en kW (puissance directe).
    """
    text = file_bytes.decode("utf-8", errors="replace")
    delim = _detect_delimiter(text)

    df = pd.read_csv(StringIO(text), sep=delim, skipinitialspace=True)
    df.columns = [str(c).strip().lower() for c in df.columns]

    meta = {"format": "générique", "unit_source": "kW", "conversion": "aucune"}

    # ── Branche Enedis ────────────────────────────────────────────────────────
    if _ENEDIS_COLS.issubset(set(df.columns)):
        meta["format"] = "Enedis API"

        # Datetime : start_time avec timezone mixte (CET +01 / CEST +02)
        # → parse en UTC d'abord pour éviter l'erreur "mixed time zones"
        # → convertit en Europe/Paris → supprime la tz (index naïf)
        dt = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("Europe/Paris").dt.tz_localize(None)

        values = pd.to_numeric(df["volume"], errors="coerce")
        unit_raw = str(df["unit"].dropna().iloc[0]).strip().lower()
        meta["unit_source"] = unit_raw

        s = pd.Series(values.values, index=pd.DatetimeIndex(dt), name=filename)
        s = s.sort_index().dropna().clip(lower=0)
        # Supprime les doublons de timestamps (changement d'heure été→hiver)
        s = s[~s.index.duplicated(keep="first")]

        # Conversion énergie/intervalle → puissance moyenne kW
        step = s.index.to_series().diff().dropna().mode().iloc[0]
        dt_h = step.total_seconds() / 3600

        # Dans cet export Enedis, le champ "volume" contient la puissance moyenne
        # en kW sur l'intervalle — malgré le label "kWh" dans la colonne unit.
        # Énergie par intervalle = valeur × dt_h  (ex. kW × 5/60 h = kWh)
        # → aucune conversion nécessaire, les valeurs sont déjà en kW.
        if unit_raw in ("wh", "w·h", "w.h"):
            # Si vraiment des Wh/pas : Wh → kW = Wh / (dt_h × 1000)
            s = s / dt_h / 1000
            meta["conversion"] = f"Wh/{int(step.total_seconds()//60)} min → kW"
        else:
            # "kwh", "kw" ou autre : valeurs déjà en kW
            meta["conversion"] = f"kW brut (label unit='{unit_raw}', énergie = valeur × {dt_h:.4f} h)"

        return s, meta

    # ── Branche générique ─────────────────────────────────────────────────────
    # Détection colonne datetime
    dt_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, utc=True)
            if parsed.notna().mean() > 0.8:
                parsed = parsed.dt.tz_convert("Europe/Paris").dt.tz_localize(None)
                df[col] = parsed
                dt_col = col
                break
        except Exception:
            continue
    if dt_col is None:
        raise ValueError(f"Colonne datetime introuvable dans « {filename} »")

    # Détection colonne valeur (premier numérique hors identifiants connus)
    val_col = None
    for col in df.columns:
        if col == dt_col or col in _SKIP_COLS:
            continue
        try:
            if df[col].dtype == object:
                v = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                    errors="coerce",
                )
            else:
                v = pd.to_numeric(df[col], errors="coerce")
            if v.notna().mean() > 0.5:
                df[col] = v
                val_col = col
                break
        except Exception:
            continue
    if val_col is None:
        raise ValueError(f"Colonne valeur introuvable dans « {filename} »")

    s = pd.Series(df[val_col].values, index=pd.DatetimeIndex(df[dt_col]), name=filename)
    s = s.sort_index().dropna().clip(lower=0)
    s = s[~s.index.duplicated(keep="first")]

    # Conversion si colonne unit présente (même logique que branche Enedis)
    if "unit" in df.columns:
        unit_raw = str(df["unit"].dropna().iloc[0]).strip().lower()
        meta["unit_source"] = unit_raw
        step = s.index.to_series().diff().dropna().mode().iloc[0]
        dt_h = step.total_seconds() / 3600
        if unit_raw in ("wh", "w·h", "w.h"):
            s = s / dt_h / 1000
            meta["conversion"] = f"Wh → kW"
        else:
            meta["conversion"] = f"kW brut (label='{unit_raw}')"

    return s, meta


def detect_step(s: pd.Series) -> pd.Timedelta:
    diffs = s.index.to_series().diff().dropna()
    return diffs.mode().iloc[0]


def resample_series(s: pd.Series, target: pd.Timedelta, source: pd.Timedelta) -> pd.Series:
    """
    Rééchantillonnage vers le pas cible.
    • Upsampling (target < source) : forward-fill — puissance constante sur l'intervalle
      → l'énergie est conservée : P × Δt reste identique.
    • Downsampling (target > source) : moyenne — préserve la puissance moyenne.
    """
    freq = f"{int(target.total_seconds() // 60)}min"
    if target <= source:
        return s.resample(freq).ffill()
    return s.resample(freq).mean()


def _detect_gaps(s: pd.Series) -> dict:
    """Détecte les trous (NaN consécutifs) dans une série après reindex."""
    missing = s.isna()
    n_missing = int(missing.sum())
    if n_missing == 0:
        return {"n_missing": 0, "pct": 0.0, "periods": []}
    # Regroupe les NaN consécutifs
    groups = (missing != missing.shift()).cumsum()
    periods = []
    for _, grp in missing.groupby(groups):
        if grp.iloc[0]:
            periods.append((grp.index[0], grp.index[-1]))
    return {
        "n_missing": n_missing,
        "pct": n_missing / len(s) * 100,
        "periods": periods[:5],   # on affiche max 5 périodes
    }


def align_series(series_dict: dict, producer_keys: list):
    """
    Aligne toutes les séries sur le pas le plus fin et la plage commune.

    Remplissage des trous :
    • Producteurs  → fillna(0)   — absence de données = pas de production
    • Consommateurs → ffill (max 1 h) puis fillna(0)
                     — trou = télérelève absente, on propage la dernière valeur connue
    Retourne aussi un dict de métadonnées sur les trous détectés.
    """
    steps = {k: detect_step(v) for k, v in series_dict.items()}
    target = min(steps.values())
    freq = f"{int(target.total_seconds() // 60)}min"
    ffill_limit = max(1, int(pd.Timedelta("1h") / target))  # 1 h en nb de pas

    resampled = {k: resample_series(v, target, steps[k]) for k, v in series_dict.items()}

    # La plage de référence est déterminée par les consommateurs (intersection).
    # Les producteurs hors de cette plage contribuent 0 — pas de production = pas d'ACC,
    # mais on ne pénalise pas la plage de calcul à cause d'un producteur plus court.
    conso_series = [s for k, s in resampled.items() if k not in producer_keys]
    ref_series   = conso_series if conso_series else list(resampled.values())
    start = max(s.index.min() for s in ref_series)
    end   = min(s.index.max() for s in ref_series)
    idx   = pd.date_range(start=start, end=end, freq=freq)

    aligned = {}
    gaps    = {}
    for k, s in resampled.items():
        s_reindexed = s.reindex(idx)
        gaps[k] = _detect_gaps(s_reindexed)
        if k in producer_keys:
            aligned[k] = s_reindexed.fillna(0).clip(lower=0)
        else:
            aligned[k] = s_reindexed.ffill(limit=ffill_limit).fillna(0).clip(lower=0)

    return aligned, target, steps, gaps


# ── Calcul ACC ────────────────────────────────────────────────────────────────

def saison(month: int) -> str:
    return "Été (avr–oct)" if 4 <= month <= 10 else "Hiver (nov–mars)"


def compute_acc(aligned: dict, producer_keys: list, timestep: pd.Timedelta):
    """
    Calcule les flux ACC et les indicateurs.
    Supporte plusieurs producteurs (la production est sommée).
    Allocation dynamique : chaque consommateur reçoit une part de l'ACC
    proportionnelle à sa consommation instantanée.
    """
    dt_h = timestep.total_seconds() / 3600  # heures par pas
    consumer_keys = [k for k in aligned if k not in producer_keys]

    df = pd.DataFrame(aligned)
    df["P_prod"]     = df[producer_keys].sum(axis=1)   # somme de tous les producteurs
    df["P_conso"]    = df[consumer_keys].sum(axis=1)
    df["P_acc"]      = np.minimum(df["P_prod"], df["P_conso"])
    df["P_surplus"]  = (df["P_prod"] - df["P_conso"]).clip(lower=0)
    df["P_deficit"]  = (df["P_conso"] - df["P_prod"]).clip(lower=0)

    # Allocation dynamique par consommateur
    for k in consumer_keys:
        mask = df["P_conso"] > 0
        df[f"P_acc_{k}"] = 0.0
        df.loc[mask, f"P_acc_{k}"] = (
            df.loc[mask, "P_acc"] * df.loc[mask, k] / df.loc[mask, "P_conso"]
        )

    # ── Énergie ──────────────────────────────────────────────────────────────
    E = (df[["P_prod", "P_conso", "P_acc", "P_surplus", "P_deficit"]] * dt_h).copy()
    E.columns = ["E_prod", "E_conso", "E_acc", "E_surplus", "E_deficit"]
    E["saison"] = E.index.month.map(saison)

    def _row(mask=None):
        sub = E if mask is None else E.loc[mask]
        ep, ec, ea = sub["E_prod"].sum(), sub["E_conso"].sum(), sub["E_acc"].sum()
        return {
            "Prod (kWh)":            round(ep),
            "Conso (kWh)":           round(ec),
            "ACC (kWh)":             round(ea),
            "Surplus réseau (kWh)":  round(sub["E_surplus"].sum()),
            "Déficit réseau (kWh)":  round(sub["E_deficit"].sum()),
            "Taux ACC prod (%)":     round(ea / ep * 100, 1) if ep > 0 else 0.0,
            "Taux ACC conso (%)":    round(ea / ec * 100, 1) if ec > 0 else 0.0,
        }

    annual = _row()

    months = E.index.to_period("M")
    monthly = pd.DataFrame(
        [dict(Mois=str(p), **_row(months == p)) for p in sorted(months.unique())]
    ).set_index("Mois")

    seasonal = pd.DataFrame(
        [dict(Saison=s, **_row(E["saison"] == s)) for s in ["Été (avr–oct)", "Hiver (nov–mars)"]]
    ).set_index("Saison")

    return df, E, annual, monthly, seasonal


# ── Tarification & économies ──────────────────────────────────────────────────

def build_tariff_series(
    idx: pd.DatetimeIndex,
    option: str,
    hc_start: datetime.time,
    hc_end: datetime.time,
    rates: dict,
) -> pd.Series:
    """
    Retourne une série (€/kWh) du tarif fourniture pour chaque pas de temps,
    selon l'option tarifaire (Monomial / HP-HC / HPH-HCH-HPE-HCE).
    La plage HC peut enjamber minuit (ex. 22 h → 6 h).
    """
    # Rates en €/MWh → conversion €/kWh
    r = {k: v / 1000.0 for k, v in rates.items()}

    if option == "Monomial":
        return pd.Series(r["mono"], index=idx, dtype=float)

    hour_cont = idx.hour + idx.minute / 60.0
    hs = hc_start.hour + hc_start.minute / 60.0
    he = hc_end.hour   + hc_end.minute   / 60.0
    is_hc = (hour_cont >= hs) | (hour_cont < he) if hs > he else (hour_cont >= hs) & (hour_cont < he)

    if option == "HP / HC":
        return pd.Series(np.where(is_hc, r["hc"], r["hp"]), index=idx, dtype=float)

    # HPH · HCH · HPE · HCE
    is_ete = (idx.month >= 4) & (idx.month <= 10)
    tarif = np.select(
        [is_hc & ~is_ete, ~is_hc & ~is_ete, is_hc & is_ete, ~is_hc & is_ete],
        [r["hch"],        r["hph"],          r["hce"],       r["hpe"]],
    )
    return pd.Series(tarif, index=idx, dtype=float)


def compute_economics(
    df_flows: pd.DataFrame,
    consumer_keys: list,
    timestep: pd.Timedelta,
    tariff_series: pd.Series,
    cee: float,
    gc: float,
    accises_mwh: float,
    prix_cession: float,
):
    """
    Calcule les économies estimées par consommateur entrant en ACC,
    avec décomposition par composante tarifaire. Tout en HT —
    la TVA s'applique identiquement en ACC et en fourniture classique.

    Par pas de temps i :
        eco_fourniture(i) = E_ACC(i) × tarif_fourniture(i)   [variable HP/HC]
        eco_cee(i)        = E_ACC(i) × CEE/1000
        eco_gc(i)         = E_ACC(i) × GC/1000
        eco_accises(i)    = E_ACC(i) × Accises/1000
        économie_HT(i)    = somme des 4 composantes
        achat_acc(i)      = E_ACC(i) × prix_cession/1000
        économie_nette(i) = économie_HT(i) − achat_acc(i)
    """
    dt_h         = timestep.total_seconds() / 3600
    cee_kwh      = cee          / 1000.0
    gc_kwh       = gc           / 1000.0
    acc_kwh      = accises_mwh  / 1000.0
    prix_ces_kwh = prix_cession / 1000.0

    annual_eco  = {}
    eco_monthly = {}

    for k in consumer_keys:
        E_acc = df_flows[f"P_acc_{k}"] * dt_h          # kWh/pas

        eco_fourniture = E_acc * tariff_series.values   # fourniture (HP/HC)
        eco_cee        = E_acc * cee_kwh
        eco_gc         = E_acc * gc_kwh
        eco_accises    = E_acc * acc_kwh
        eco_ht         = eco_fourniture + eco_cee + eco_gc + eco_accises
        achat_acc      = E_acc * prix_ces_kwh
        eco_nette      = eco_ht - achat_acc

        mi = E_acc.index.to_period("M")
        monthly_k = pd.DataFrame({
            "E_acc (kWh)":               E_acc.groupby(mi).sum(),
            "dont Fourniture HT (€)":    eco_fourniture.groupby(mi).sum(),
            "dont CEE HT (€)":           eco_cee.groupby(mi).sum(),
            "dont GC HT (€)":            eco_gc.groupby(mi).sum(),
            "dont Accises HT (€)":       eco_accises.groupby(mi).sum(),
            "Coût sans ACC HT (€)":      eco_ht.groupby(mi).sum(),
            "Achat ACC HT (€)":          achat_acc.groupby(mi).sum(),
            "Économie grâce à l'ACC HT (€)": eco_nette.groupby(mi).sum(),
        })
        monthly_k.index = monthly_k.index.astype(str)
        eco_monthly[k] = monthly_k

        e_tot  = E_acc.sum()
        ht_tot = eco_ht.sum()
        annual_eco[k] = {
            "E_acc (kWh)":                   round(e_tot),
            "dont Fourniture HT (€)":        round(eco_fourniture.sum(), 0),
            "dont CEE HT (€)":               round(eco_cee.sum(), 0),
            "dont GC HT (€)":                round(eco_gc.sum(), 0),
            "dont Accises HT (€)":           round(eco_accises.sum(), 0),
            "Coût sans ACC HT (€)":          round(ht_tot, 0),
            "Achat ACC HT (€)":              round(achat_acc.sum(), 0),
            "Économie grâce à l'ACC HT (€)": round(eco_nette.sum(), 0),
            "Gain net (€/MWh)":              round(eco_nette.sum() / e_tot * 1000, 1) if e_tot > 0 else 0.0,
        }

    return annual_eco, eco_monthly


# ── Visualisations Plotly ─────────────────────────────────────────────────────

COLORS = {
    "prod":    "#16a34a",
    "conso":   "#2563eb",
    "acc":     "rgba(34,197,94,0.35)",
    "surplus": "rgba(234,179,8,0.40)",
    "deficit": "rgba(239,68,68,0.35)",
    "prod_line":    "#16a34a",
    "conso_line":   "#2563eb",
}

LAYOUT_BASE = dict(
    font=dict(family="Inter, sans-serif", size=12),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _add_rangebar(fig):
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=1,  label="1 j",  step="day",   stepmode="backward"),
            dict(count=7,  label="1 sem", step="day",  stepmode="backward"),
            dict(count=1,  label="1 mois",step="month",stepmode="backward"),
            dict(step="all", label="Tout"),
        ]),
        rangeslider=dict(visible=True, thickness=0.06),
        type="date",
    )


def chart_raw_curves(series_dict: dict, steps: dict, producer_keys: list) -> go.Figure:
    """Courbes brutes en sous-graphiques pour vérification. Producteurs en vert, consommateurs en bleu."""
    n = len(series_dict)
    labels = list(series_dict.keys())
    palette_prod  = ["#16a34a", "#15803d", "#166534", "#4ade80", "#86efac"]
    palette_conso = ["#2563eb", "#7c3aed", "#0891b2", "#ea580c", "#db2777"]
    prod_idx  = 0
    conso_idx = 0

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=[
            f"{'[PROD] ' if k in producer_keys else '[CONSO] '}{k}  ·  pas {int(steps[k].total_seconds()//60)} min"
            for k in labels
        ],
        vertical_spacing=0.08,
    )
    for i, k in enumerate(labels, 1):
        s = series_dict[k]
        if k in producer_keys:
            color = palette_prod[prod_idx % len(palette_prod)]
            prod_idx += 1
        else:
            color = palette_conso[conso_idx % len(palette_conso)]
            conso_idx += 1
        fig.add_trace(
            go.Scattergl(
                x=s.index, y=s.values,
                mode="lines",
                line=dict(color=color, width=1),
                name=k,
                hovertemplate="%{y:.1f} kW<extra></extra>",
            ),
            row=i, col=1,
        )
        fig.update_yaxes(title_text="kW", row=i, col=1, gridcolor="#f1f5f9")
    fig.update_layout(**LAYOUT_BASE, height=max(260 * n, 380), showlegend=False, title_text="Vérification des courbes de charge brutes")
    _add_rangebar(fig)
    return fig


def chart_comparison(df: pd.DataFrame, display_freq: str) -> go.Figure:
    """Production vs Consommation totale – superposition."""
    agg = df[["P_prod", "P_conso"]].resample(display_freq).mean().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["P_prod"],
        mode="lines", line=dict(color=COLORS["prod_line"], width=1.5),
        name="Production", hovertemplate="%{y:.1f} kW<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["P_conso"],
        mode="lines", line=dict(color=COLORS["conso_line"], width=1.5),
        name="Consommation totale", hovertemplate="%{y:.1f} kW<extra></extra>",
    ))
    fig.update_yaxes(title_text="kW (moyenne)", gridcolor="#f1f5f9")
    fig.update_layout(**LAYOUT_BASE, title_text="Production vs Consommation", height=420)
    _add_rangebar(fig)
    return fig


def chart_acc_flows(df: pd.DataFrame, display_freq: str) -> go.Figure:
    """
    Visualisation des flux ACC.
    Zone verte  = énergie autoconsommée (ACC)
    Zone jaune  = surplus injecté réseau
    Zone rouge  = déficit soutiré réseau
    """
    agg = df[["P_prod", "P_conso", "P_acc", "P_surplus", "P_deficit"]].resample(display_freq).mean().dropna()
    t = agg.index

    prod   = agg["P_prod"].values
    conso  = agg["P_conso"].values
    acc    = agg["P_acc"].values
    surplus_upper = np.where(prod > conso, prod, conso)   # prod quand surplus, sinon conso (pas de remplissage)
    deficit_upper = conso                                  # conso quand déficit
    deficit_lower = np.where(conso > prod, prod, conso)   # prod quand déficit, sinon conso (pas de remplissage)

    fig = go.Figure()

    # ① ACC (vert, de 0 à min(prod, conso))
    fig.add_trace(go.Scatter(
        x=t, y=acc,
        fill="tozeroy", fillcolor=COLORS["acc"],
        line=dict(width=0), name="ACC (autoconsommé)",
        hovertemplate="%{y:.1f} kW<extra></extra>",
    ))

    # ② Surplus (jaune, de conso à prod quand prod > conso)
    fig.add_trace(go.Scatter(
        x=t, y=conso,
        fill=None, line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=surplus_upper,
        fill="tonexty", fillcolor=COLORS["surplus"],
        line=dict(width=0), name="Surplus → réseau",
        hovertemplate="%{y:.1f} kW<extra></extra>",
    ))

    # ③ Déficit (rouge, de prod à conso quand conso > prod)
    fig.add_trace(go.Scatter(
        x=t, y=deficit_lower,
        fill=None, line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=deficit_upper,
        fill="tonexty", fillcolor=COLORS["deficit"],
        line=dict(width=0), name="Déficit ← réseau",
        hovertemplate="%{y:.1f} kW<extra></extra>",
    ))

    # ④ Lignes par-dessus
    fig.add_trace(go.Scattergl(
        x=t, y=prod,
        mode="lines", line=dict(color=COLORS["prod_line"], width=1.5),
        name="Production", hovertemplate="%{y:.1f} kW<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=t, y=conso,
        mode="lines", line=dict(color=COLORS["conso_line"], width=1.5),
        name="Consommation", hovertemplate="%{y:.1f} kW<extra></extra>",
    ))

    fig.update_yaxes(title_text="kW (moyenne)", gridcolor="#f1f5f9")
    fig.update_layout(**LAYOUT_BASE, title_text="Flux ACC – Production / Consommation / Échanges réseau", height=450)
    _add_rangebar(fig)
    return fig


def chart_monthly_bars(monthly: pd.DataFrame) -> go.Figure:
    """Barres empilées mensuelles + ligne taux ACC."""
    months = monthly.index.tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=months, y=monthly["ACC (kWh)"],
        name="ACC (kWh)", marker_color="#16a34a", opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=months, y=monthly["Surplus réseau (kWh)"],
        name="Surplus réseau (kWh)", marker_color="#ca8a04", opacity=0.75,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=months, y=monthly["Déficit réseau (kWh)"],
        name="Déficit réseau (kWh)", marker_color="#dc2626", opacity=0.65,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=months, y=monthly["Taux ACC conso (%)"],
        mode="lines+markers", name="Taux ACC conso (%)",
        line=dict(color="#7c3aed", width=2.5, dash="dot"),
        marker=dict(size=6),
        hovertemplate="%{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=months, y=monthly["Taux ACC prod (%)"],
        mode="lines+markers", name="Taux ACC prod (%)",
        line=dict(color="#0891b2", width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="%{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **LAYOUT_BASE, barmode="stack",
        title_text="Bilan mensuel – Énergie & Taux d'ACC",
        height=420,
    )
    fig.update_yaxes(title_text="Énergie (kWh)", secondary_y=False, gridcolor="#f1f5f9")
    fig.update_yaxes(title_text="Taux ACC (%)", secondary_y=True, range=[0, 105], gridcolor=None)
    return fig


def chart_economics_monthly(eco_monthly: dict, consumer_keys: list) -> go.Figure:
    """
    Barres empilées par composante tarifaire (HT) + ligne économie nette TTC.
    Si plusieurs consommateurs : barres groupées (total de chaque conso).
    """
    COMP_COLORS = {
        "dont Fourniture HT (€)": "#2563eb",
        "dont CEE HT (€)":        "#7c3aed",
        "dont GC HT (€)":         "#0891b2",
        "dont Accises HT (€)":    "#ca8a04",
    }
    COMP_LABELS = {
        "dont Fourniture HT (€)": "Fourniture",
        "dont CEE HT (€)":        "CEE",
        "dont GC HT (€)":         "GC",
        "dont Accises HT (€)":    "Accises",
    }
    NET_COLORS = ["#16a34a", "#ea580c", "#db2777", "#166534"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if len(consumer_keys) == 1:
        k      = consumer_keys[0]
        df     = eco_monthly[k]
        months = df.index.tolist()

        for col, color in COMP_COLORS.items():
            fig.add_trace(go.Bar(
                x=months, y=df[col].round(0),
                name=COMP_LABELS[col], marker_color=color, opacity=0.85,
                hovertemplate=f"{COMP_LABELS[col]} : %{{y:,.0f}} €<extra></extra>",
            ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=months, y=df["Économie grâce à l'ACC HT (€)"].round(0),
            mode="lines+markers", name="Économie ACC HT",
            line=dict(color="#16a34a", width=2.5), marker=dict(size=6),
            hovertemplate="Économie ACC : %{y:,.0f} €<extra></extra>",
        ), secondary_y=False)

    else:
        # Multi-conso : barres groupées par consommateur (total HT)
        for i, k in enumerate(consumer_keys):
            df     = eco_monthly[k]
            months = df.index.tolist()
            color  = NET_COLORS[i % len(NET_COLORS)]
            fig.add_trace(go.Bar(
                x=months, y=df["Total éco. HT (€)"].round(0),
                name=k, marker_color=color, opacity=0.85,
                hovertemplate=f"{k} HT : %{{y:,.0f}} €<extra></extra>",
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=months, y=df["Économie grâce à l'ACC HT (€)"].round(0),
                mode="lines+markers", name=f"Économie ACC – {k}",
                line=dict(color=color, width=2, dash="dot"), marker=dict(size=5),
                hovertemplate="Économie ACC : %{y:,.0f} €<extra></extra>",
            ), secondary_y=False)

    fig.update_layout(
        **LAYOUT_BASE, barmode="stack",
        title_text="Décomposition mensuelle du coût évité (sans ACC) et économie ACC",
        height=440,
    )
    fig.update_yaxes(title_text="Économies (€)", secondary_y=False, gridcolor="#f1f5f9")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ Outil ACC")
    st.caption("Autoconsommation collective – Dimensionnement énergétique")
    st.divider()

    st.markdown("**1 · Charger le(s) producteur(s)**")
    prod_files = st.file_uploader(
        "Courbe(s) de charge producteur(s) (.csv)",
        type="csv", accept_multiple_files=True, key="prod",
    )

    st.markdown("**2 · Charger le(s) consommateur(s)**")
    conso_files = st.file_uploader(
        "Courbe(s) de charge consommateur(s) (.csv)",
        type="csv", accept_multiple_files=True, key="conso",
    )

    st.divider()
    st.markdown("**Résolution d'affichage**")
    display_options = {
        "5 min (brut)": "5min",
        "15 min":       "15min",
        "30 min":       "30min",
        "1 heure":      "1h",
        "4 heures":     "4h",
        "1 jour":       "1D",
    }
    display_label = st.selectbox(
        "Agrégation pour les graphiques",
        list(display_options.keys()),
        index=3,
        help="Les calculs sont toujours faits au pas natif. Seul l'affichage est agrégé.",
    )
    display_freq = display_options[display_label]

    st.divider()
    st.markdown("**3 · Paramètres tarifaires**")

    tariff_option = st.radio(
        "Structure tarifaire",
        ["Monomial", "HP / HC", "HPH · HCH · HPE · HCE"],
        help="Option tarifaire du contrat de fourniture du consommateur",
    )

    if tariff_option != "Monomial":
        st.caption("Plages heures creuses (HC)")
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            hc_start = st.time_input("HC de", datetime.time(22, 0), step=1800)
        with _tc2:
            hc_end = st.time_input("HC à",  datetime.time(6,  0), step=1800)
    else:
        hc_start = datetime.time(22, 0)
        hc_end   = datetime.time(6,  0)

    st.caption("Tarif(s) de fourniture HT (€/MWh)")

    if tariff_option == "Monomial":
        _mono = st.number_input("Tarif fourniture HT (€/MWh)", min_value=0.0,
                                value=120.0, step=0.5, format="%.2f")
        tariff_rates = {"mono": _mono}

    elif tariff_option == "HP / HC":
        _t1, _t2 = st.columns(2)
        with _t1:
            _hp = st.number_input("HP (€/MWh)", min_value=0.0, value=145.0,
                                  step=0.5, format="%.2f")
        with _t2:
            _hc = st.number_input("HC (€/MWh)", min_value=0.0, value=100.0,
                                  step=0.5, format="%.2f")
        tariff_rates = {"hp": _hp, "hc": _hc}

    else:  # HPH · HCH · HPE · HCE
        _t1, _t2 = st.columns(2)
        with _t1:
            _hph = st.number_input("HPH (€/MWh)", min_value=0.0, value=170.0,
                                   step=0.5, format="%.2f")
            _hpe = st.number_input("HPE (€/MWh)", min_value=0.0, value=130.0,
                                   step=0.5, format="%.2f")
        with _t2:
            _hch = st.number_input("HCH (€/MWh)", min_value=0.0, value=120.0,
                                   step=0.5, format="%.2f")
            _hce = st.number_input("HCE (€/MWh)", min_value=0.0, value=90.0,
                                   step=0.5, format="%.2f")
        tariff_rates = {"hph": _hph, "hch": _hch, "hpe": _hpe, "hce": _hce}

    st.caption("Taxes & contributions (hors TURPE) — €/MWh")
    cee = st.number_input("CEE (€/MWh)", min_value=0.0, value=1.5,
                          step=0.1, format="%.2f",
                          help="Certificats d'économies d'énergie")
    gc  = st.number_input("GC – Garanties de capacité (€/MWh)", min_value=0.0,
                          value=3.0, step=0.1, format="%.2f")
    accises_mwh = st.number_input(
        "Accises sur l'électricité (€/MWh)", min_value=0.0, value=21.0,
        step=0.5, format="%.2f",
        help="Ancienne TICFE — 21 €/MWh taux standard entreprises",
    )

    st.divider()
    st.markdown("**Prix de cession ACC**")
    st.caption("Ce que le consommateur paye au producteur pour l'ACC")
    prix_cession = st.number_input(
        "Prix de cession HT (€/MWh)", min_value=0.0, value=70.0,
        step=0.5, format="%.2f",
        help="Laisser à 0 si non défini",
    )

    st.divider()
    st.caption("💡 Les courbes peuvent avoir des pas de temps différents — elles seront automatiquement rééchantillonnées au pas le plus fin.")


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">Outil de dimensionnement ACC</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Autoconsommation collective · Taux d\'ACC annuel / mensuel / saisonnier</p>', unsafe_allow_html=True)

if not prod_files or not conso_files:
    st.info("Chargez au minimum **1 producteur** et **1 consommateur** dans la barre latérale pour commencer.", icon="📂")
    st.stop()

# ── Chargement ────────────────────────────────────────────────────────────────

with st.spinner("Chargement et traitement des courbes de charge…"):
    try:
        raw = {}        # series (kW)
        raw_meta = {}   # métadonnées de chargement
        producer_keys = []
        for f in prod_files:
            s, m = load_curve(f.read(), f.name)
            raw[f.name]      = s
            raw_meta[f.name] = m
            producer_keys.append(f.name)
        for f in conso_files:
            s, m = load_curve(f.read(), f.name)
            raw[f.name]      = s
            raw_meta[f.name] = m
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()

    try:
        aligned, timestep, raw_steps, gaps = align_series(raw, producer_keys)
        df_flows, df_energy, annual, monthly, seasonal = compute_acc(aligned, producer_keys, timestep)
    except Exception as e:
        st.error(f"Erreur lors du calcul ACC : {e}")
        st.stop()

consumer_keys    = [k for k in aligned if k not in producer_keys]
step_aligned_min = int(timestep.total_seconds() // 60)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_data, tab_curves, tab_acc, tab_synth, tab_eco = st.tabs([
    "📋 Vérification données",
    "📈 Courbes croisées",
    "⚡ Flux ACC",
    "📊 Synthèse",
    "💰 Économies",
])

# ─── Tab 1 : Vérification données ────────────────────────────────────────────

with tab_data:
    st.subheader("Métadonnées des fichiers chargés")

    meta_rows = []
    for name, s in raw.items():
        role = f"Producteur {producer_keys.index(name) + 1}" if name in producer_keys else "Consommateur"
        step_min = int(raw_steps[name].total_seconds() // 60)
        m = raw_meta[name]
        e_total = (s * raw_steps[name].total_seconds() / 3600).sum()
        meta_rows.append({
            "Fichier":              name,
            "Rôle":                 role,
            "Format détecté":       m.get("format", "—"),
            "Unité source":         m.get("unit_source", "—"),
            "Conversion appliquée": m.get("conversion", "—"),
            "Pas de temps":         f"{step_min} min",
            "Début":                s.index.min().strftime("%d/%m/%Y %H:%M"),
            "Fin":                  s.index.max().strftime("%d/%m/%Y %H:%M"),
            "Nb de points":         f"{len(s):,}".replace(",", "\u202f"),
            "Min (kW)":             f"{s.min():.2f}",
            "Max (kW)":             f"{s.max():.2f}",
            "Moyenne (kW)":         f"{s.mean():.2f}",
            "Total énergie (kWh)":  f"{e_total:,.0f}".replace(",", "\u202f"),
        })

    st.dataframe(pd.DataFrame(meta_rows).set_index("Fichier"), use_container_width=True)

    # ── Avertissements trous ──────────────────────────────────────────────────
    files_with_gaps = {k: v for k, v in gaps.items() if v["n_missing"] > 0}
    if files_with_gaps:
        st.warning(
            f"{len(files_with_gaps)} fichier(s) contiennent des trous dans les données. "
            "Voir le détail ci-dessous.",
            icon="⚠️",
        )
        for name, g in files_with_gaps.items():
            is_prod = name in producer_keys
            strategie = "comblés à **0** (pas de production)" if is_prod else "comblés par **propagation** de la dernière valeur connue (max 1 h), puis 0"
            with st.expander(f"{'[PROD]' if is_prod else '[CONSO]'} {name} — {g['n_missing']:,} points manquants ({g['pct']:.1f}%)".replace(",", "\u202f")):
                st.caption(f"Stratégie : trous {strategie}")
                if g["periods"]:
                    rows_gap = [{"Début": s.strftime("%d/%m/%Y %H:%M"), "Fin": e.strftime("%d/%m/%Y %H:%M"),
                                 "Durée": str(e - s)} for s, e in g["periods"]]
                    st.dataframe(pd.DataFrame(rows_gap), use_container_width=True, hide_index=True)
                    if len(g["periods"]) == 5:
                        st.caption("(5 premières périodes affichées)")
    else:
        st.success("Aucun trou détecté dans les données.", icon="✅")

    plage = (
        f"{aligned[producer_keys[0]].index.min().strftime('%d/%m/%Y')} → "
        f"{aligned[producer_keys[0]].index.max().strftime('%d/%m/%Y')}"
    )
    st.success(
        f"Plage commune : **{plage}**  ·  "
        f"Pas de calcul retenu : **{step_aligned_min} min**  ·  "
        f"{len(df_flows):,} pas de temps".replace(",", "\u202f"),
        icon="✅",
    )

    st.divider()
    st.subheader("Courbes brutes (valeurs converties en kW)")
    st.plotly_chart(chart_raw_curves(raw, raw_steps, producer_keys), use_container_width=True)

    # Légende stratégie de remplissage sous le graphe
    st.markdown("**Stratégie de remplissage des trous appliquée :**")
    cols = st.columns(len(raw))
    for col, (name, s) in zip(cols, raw.items()):
        is_prod = name in producer_keys
        g = gaps[name]
        role_label = f"🟢 {name}" if is_prod else f"🔵 {name}"
        if g["n_missing"] == 0:
            status = "✅ Aucun trou"
            detail = "—"
        else:
            status = f"⚠️ {g['n_missing']:,} pts manquants ({g['pct']:.1f}%)".replace(",", "\u202f")
            detail = "Rempli à **0**" if is_prod else "**Propagation** dernière valeur (max 1 h), puis 0"
        with col:
            st.markdown(f"""<div class="kpi-box">
                <div style="font-size:0.78rem;font-weight:600;color:#475569">{role_label}</div>
                <div style="font-size:0.82rem;margin-top:4px">{status}</div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:2px">{detail}</div>
            </div>""", unsafe_allow_html=True)

# ─── Tab 2 : Courbes croisées ─────────────────────────────────────────────────

with tab_curves:
    st.subheader("Production vs Consommation totale")
    st.caption(f"Affichage agrégé à : **{display_label}**  ·  Calculs effectués au pas natif ({step_aligned_min} min)")
    st.plotly_chart(chart_comparison(df_flows, display_freq), use_container_width=True)

    if len(producer_keys) > 1:
        st.divider()
        st.subheader("Détail par producteur")
        fig_prod = go.Figure()
        palette_prod = ["#16a34a", "#15803d", "#4ade80", "#86efac", "#166534"]
        agg_prod = df_flows[producer_keys].resample(display_freq).mean().dropna()
        for i, k in enumerate(producer_keys):
            fig_prod.add_trace(go.Scattergl(
                x=agg_prod.index, y=agg_prod[k],
                mode="lines", line=dict(color=palette_prod[i % len(palette_prod)], width=1.5),
                name=k, hovertemplate="%{y:.1f} kW<extra></extra>",
            ))
        fig_prod.update_yaxes(title_text="kW (moyenne)", gridcolor="#f1f5f9")
        fig_prod.update_layout(**LAYOUT_BASE, title_text="Courbes producteurs individuels", height=380)
        _add_rangebar(fig_prod)
        st.plotly_chart(fig_prod, use_container_width=True)

    if len(consumer_keys) > 1:
        st.divider()
        st.subheader("Détail par consommateur")
        fig_detail = go.Figure()
        palette_conso = ["#2563eb", "#7c3aed", "#0891b2", "#ea580c", "#db2777"]
        agg_detail = df_flows[consumer_keys].resample(display_freq).mean().dropna()
        for i, k in enumerate(consumer_keys):
            fig_detail.add_trace(go.Scattergl(
                x=agg_detail.index, y=agg_detail[k],
                mode="lines", line=dict(color=palette_conso[i % len(palette_conso)], width=1.5),
                name=k, hovertemplate="%{y:.1f} kW<extra></extra>",
            ))
        fig_detail.update_yaxes(title_text="kW (moyenne)", gridcolor="#f1f5f9")
        fig_detail.update_layout(**LAYOUT_BASE, title_text="Courbes consommateurs", height=380)
        _add_rangebar(fig_detail)
        st.plotly_chart(fig_detail, use_container_width=True)

# ─── Tab 3 : Flux ACC ────────────────────────────────────────────────────────

with tab_acc:
    st.subheader("Analyse des flux d'autoconsommation")
    st.caption(
        "**Vert** = énergie autoconsommée (ACC)   |   "
        "**Jaune** = surplus injecté au réseau   |   "
        "**Rouge** = déficit soutiré au réseau"
    )
    st.plotly_chart(chart_acc_flows(df_flows, display_freq), use_container_width=True)

    # KPIs rapides
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-green">{annual['Taux ACC prod (%)']:.1f} %</div>
            <div class="kpi-label">Taux ACC producteur<br>(autoconsommation)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-blue">{annual['Taux ACC conso (%)']:.1f} %</div>
            <div class="kpi-label">Taux ACC consommateur<br>(autoproduction)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-green">{annual['ACC (kWh)']:,.0f} kWh</div>
            <div class="kpi-label">Énergie autoconsommée</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-orange">{annual['Surplus réseau (kWh)']:,.0f} kWh</div>
            <div class="kpi-label">Surplus injecté réseau</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-red">{annual['Déficit réseau (kWh)']:,.0f} kWh</div>
            <div class="kpi-label">Déficit soutiré réseau</div>
        </div>""", unsafe_allow_html=True)

# ─── Tab 4 : Synthèse ────────────────────────────────────────────────────────

with tab_synth:
    # ── Bilan annuel ──────────────────────────────────────────────────────────
    st.subheader("Bilan annuel")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        ann_df = pd.DataFrame({
            "Indicateur": list(annual.keys()),
            "Valeur":     [
                f"{v:,.0f} kWh" if "kWh" in k else f"{v:.1f} %"
                for k, v in annual.items()
            ],
        })
        st.dataframe(ann_df, use_container_width=True, hide_index=True)
    with col_b:
        # Camembert des flux énergétiques
        labels = ["ACC (autoconsommé)", "Surplus réseau", "Prod non valorisée (cohérence)"]
        acc_val     = annual["ACC (kWh)"]
        surplus_val = annual["Surplus réseau (kWh)"]
        fig_pie = go.Figure(go.Pie(
            labels=["ACC (autoconsommé)", "Surplus → réseau"],
            values=[acc_val, surplus_val],
            marker_colors=["#16a34a", "#ca8a04"],
            hole=0.45,
            textinfo="percent+label",
            hovertemplate="%{label}: %{value:,.0f} kWh<extra></extra>",
        ))
        fig_pie.update_layout(
            title_text="Destination de la production",
            margin=dict(l=20, r=20, t=50, b=20),
            height=280,
            font=dict(family="Inter, sans-serif", size=12),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Bilan saisonnier ─────────────────────────────────────────────────────
    st.subheader("Bilan saisonnier – Tarifaire Enedis")
    st.caption("Été tarifaire : 1 avril – 31 octobre  ·  Hiver tarifaire : 1 novembre – 31 mars")

    def style_pct(val):
        if isinstance(val, (int, float)):
            if val >= 70:
                return "color: #16a34a; font-weight: 600"
            elif val >= 40:
                return "color: #ca8a04; font-weight: 600"
            else:
                return "color: #dc2626; font-weight: 600"
        return ""

    styled_seasonal = seasonal.style.map(
        style_pct, subset=["Taux ACC prod (%)", "Taux ACC conso (%)"]
    ).format("{:,.0f}", subset=["Prod (kWh)", "Conso (kWh)", "ACC (kWh)", "Surplus réseau (kWh)", "Déficit réseau (kWh)"])
    st.dataframe(styled_seasonal, use_container_width=True)

    st.divider()

    # ── Bilan mensuel graphique ───────────────────────────────────────────────
    st.subheader("Bilan mensuel")
    st.plotly_chart(chart_monthly_bars(monthly), use_container_width=True)

    # ── Tableau mensuel détaillé ──────────────────────────────────────────────
    st.caption("Tableau détaillé mensuel")
    styled_monthly = monthly.style.map(
        style_pct, subset=["Taux ACC prod (%)", "Taux ACC conso (%)"]
    ).format("{:,.0f}", subset=["Prod (kWh)", "Conso (kWh)", "ACC (kWh)", "Surplus réseau (kWh)", "Déficit réseau (kWh)"])
    st.dataframe(styled_monthly, use_container_width=True)

# ─── Tab 5 : Économies ────────────────────────────────────────────────────────

with tab_eco:
    st.subheader("Estimation des économies – Rejoindre l'ACC")

    # Description de l'option tarifaire choisie
    if tariff_option == "Monomial":
        _td = f"Tarif monomial : {tariff_rates['mono']:.4f} €/kWh"
    elif tariff_option == "HP / HC":
        _td = (f"HP : {tariff_rates['hp']:.4f} €/kWh  ·  HC : {tariff_rates['hc']:.4f} €/kWh  ·  "
               f"HC {hc_start.strftime('%H:%M')} → {hc_end.strftime('%H:%M')}")
    else:
        _td = (f"HPH : {tariff_rates['hph']:.4f}  ·  HCH : {tariff_rates['hch']:.4f}  ·  "
               f"HPE : {tariff_rates['hpe']:.4f}  ·  HCE : {tariff_rates['hce']:.4f} €/kWh  ·  "
               f"HC {hc_start.strftime('%H:%M')} → {hc_end.strftime('%H:%M')}")

    with st.expander("ℹ️ Méthode de calcul & hypothèses", expanded=False):
        st.markdown(f"""
**Option tarifaire :** {tariff_option} — {_td}

**Taxes évitées (fixes) :** CEE {cee:.2f} €/MWh · GC {gc:.2f} €/MWh · Accises {accises_mwh:.2f} €/MWh = {(cee+gc+accises_mwh):.2f} €/MWh total taxes

**Prix achat ACC :** {prix_cession:.2f} €/MWh HT

**Formule appliquée à chaque pas de temps (tout en HT) :**

Le calcul porte **uniquement sur les volumes autoconsommés (E_ACC)**. Le reste de la consommation (soutirage réseau) est hors périmètre — il reste facturé par le fournisseur classique.

- `Coût fournisseur classique(t)` = E_ACC(t) × [tarif fourniture(t) + CEE + GC + Accises]
- `Achat ACC(t)` = E_ACC(t) × Prix achat ACC
- `Économie grâce à l'ACC(t)` = Coût fournisseur classique(t) − Achat ACC(t)

*La TVA s'applique identiquement dans les deux cas — la comparaison est faite en HT.*

⚠️ *Estimation indicative. Ne tient pas compte du TURPE ACC spécifique ni des ajustements contractuels.*
        """)

    try:
        tariff_series = build_tariff_series(df_flows.index, tariff_option, hc_start, hc_end, tariff_rates)
        annual_eco, eco_monthly = compute_economics(
            df_flows, consumer_keys, timestep,
            tariff_series, cee, gc, accises_mwh, prix_cession,
        )
    except Exception as e:
        st.error(f"Erreur lors du calcul économique : {e}")
        st.stop()

    # ── KPIs résumé ───────────────────────────────────────────────────────────
    total_e_acc   = sum(v["E_acc (kWh)"]                   for v in annual_eco.values())
    total_sans    = sum(v["Coût sans ACC HT (€)"]          for v in annual_eco.values())
    total_achat   = sum(v["Achat ACC HT (€)"]              for v in annual_eco.values())
    total_nette   = sum(v["Économie grâce à l'ACC HT (€)"] for v in annual_eco.values())
    gain_mwh      = total_nette / total_e_acc * 1000 if total_e_acc > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-blue">{total_e_acc:,.0f} kWh</div>
            <div class="kpi-label">Énergie ACC</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-red">{total_sans:,.0f} €</div>
            <div class="kpi-label">Coût sans ACC HT<br>(fourniture classique)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-orange">{total_achat:,.0f} €</div>
            <div class="kpi-label">Achat ACC HT<br>(payé au producteur)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-green">{total_nette:,.0f} €</div>
            <div class="kpi-label">Économie grâce à l'ACC HT</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-value kpi-green">{gain_mwh:.1f} €/MWh</div>
            <div class="kpi-label">Gain net moyen<br>par MWh ACC</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Décomposition par composante (annuel) ────────────────────────────────
    st.subheader("Décomposition annuelle des économies")

    def _breakdown_table(eco: dict) -> pd.DataFrame:
        sans = eco["Coût sans ACC HT (€)"]
        rows = [
            {"Ligne": "― Sur les volumes ACC uniquement ―",             "Montant HT (€)": None, "% du coût classique": None},
            {"Ligne": "Sans ACC : coût chez fournisseur classique",     "Montant HT (€)": None, "% du coût classique": None},
            {"Ligne": "  dont Fourniture",      "Montant HT (€)": eco["dont Fourniture HT (€)"],  "% du coût classique": eco["dont Fourniture HT (€)"]  / sans * 100 if sans else 0},
            {"Ligne": "  dont CEE",             "Montant HT (€)": eco["dont CEE HT (€)"],         "% du coût classique": eco["dont CEE HT (€)"]         / sans * 100 if sans else 0},
            {"Ligne": "  dont GC",              "Montant HT (€)": eco["dont GC HT (€)"],          "% du coût classique": eco["dont GC HT (€)"]          / sans * 100 if sans else 0},
            {"Ligne": "  dont Accises",         "Montant HT (€)": eco["dont Accises HT (€)"],     "% du coût classique": eco["dont Accises HT (€)"]     / sans * 100 if sans else 0},
            {"Ligne": "  Total fournisseur classique", "Montant HT (€)": sans,                    "% du coût classique": 100.0},
            {"Ligne": "Avec ACC : achat au producteur", "Montant HT (€)": eco["Achat ACC HT (€)"], "% du coût classique": eco["Achat ACC HT (€)"]       / sans * 100 if sans else 0},
            {"Ligne": "→ ÉCONOMIE GRÂCE À L'ACC",      "Montant HT (€)": eco["Économie grâce à l'ACC HT (€)"], "% du coût classique": eco["Économie grâce à l'ACC HT (€)"] / sans * 100 if sans else 0},
        ]
        df = pd.DataFrame(rows).set_index("Ligne")
        return df

    def _fmt_val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:,.0f}"

    def _fmt_pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.1f} %"

    if len(consumer_keys) == 1:
        k = consumer_keys[0]
        st.dataframe(
            _breakdown_table(annual_eco[k]).style.format(
                {"Montant HT (€)": _fmt_val, "% du coût classique": _fmt_pct}
            ),
            use_container_width=True,
        )
    else:
        tabs_conso = st.tabs([k for k in consumer_keys])
        for tab_k, k in zip(tabs_conso, consumer_keys):
            with tab_k:
                st.dataframe(
                    _breakdown_table(annual_eco[k]).style.format(
                        {"Montant HT (€)": _fmt_val, "% du coût classique": _fmt_pct}
                    ),
                    use_container_width=True,
                )

    st.divider()

    # ── Graphique mensuel ────────────────────────────────────────────────────
    st.subheader("Économies mensuelles – décomposition par composante")
    st.plotly_chart(chart_economics_monthly(eco_monthly, consumer_keys), use_container_width=True)

    # ── Tableau mensuel détaillé ──────────────────────────────────────────────
    st.divider()
    _fmt_monthly = {c: "{:,.0f}" for c in eco_monthly[consumer_keys[0]].columns if c != "E_acc (kWh)"}
    _fmt_monthly["E_acc (kWh)"] = "{:,.0f}"

    if len(consumer_keys) == 1:
        st.caption("Tableau mensuel détaillé")
        st.dataframe(eco_monthly[consumer_keys[0]].style.format(_fmt_monthly),
                     use_container_width=True)
    else:
        st.subheader("Tableaux mensuels par consommateur")
        for k in consumer_keys:
            with st.expander(f"Tableau mensuel – {k}"):
                st.dataframe(eco_monthly[k].style.format(_fmt_monthly),
                             use_container_width=True)
