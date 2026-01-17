import io
import re
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
import statsmodels.formula.api as smf


# =========================
# App config
# =========================
st.set_page_config(
    page_title="Hỗ trợ nghiên cứu cho bác sĩ gia đình",
    page_icon="🔬",
    layout="wide",
)

APP_TITLE = "Hỗ trợ nghiên cứu cho bác sĩ gia đình"


# =========================
# FullHD compact UI + FIX title cut/overlay
# =========================
st.markdown(
    """
    <style>
    /* ======= FIX: Safe top area to avoid Streamlit toolbar overlay ======= */
    :root { --app-top-safe: 64px; }  /* nếu vẫn che, tăng 72px */

    /* ======= Layout width + padding ======= */
    .block-container{
        padding-top: calc(var(--app-top-safe) + 0.75rem) !important; /* FIX: không bị che */
        padding-bottom: 0.60rem !important;
        padding-left: 0.90rem !important;
        padding-right: 0.90rem !important;
        max-width: 1600px !important;      /* Full HD */
    }

    /* ======= Typography (gọn hơn ~80%) ======= */
    h1 {
        font-size: 1.70rem !important;
        margin: 0.0rem 0 0.10rem 0 !important;
        line-height: 2.05rem !important;
    }
    h2 { font-size: 1.25rem !important; margin: 0.35rem 0 0.20rem 0 !important; }
    h3 { font-size: 1.05rem !important; margin: 0.35rem 0 0.20rem 0 !important; }
    p, li, label, div { font-size: 0.95rem; }

    /* ======= Reduce gaps ======= */
    div[data-testid="stVerticalBlock"] { gap: 0.28rem; }
    .stMarkdown { margin-bottom: 0.10rem !important; }
    .stCaptionContainer { margin-top: -0.18rem !important; }

    /* ======= Widgets spacing ======= */
    .stSelectbox, .stMultiSelect, .stTextInput, .stFileUploader, .stRadio, .stCheckbox {
        margin-bottom: 0.15rem !important;
    }

    /* ======= Divider ======= */
    hr { margin: 0.40rem 0 !important; }

    /* ======= Buttons (gọn hơn) ======= */
    div.stButton > button{
        width: 100%;
        padding: 8px 10px !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        font-weight: 780 !important;
        border: 1px solid rgba(0,0,0,0.10) !important;
        box-shadow: 0 1px 5px rgba(0,0,0,0.06) !important;
    }

    /* ======= Sidebar compact ======= */
    section[data-testid="stSidebar"] .block-container{
        padding-top: 0.55rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div{
        font-size: 0.90rem !important;
    }

    /* ======= Dataframes ======= */
    .stDataFrame { margin-top: 0.10rem !important; }

    /* ======= Caption dưới stepper gọn ======= */
    [data-testid="stCaptionContainer"] {
        font-size: 0.80rem !important;
        line-height: 1.05rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Helpers: safe name + hashing
# =========================
def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip())[:80] or "file"


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _df_sha256(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(h).hexdigest()


# =========================
# Read files safely
# =========================
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _read_via_tempfile(raw: bytes, suffix: str, reader_fn):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        return reader_fn(tmp_path)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def read_file_safely(uploaded_file) -> Dict[str, pd.DataFrame]:
    name = uploaded_file.name
    ext = Path(name).suffix.lower()
    raw = uploaded_file.getvalue()

    if ext == ".csv":
        return {"data": read_csv_safely(uploaded_file)}

    if ext == ".xlsx":
        xls = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        out: Dict[str, pd.DataFrame] = {}
        for sh in xls.sheet_names:
            out[str(sh)] = pd.read_excel(xls, sheet_name=sh)
        return out

    if ext == ".xls":
        # cần xlrd>=2.0.1
        xls = pd.ExcelFile(io.BytesIO(raw), engine="xlrd")
        out: Dict[str, pd.DataFrame] = {}
        for sh in xls.sheet_names:
            out[str(sh)] = pd.read_excel(xls, sheet_name=sh, engine="xlrd")
        return out

    if ext in [".sav", ".zsav"]:
        # FIX lỗi BytesIO: đọc qua file tạm
        df = _read_via_tempfile(raw, ext, pd.read_spss)
        return {"data": df}

    if ext == ".dta":
        df = _read_via_tempfile(raw, ".dta", pd.read_stata)
        return {"data": df}

    if ext == ".rds":
        try:
            import pyreadr  # type: ignore
        except Exception as e:
            raise RuntimeError("Thiếu pyreadr để đọc .rds. Cài: pip install pyreadr") from e

        def _read_rds(path: str):
            res = pyreadr.read_r(path)
            out: Dict[str, pd.DataFrame] = {}
            for k, v in res.items():
                if isinstance(v, pd.DataFrame):
                    out[str(k) if k else "data"] = v
            return out

        out = _read_via_tempfile(raw, ".rds", _read_rds)
        if not out:
            raise RuntimeError("File .rds không chứa DataFrame (hoặc object không hỗ trợ).")
        return out

    raise RuntimeError(f"Định dạng {ext} chưa được hỗ trợ.")


# =========================
# Type detection
# =========================
def is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        nunique = s.dropna().nunique()
        if nunique <= 10:
            return True
    return False


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def var_kind(s: pd.Series, forced: str = "Tự động") -> str:
    if forced == "Định lượng (numeric)":
        return "num"
    if forced == "Phân loại (categorical)":
        return "cat"
    return "cat" if is_categorical(s) else "num"


# =========================
# Summaries
# =========================
def summarize_variable(df: pd.DataFrame, col: str) -> Dict[str, str]:
    s = df[col]
    miss = int(s.isna().sum())
    n = int(len(s))
    nunique = int(s.dropna().nunique())

    if is_categorical(s):
        vc = s.astype("string").value_counts(dropna=True).head(3)
        top = ", ".join([f"{idx} ({val})" for idx, val in vc.items()]) if len(vc) else "-"
        return {"Tên biến": col, "Đặc tính biến": f"Phân loại | mức={nunique} | thiếu={miss}/{n} | top: {top}"}

    x = coerce_numeric(s)
    x_non = x.dropna()
    if len(x_non) == 0:
        return {"Tên biến": col, "Đặc tính biến": f"Định lượng | thiếu={miss}/{n} | (không đọc được số)"}

    mean = float(x_non.mean())
    sd = float(x_non.std(ddof=1)) if len(x_non) >= 2 else float("nan")
    med = float(x_non.median())
    q1 = float(x_non.quantile(0.25))
    q3 = float(x_non.quantile(0.75))
    return {
        "Tên biến": col,
        "Đặc tính biến": f"Định lượng | thiếu={miss}/{n} | mean={mean:.2f}, SD={sd:.2f} | median={med:.2f} (IQR {q1:.2f}-{q3:.2f})",
    }


def overall_summary(df: pd.DataFrame) -> Dict[str, int]:
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    numeric_cols = sum([pd.api.types.is_numeric_dtype(df[c]) and (not is_categorical(df[c])) for c in df.columns])
    cat_cols = n_cols - numeric_cols
    return {
        "Số dòng": n_rows,
        "Số biến": n_cols,
        "Biến định lượng": int(numeric_cols),
        "Biến phân loại": int(cat_cols),
        "Ô thiếu (NA)": missing_cells,
    }


# =========================
# Assumptions: normality & homogeneity
# =========================
def normality_pvalue(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    try:
        if n <= 5000:
            return float(stats.shapiro(x).pvalue)
        return float(stats.normaltest(x).pvalue)
    except Exception:
        return float("nan")


def variance_homogeneity_pvalue(groups: List[np.ndarray]) -> float:
    clean = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 2]
    if len(clean) < 2:
        return float("nan")
    try:
        return float(stats.levene(*clean, center="median").pvalue)
    except Exception:
        return float("nan")


def assumption_report_num_by_group(df: pd.DataFrame, y_num: str, group_cat: str) -> dict:
    tmp = df[[y_num, group_cat]].dropna().copy()
    tmp[y_num] = pd.to_numeric(tmp[y_num], errors="coerce")
    tmp = tmp.dropna()

    levels = sorted(tmp[group_cat].astype(str).unique().tolist())
    arrays = []
    norm_p = {}
    ns = {}

    for lv in levels:
        a = tmp.loc[tmp[group_cat].astype(str) == lv, y_num].to_numpy()
        a = a[~np.isnan(a)]
        arrays.append(a)
        ns[lv] = int(len(a))
        norm_p[lv] = normality_pvalue(a)

    lev_p = variance_homogeneity_pvalue(arrays)
    return {"levels": levels, "n": ns, "normality_p": norm_p, "levene_p": lev_p, "total_n": int(tmp.shape[0])}


def _norm_ok(report: dict, alpha: float = 0.05) -> bool:
    for lv, n in report["n"].items():
        if n < 3:
            return False
        p = report["normality_p"].get(lv, float("nan"))
        if np.isnan(p) or p < alpha:
            return False
    return True


def _var_ok(report: dict, alpha: float = 0.05) -> bool:
    p = report.get("levene_p", float("nan"))
    return (not np.isnan(p)) and (p >= alpha)


def _assumption_text(rep: dict) -> str:
    norm = ", ".join(
        [
            f"{k}: p={rep['normality_p'][k]:.4f}" if not np.isnan(rep["normality_p"][k]) else f"{k}: p=NA"
            for k in rep["levels"]
        ]
    )
    lev = rep.get("levene_p", float("nan"))
    lev_s = f"{lev:.4f}" if not np.isnan(lev) else "NA"
    return f"Giả định: Shapiro theo nhóm [{norm}]; Levene p={lev_s}."


# =========================
# Single-X: suggest + run
# =========================
def suggest_single_x_test(
    df: pd.DataFrame,
    y: str,
    x: str,
    y_forced: str = "Tự động",
    x_forced: str = "Tự động",
) -> Tuple[str, str, str]:
    yk = var_kind(df[y], y_forced)
    xk = var_kind(df[x], x_forced)

    tmp = df[[y, x]].dropna()
    if tmp.shape[0] < 3:
        return ("Không đủ dữ liệu", "Sau khi loại NA, số dòng quá ít để kiểm định.", "none")

    # cat vs cat
    if yk == "cat" and xk == "cat":
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        if tab.shape == (2, 2) and (tab.values < 5).any():
            return ("Fisher exact (2x2)", "Bảng 2x2 có ô nhỏ → ưu tiên Fisher.", "fisher_2x2")
        return ("Chi-bình phương (Chi-square)", "X và Y đều phân loại → Chi-square.", "chisq")

    # y numeric by group x categorical
    if yk == "num" and xk == "cat":
        rep = assumption_report_num_by_group(df, y_num=y, group_cat=x)
        n_levels = len(rep["levels"])
        norm_ok = _norm_ok(rep)
        var_ok = _var_ok(rep)

        if n_levels == 2:
            if norm_ok and var_ok:
                return ("t-test (Student)", "2 nhóm, đạt chuẩn & phương sai tương đương → Student.", "ttest_student")
            if norm_ok and (not var_ok):
                return ("t-test (Welch)", "2 nhóm, chuẩn nhưng phương sai khác → Welch.", "ttest_welch")
            return ("Mann–Whitney U", "2 nhóm, không đạt giả định chuẩn → Mann–Whitney.", "mwu")

        if norm_ok and var_ok:
            return ("ANOVA một yếu tố", "Nhiều nhóm, đạt chuẩn & đồng nhất phương sai → ANOVA.", "anova")
        return ("Kruskal–Wallis", "Nhiều nhóm, không đạt giả định → Kruskal.", "kruskal")

    # x numeric by group y categorical (swap)
    if yk == "cat" and xk == "num":
        rep = assumption_report_num_by_group(df, y_num=x, group_cat=y)
        n_levels = len(rep["levels"])
        norm_ok = _norm_ok(rep)
        var_ok = _var_ok(rep)

        if n_levels == 2:
            if norm_ok and var_ok:
                return ("t-test (Student)", "2 nhóm, đạt chuẩn & phương sai tương đương → Student.", "ttest_student_swapped")
            if norm_ok and (not var_ok):
                return ("t-test (Welch)", "2 nhóm, chuẩn nhưng phương sai khác → Welch.", "ttest_welch_swapped")
            return ("Mann–Whitney U", "2 nhóm, không đạt giả định chuẩn → Mann–Whitney.", "mwu_swapped")

        if norm_ok and var_ok:
            return ("ANOVA một yếu tố", "Nhiều nhóm, đạt chuẩn & đồng nhất phương sai → ANOVA.", "anova_swapped")
        return ("Kruskal–Wallis", "Nhiều nhóm, không đạt giả định → Kruskal.", "kruskal_swapped")

    # num vs num: correlation
    if yk == "num" and xk == "num":
        tmp2 = df[[y, x]].copy()
        tmp2[y] = coerce_numeric(tmp2[y])
        tmp2[x] = coerce_numeric(tmp2[x])
        tmp2 = tmp2.dropna()
        if tmp2.shape[0] < 3:
            return ("Không đủ dữ liệu", "Không đủ dòng số để tính tương quan.", "none")

        pny = normality_pvalue(tmp2[y].to_numpy())
        pnx = normality_pvalue(tmp2[x].to_numpy())
        if (not np.isnan(pny)) and (not np.isnan(pnx)) and (pny >= 0.05) and (pnx >= 0.05):
            return ("Tương quan Pearson", "X và Y gần chuẩn → Pearson.", "corr_pearson")
        return ("Tương quan Spearman", "X hoặc Y không chuẩn/ordinal → Spearman.", "corr_spearman")

    return ("Không xác định", "Không xác định được phép kiểm phù hợp.", "none")


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if sp == 0:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / sp


def _cramers_v(tab: pd.DataFrame) -> float:
    chi2, p, dof, exp = stats.chi2_contingency(tab.values)
    n = tab.values.sum()
    if n == 0:
        return float("nan")
    r, k = tab.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else float("nan")


def run_single_x_test(df: pd.DataFrame, y: str, x: str, test_kind: str) -> Tuple[pd.DataFrame, str]:
    if test_kind == "chisq":
        tmp = df[[y, x]].dropna()
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        chi2, p, dof, exp = stats.chi2_contingency(tab.values)
        v = _cramers_v(tab)
        out = pd.DataFrame({"Chỉ số": ["Chi2", "df", "p-value", "Cramer's V"], "Giá trị": [chi2, dof, p, v]})
        interp = "Diễn giải: p nhỏ → gợi ý có liên quan. Cramer's V đánh giá độ mạnh liên quan."
        return out, interp

    if test_kind == "fisher_2x2":
        tmp = df[[y, x]].dropna()
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        if tab.shape != (2, 2):
            raise ValueError("Fisher exact chỉ áp dụng bảng 2x2.")
        oddsratio, p = stats.fisher_exact(tab.values)
        out = pd.DataFrame({"Chỉ số": ["Odds ratio", "p-value"], "Giá trị": [oddsratio, p]})
        interp = "Diễn giải: p nhỏ → gợi ý liên quan. OR diễn giải theo nhóm tham chiếu."
        return out, interp

    if test_kind in ("ttest_student", "ttest_welch", "mwu", "anova", "kruskal"):
        tmp = df[[y, x]].dropna().copy()
        tmp[y] = coerce_numeric(tmp[y])
        tmp = tmp.dropna()
        groups = tmp[x].astype(str)
        levels = sorted(groups.unique().tolist())
        arrays = [tmp.loc[groups == lv, y].to_numpy() for lv in levels]
        rep = assumption_report_num_by_group(df, y_num=y, group_cat=x)
        assump = _assumption_text(rep)

        if test_kind in ("ttest_student", "ttest_welch"):
            if len(levels) != 2:
                raise ValueError("t-test cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            equal_var = (test_kind == "ttest_student")
            tstat, p = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
            d = _cohens_d(a, b)
            out = pd.DataFrame({"Chỉ số": ["t", "p-value", "Cohen's d"], "Giá trị": [tstat, p, d]})
            interp = f"{assump}\nDiễn giải: p nhỏ → trung bình khác nhau giữa 2 nhóm. Cohen’s d là effect size."
            return out, interp

        if test_kind == "mwu":
            if len(levels) != 2:
                raise ValueError("Mann–Whitney cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            out = pd.DataFrame({"Chỉ số": ["U", "p-value"], "Giá trị": [u, p]})
            interp = f"{assump}\nDiễn giải: dùng khi không đạt giả định chuẩn."
            return out, interp

        if test_kind == "anova":
            f, p = stats.f_oneway(*arrays)
            out = pd.DataFrame({"Chỉ số": ["F", "p-value"], "Giá trị": [f, p]})
            interp = f"{assump}\nDiễn giải: p nhỏ → có ít nhất 1 nhóm khác trung bình; nên làm post-hoc."
            return out, interp

        if test_kind == "kruskal":
            h, p = stats.kruskal(*arrays)
            out = pd.DataFrame({"Chỉ số": ["H (Kruskal)", "p-value"], "Giá trị": [h, p]})
            interp = f"{assump}\nDiễn giải: dùng khi không đạt giả định; nếu có ý nghĩa nên post-hoc."
            return out, interp

    if test_kind.endswith("_swapped"):
        tmp = df[[y, x]].dropna().copy()
        tmp[x] = coerce_numeric(tmp[x])
        tmp = tmp.dropna()
        groups = tmp[y].astype(str)
        levels = sorted(groups.unique().tolist())
        arrays = [tmp.loc[groups == lv, x].to_numpy() for lv in levels]
        rep = assumption_report_num_by_group(df, y_num=x, group_cat=y)
        assump = _assumption_text(rep)
        base = test_kind.replace("_swapped", "")

        if base in ("ttest_student", "ttest_welch"):
            if len(levels) != 2:
                raise ValueError("t-test cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            equal_var = (base == "ttest_student")
            tstat, p = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
            d = _cohens_d(a, b)
            out = pd.DataFrame({"Chỉ số": ["t", "p-value", "Cohen's d"], "Giá trị": [tstat, p, d]})
            interp = f"{assump}\nDiễn giải: p nhỏ → trung bình khác nhau giữa 2 nhóm (theo Y)."
            return out, interp

        if base == "mwu":
            if len(levels) != 2:
                raise ValueError("Mann–Whitney cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            out = pd.DataFrame({"Chỉ số": ["U", "p-value"], "Giá trị": [u, p]})
            interp = f"{assump}\nDiễn giải: dùng khi không đạt giả định chuẩn."
            return out, interp

        if base == "anova":
            f, p = stats.f_oneway(*arrays)
            out = pd.DataFrame({"Chỉ số": ["F", "p-value"], "Giá trị": [f, p]})
            interp = f"{assump}\nDiễn giải: p nhỏ → có ít nhất 1 nhóm khác trung bình; nên làm post-hoc."
            return out, interp

        if base == "kruskal":
            h, p = stats.kruskal(*arrays)
            out = pd.DataFrame({"Chỉ số": ["H (Kruskal)", "p-value"], "Giá trị": [h, p]})
            interp = f"{assump}\nDiễn giải: dùng khi không đạt giả định; nếu có ý nghĩa nên post-hoc."
            return out, interp

    if test_kind == "corr_pearson":
        tmp = df[[y, x]].copy()
        tmp[y] = coerce_numeric(tmp[y])
        tmp[x] = coerce_numeric(tmp[x])
        tmp = tmp.dropna()
        r, p = stats.pearsonr(tmp[x].to_numpy(), tmp[y].to_numpy())
        out = pd.DataFrame({"Chỉ số": ["Pearson r", "p-value", "n"], "Giá trị": [r, p, tmp.shape[0]]})
        interp = "Diễn giải: r gần 0 → yếu; gần ±1 → mạnh. p nhỏ → liên quan tuyến tính có ý nghĩa."
        return out, interp

    if test_kind == "corr_spearman":
        tmp = df[[y, x]].copy()
        tmp[y] = coerce_numeric(tmp[y])
        tmp[x] = coerce_numeric(tmp[x])
        tmp = tmp.dropna()
        rho, p = stats.spearmanr(tmp[x].to_numpy(), tmp[y].to_numpy())
        out = pd.DataFrame({"Chỉ số": ["Spearman rho", "p-value", "n"], "Giá trị": [rho, p, tmp.shape[0]]})
        interp = "Diễn giải: Spearman đánh giá liên quan đơn điệu, phù hợp khi dữ liệu không chuẩn/ordinal."
        return out, interp

    raise ValueError("Không có kiểm định phù hợp (test_kind=none).")


# =========================
# Model: suggest + build + run
# =========================
def suggest_model(df: pd.DataFrame, y: str, xs: List[str]) -> Tuple[str, str]:
    y_s = df[y]
    if is_categorical(y_s):
        n_levels = int(y_s.dropna().nunique())
        if n_levels <= 1:
            return ("Không đủ dữ liệu", "Y chỉ có 0–1 mức sau khi loại thiếu. Hãy kiểm tra dữ liệu.")
        if n_levels == 2:
            return ("Hồi quy Logistic nhị phân (Binary Logistic)", "Y phân loại 2 mức → logistic nhị phân để ước lượng OR.")
        return ("Hồi quy Logistic đa danh (Multinomial Logistic)", f"Y >2 mức (mức={n_levels}) → logistic đa danh.")
    return ("Hồi quy tuyến tính (OLS)", "Y định lượng → hồi quy tuyến tính (OLS).")


def build_formula(
    df: pd.DataFrame,
    y: str,
    xs: List[str],
    y_binary_event: Optional[str] = None,
) -> Tuple[str, pd.DataFrame, str]:
    tmp = df[[y] + xs].copy().dropna()

    if is_categorical(tmp[y]):
        n_levels = int(tmp[y].nunique())

        if n_levels == 2:
            y_cat = tmp[y].astype("category")
            cats = list(y_cat.cat.categories)
            event = y_binary_event if (y_binary_event in cats) else cats[1]
            tmp["_y01_"] = (tmp[y] == event).astype(int)

            terms = []
            for x in xs:
                terms.append(f"C(Q('{x}'))" if is_categorical(tmp[x]) else f"Q('{x}')")

            formula = "_y01_ ~ " + " + ".join(terms)
            return formula, tmp, f"logit||Logistic nhị phân: sự kiện (Y=1)='{event}'"

        tmp["_ycat_"] = tmp[y].astype("category")
        tmp["_ycode_"] = tmp["_ycat_"].cat.codes

        terms = []
        for x in xs:
            terms.append(f"C(Q('{x}'))" if is_categorical(tmp[x]) else f"Q('{x}')")

        formula = "_ycode_ ~ " + " + ".join(terms)
        return formula, tmp, "mnlogit||Multinomial: hệ số theo nhóm tham chiếu"

    tmp[y] = coerce_numeric(tmp[y])
    tmp = tmp.dropna()

    terms = []
    for x in xs:
        terms.append(f"C(Q('{x}'))" if is_categorical(tmp[x]) else f"Q('{x}')")

    formula = f"Q('{y}') ~ " + " + ".join(terms)
    return formula, tmp, "ols||OLS"


def run_model(formula: str, data_used: pd.DataFrame, model_kind: str):
    kind, note = model_kind.split("||", 1)

    if kind == "ols":
        return smf.ols(formula=formula, data=data_used).fit(), note

    if kind == "logit":
        return smf.logit(formula=formula, data=data_used).fit(disp=0), note

    if kind == "mnlogit":
        return smf.mnlogit(formula=formula, data=data_used).fit(disp=0), note

    raise ValueError("Unknown model kind")


def ols_table(fit) -> pd.DataFrame:
    conf = fit.conf_int()
    out = pd.DataFrame(
        {"Hệ số": fit.params, "CI 2.5%": conf[0], "CI 97.5%": conf[1], "p-value": fit.pvalues}
    )
    out.index.name = "Biến"
    return out.sort_values("p-value")


def logit_or_table(fit) -> pd.DataFrame:
    conf = fit.conf_int()
    out = pd.DataFrame(
        {
            "OR": np.exp(fit.params),
            "CI 2.5%": np.exp(conf[0]),
            "CI 97.5%": np.exp(conf[1]),
            "p-value": fit.pvalues,
        }
    )
    out.index.name = "Biến"
    return out.sort_values("p-value")


# =========================
# OLS equation + detailed interpretation
# =========================
def format_ols_equation(fit, y_name: str) -> str:
    params = fit.params.to_dict()
    parts = []
    b0 = float(params.get("Intercept", 0.0))
    parts.append(f"{b0:.4f}")

    for term, b in params.items():
        if term == "Intercept":
            continue

        m_num = re.match(r"Q\('(.+)'\)", term)
        if m_num:
            var = m_num.group(1)
            parts.append(f"{float(b):+.4f}*{var}")
            continue

        m_cat = re.match(r"C\(Q\('(.+)'\)\)\[T\.(.+)\]", term)
        if m_cat:
            var = m_cat.group(1)
            lv = m_cat.group(2)
            parts.append(f"{float(b):+.4f}*I({var}={lv})")
            continue

        parts.append(f"{float(b):+.4f}*({term})")

    return f"**Phương trình (OLS):**  Ŷ({y_name}) = " + " ".join(parts)


def explain_ols_effects(fit, y_name: str, alpha: float = 0.05) -> List[str]:
    conf = fit.conf_int()
    lines: List[str] = []
    for term in fit.params.index:
        if term == "Intercept":
            continue
        b = float(fit.params[term])
        p = float(fit.pvalues[term])
        lo = float(conf.loc[term, 0])
        hi = float(conf.loc[term, 1])
        sig = "có ý nghĩa thống kê" if p < alpha else "chưa đủ ý nghĩa thống kê"

        m_num = re.match(r"Q\('(.+)'\)", term)
        if m_num:
            var = m_num.group(1)
            direction = "tăng" if b > 0 else "giảm"
            lines.append(
                f"- **{var}**: tăng 1 đơn vị → **{y_name} {direction} {abs(b):.4f} đơn vị** (đã hiệu chỉnh). "
                f"p={p:.4g}, CI95%=[{lo:.4f}; {hi:.4f}] → {sig}."
            )
            continue

        m_cat = re.match(r"C\(Q\('(.+)'\)\)\[T\.(.+)\]", term)
        if m_cat:
            var = m_cat.group(1)
            lv = m_cat.group(2)
            direction = "cao hơn" if b > 0 else "thấp hơn"
            lines.append(
                f"- **{var}={lv}** (so với nhóm tham chiếu): **{y_name} {direction} {abs(b):.4f} đơn vị** (đã hiệu chỉnh). "
                f"p={p:.4g}, CI95%=[{lo:.4f}; {hi:.4f}] → {sig}."
            )
            continue

        lines.append(
            f"- **{term}**: coef={b:.4f}, p={p:.4g}, CI95%=[{lo:.4f}; {hi:.4f}] → {sig}."
        )
    return lines or ["- Không có biến giải thích (chỉ intercept)."]


# =========================
# Session state
# =========================
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
if "active_name" not in st.session_state:
    st.session_state["active_name"] = None

if "pending_tables" not in st.session_state:
    st.session_state["pending_tables"] = None
if "pending_fname" not in st.session_state:
    st.session_state["pending_fname"] = None
if "pending_file_hash" not in st.session_state:
    st.session_state["pending_file_hash"] = None

if "hash_to_key" not in st.session_state:
    st.session_state["hash_to_key"] = {}
if "key_to_hashes" not in st.session_state:
    st.session_state["key_to_hashes"] = {}

if "last_upload_hash" not in st.session_state:
    st.session_state["last_upload_hash"] = None

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_run_meta" not in st.session_state:
    st.session_state["last_run_meta"] = None

if "active_step" not in st.session_state:
    st.session_state["active_step"] = 1


def _register_dataset(key: str, df: pd.DataFrame, hashes: List[str]):
    st.session_state["datasets"][key] = df
    st.session_state["active_name"] = key
    st.session_state["key_to_hashes"].setdefault(key, set())
    for h in hashes:
        st.session_state["hash_to_key"][h] = key
        st.session_state["key_to_hashes"][key].add(h)


def _delete_dataset(key: str):
    st.session_state["datasets"].pop(key, None)
    hashes = st.session_state["key_to_hashes"].pop(key, set())
    for h in list(hashes):
        if st.session_state["hash_to_key"].get(h) == key:
            st.session_state["hash_to_key"].pop(h, None)


# =========================
# Header (compact, safe top)
# =========================
st.markdown(
    f"""
    <div style="padding:0.10rem 0 0.10rem 0; margin-top:0.20rem;">
      <h1 style="margin:0;">{APP_TITLE}</h1>
      <div style="color:#6b7280; font-size:0.88rem; margin-top:0.08rem;">
        Upload dữ liệu → chọn biến → kiểm định (1 X) hoặc mô hình (nhiều X) → kết quả + giải thích
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## ⬆️ Upload")
    up = st.file_uploader(
        "Tải lên dữ liệu (CSV/XLSX/XLS/SAV/ZsAV/DTA/RDS)",
        type=["csv", "xlsx", "xls", "sav", "zsav", "dta", "rds"],
        accept_multiple_files=False,
    )

    if up is not None:
        try:
            raw = up.getvalue()
            file_hash = _file_sha256(raw)

            # chống xử lý lại cùng 1 upload
            if st.session_state["last_upload_hash"] != file_hash:
                st.session_state["last_upload_hash"] = file_hash

                # nếu file đã từng import → chọn lại dataset cũ (tránh duplicate)
                if file_hash in st.session_state["hash_to_key"]:
                    existed_key = st.session_state["hash_to_key"][file_hash]
                    st.session_state["active_name"] = existed_key
                    st.info(f"Đã có trước đó → {existed_key}")
                else:
                    tables = read_file_safely(up)

                    # file nhiều sheet/object
                    if len(tables) > 1:
                        st.session_state["pending_tables"] = tables
                        st.session_state["pending_fname"] = up.name
                        st.session_state["pending_file_hash"] = file_hash
                        st.info("File có nhiều bảng → chọn 1 bảng để nhập.")
                    else:
                        df_new = list(tables.values())[0]
                        base = _safe_name(Path(up.name).stem)
                        key = base
                        i = 2
                        while key in st.session_state["datasets"]:
                            key = f"{base}_{i}"
                            i += 1

                        df_hash = _df_sha256(df_new)
                        _register_dataset(key, df_new, hashes=[file_hash, df_hash])
                        st.session_state["active_step"] = 1
                        st.success(f"Đã tải: {key}")

        except Exception as e:
            st.error(f"Không đọc được file: {e}")

    # chọn sheet/object khi file có nhiều bảng
    if st.session_state["pending_tables"] is not None:
        st.markdown("### Chọn sheet/object")
        tables = st.session_state["pending_tables"]
        fname = st.session_state["pending_fname"] or "file"
        pending_file_hash = st.session_state["pending_file_hash"]

        chosen_table = st.selectbox("Sheet/Object", options=list(tables.keys()))
        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            if st.button("Nhập", use_container_width=True):
                df_new = tables[chosen_table]
                table_hash = _df_sha256(df_new)

                if table_hash in st.session_state["hash_to_key"]:
                    existed_key = st.session_state["hash_to_key"][table_hash]
                    st.session_state["active_name"] = existed_key
                    st.info(f"Đã nhập trước đó → {existed_key}")
                else:
                    base = _safe_name(Path(fname).stem)
                    sh = _safe_name(chosen_table)
                    key_base = f"{base}__{sh}"
                    key = key_base
                    i = 2
                    while key in st.session_state["datasets"]:
                        key = f"{key_base}_{i}"
                        i += 1

                    hashes = [table_hash]
                    if pending_file_hash:
                        hashes.append(pending_file_hash)

                    _register_dataset(key, df_new, hashes=hashes)
                    st.session_state["active_step"] = 1
                    st.success(f"Đã nhập: {key}")

                st.session_state["pending_tables"] = None
                st.session_state["pending_fname"] = None
                st.session_state["pending_file_hash"] = None
                st.rerun()

        with c2:
            if st.button("Huỷ", use_container_width=True):
                st.session_state["pending_tables"] = None
                st.session_state["pending_fname"] = None
                st.session_state["pending_file_hash"] = None
                st.rerun()

    st.markdown("---")
    st.markdown("## 📁 Dataset")

    names_all = list(st.session_state["datasets"].keys())
    if not names_all:
        st.info("Chưa có dữ liệu.")
        st.stop()

    ds_q = st.text_input("Tìm dataset", value="", placeholder="gõ tên dataset...")
    if ds_q.strip():
        names = [n for n in names_all if ds_q.lower() in n.lower()] or names_all
    else:
        names = names_all

    active = st.session_state["active_name"] or names_all[0]
    if active not in names_all:
        active = names_all[0]
        st.session_state["active_name"] = active

    chosen = st.selectbox("Chọn dataset", options=names, index=names.index(active) if active in names else 0)
    st.session_state["active_name"] = chosen

    with st.expander("✏️ Đổi tên dataset"):
        new_name = st.text_input("Tên mới", value=chosen)
        if st.button("Lưu tên", use_container_width=True):
            new_name = _safe_name(new_name)
            if (new_name != chosen) and (new_name in st.session_state["datasets"]):
                st.error("Tên đã tồn tại.")
            else:
                df_tmp = st.session_state["datasets"].pop(chosen)
                st.session_state["datasets"][new_name] = df_tmp

                hashes = st.session_state["key_to_hashes"].pop(chosen, set())
                st.session_state["key_to_hashes"][new_name] = hashes
                for h in list(hashes):
                    if st.session_state["hash_to_key"].get(h) == chosen:
                        st.session_state["hash_to_key"][h] = new_name

                st.session_state["active_name"] = new_name
                st.success("Đã đổi tên.")
                st.rerun()

    df_active = st.session_state["datasets"][st.session_state["active_name"]]
    summ_side = overall_summary(df_active)
    st.caption(f"rows={summ_side['Số dòng']} | biến={summ_side['Số biến']} | thiếu={summ_side['Ô thiếu (NA)']}")

    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        if st.button("Xoá", use_container_width=True):
            _delete_dataset(chosen)
            remaining = list(st.session_state["datasets"].keys())
            st.session_state["active_name"] = remaining[0] if remaining else None
            st.session_state["last_result"] = None
            st.session_state["last_run_meta"] = None
            st.session_state["active_step"] = 1
            st.rerun()

    with c2:
        if st.button("Xoá hết", use_container_width=True):
            st.session_state["datasets"] = {}
            st.session_state["active_name"] = None
            st.session_state["pending_tables"] = None
            st.session_state["pending_fname"] = None
            st.session_state["pending_file_hash"] = None
            st.session_state["hash_to_key"] = {}
            st.session_state["key_to_hashes"] = {}
            st.session_state["last_upload_hash"] = None
            st.session_state["last_result"] = None
            st.session_state["last_run_meta"] = None
            st.session_state["active_step"] = 1
            st.rerun()


# =========================
# Main data
# =========================
df = st.session_state["datasets"][st.session_state["active_name"]]
cols = df.columns.tolist()


# =========================
# Stepper
# =========================
st.markdown("## 🧭 Các bước")
b1, b2, b3 = st.columns(3, gap="small")

with b1:
    t = "primary" if st.session_state["active_step"] == 1 else "secondary"
    if st.button("1) 📄 Dữ liệu", type=t, use_container_width=True):
        st.session_state["active_step"] = 1
        st.rerun()
    st.caption("Tổng quan • xem bảng • danh sách biến")

with b2:
    t = "primary" if st.session_state["active_step"] == 2 else "secondary"
    if st.button("2) 🎯 Chọn biến", type=t, use_container_width=True):
        st.session_state["active_step"] = 2
        st.rerun()
    st.caption("Chọn Y/X • gợi ý • bấm Run")

with b3:
    t = "primary" if st.session_state["active_step"] == 3 else "secondary"
    if st.button("3) 📌 Kết quả", type=t, use_container_width=True):
        st.session_state["active_step"] = 3
        st.rerun()
    st.caption("Bảng • biểu đồ • diễn giải")

st.divider()


# =========================
# Compute & store results
# =========================
def _compute_and_store(y: str, xs: List[str], y_force: str, x_force: str, y_event: Optional[str]):
    # 1 X -> test
    if len(xs) == 1:
        suggestion, explanation, test_kind = suggest_single_x_test(df, y, xs[0], y_forced=y_force, x_forced=x_force)
        result_df, interp = run_single_x_test(df, y, xs[0], test_kind=test_kind)

        st.session_state["last_run_meta"] = {
            "dataset": st.session_state["active_name"],
            "mode": "test",
            "y": y,
            "xs": xs,
            "suggestion": suggestion,
            "explanation": explanation,
            "test_kind": test_kind,
            "y_force": y_force,
            "x_force": x_force,
        }
        st.session_state["last_result"] = {"table": result_df, "interp": interp}
        return

    # many X -> model
    tmp_for_suggest = df.copy()
    if y_force == "Định lượng (numeric)":
        tmp_for_suggest[y] = coerce_numeric(tmp_for_suggest[y])
    elif y_force == "Phân loại (categorical)":
        tmp_for_suggest[y] = tmp_for_suggest[y].astype("string")

    suggestion, explanation = suggest_model(tmp_for_suggest, y, xs)

    df_model = df.copy()
    if y_force == "Định lượng (numeric)":
        df_model[y] = coerce_numeric(df_model[y])
    elif y_force == "Phân loại (categorical)":
        df_model[y] = df_model[y].astype("string")

    formula, data_used, model_kind = build_formula(df_model, y, xs, y_binary_event=y_event)
    fit, note = run_model(formula, data_used, model_kind)
    kind = model_kind.split("||", 1)[0]

    table = None
    if kind == "ols":
        table = ols_table(fit)
    elif kind == "logit":
        table = logit_or_table(fit)

    st.session_state["last_run_meta"] = {
        "dataset": st.session_state["active_name"],
        "mode": "model",
        "y": y,
        "xs": xs,
        "suggestion": suggestion,
        "explanation": explanation,
        "formula": formula,
        "n_used": int(data_used.shape[0]),
        "model_kind": model_kind,
        "note": note,
        "y_force": y_force,
        "x_force": x_force,
        "y_event": y_event,
    }
    st.session_state["last_result"] = {"fit": fit, "kind": kind, "table": table, "data_used": data_used}


# =========================
# STEP 1: Data
# =========================
if st.session_state["active_step"] == 1:
    st.subheader("📄 Dữ liệu")

    summ = overall_summary(df)
    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    m1.metric("Dòng", summ["Số dòng"])
    m2.metric("Biến", summ["Số biến"])
    m3.metric("Định lượng", summ["Biến định lượng"])
    m4.metric("Phân loại", summ["Biến phân loại"])
    m5.metric("NA", summ["Ô thiếu (NA)"])

    cL, cR = st.columns([1.2, 1.0], gap="small")
    with cL:
        st.markdown("### 👀 Xem nhanh")
        st.dataframe(df.head(25), use_container_width=True, height=240)

    with cR:
        st.markdown("### 🧾 Danh sách biến")
        q = st.text_input("Tìm biến", value="", placeholder="vd: age, weight...")
        filter_opt = st.selectbox("Lọc", ["Tất cả", "Chỉ định lượng", "Chỉ phân loại"], index=0)

        var_rows = [summarize_variable(df, c) for c in cols]
        var_df = pd.DataFrame(var_rows)

        if q.strip():
            var_df = var_df[var_df["Tên biến"].str.contains(q.strip(), case=False, na=False)].copy()

        if filter_opt == "Chỉ định lượng":
            var_df = var_df[var_df["Đặc tính biến"].str.contains("Định lượng", na=False)]
        elif filter_opt == "Chỉ phân loại":
            var_df = var_df[var_df["Đặc tính biến"].str.contains("Phân loại", na=False)]

        st.dataframe(var_df, use_container_width=True, height=240)

    st.info("👉 Sang **2) Chọn biến** để chọn Y/X và bấm Run.")


# =========================
# STEP 2: Choose variables
# =========================
elif st.session_state["active_step"] == 2:
    st.subheader("🎯 Chọn biến")

    left, right = st.columns([2.0, 1.0], gap="small")

    with left:
        vq = st.text_input("Tìm biến (tuỳ chọn)", value="", placeholder="gõ để lọc danh sách...")
        cols_show = [c for c in cols if vq.lower() in c.lower()] if vq.strip() else cols
        if not cols_show:
            cols_show = cols

        y = st.selectbox("Biến phụ thuộc (Y)", options=cols_show, index=0)
        xs = st.multiselect("Biến độc lập (X)", options=[c for c in cols_show if c != y])

        force_opts = ["Tự động", "Định lượng (numeric)", "Phân loại (categorical)"]
        y_force = st.selectbox("Kiểu Y", options=force_opts, index=0)

        x_force = "Tự động"
        if len(xs) == 1:
            x_force = st.selectbox("Kiểu X (chỉ khi 1 X)", options=force_opts, index=0)

        y_event = None
        if var_kind(df[y], y_force) == "cat":
            levels = sorted(df[y].dropna().astype(str).unique().tolist())
            if len(levels) == 2:
                y_event = st.selectbox("Sự kiện (Y=1) cho logistic", options=levels, index=1)

        st.markdown("### ✅ Gợi ý")
        if len(xs) == 0:
            st.info("Chọn ít nhất 1 biến X.")
        else:
            if len(xs) == 1:
                suggestion, explanation, _ = suggest_single_x_test(df, y, xs[0], y_forced=y_force, x_forced=x_force)
                mode_label = "Kiểm định"
            else:
                tmp_for_suggest = df.copy()
                if y_force == "Định lượng (numeric)":
                    tmp_for_suggest[y] = coerce_numeric(tmp_for_suggest[y])
                elif y_force == "Phân loại (categorical)":
                    tmp_for_suggest[y] = tmp_for_suggest[y].astype("string")
                suggestion, explanation = suggest_model(tmp_for_suggest, y, xs)
                mode_label = "Mô hình"

            st.write(f"**Chế độ:** {mode_label}")
            st.write(f"**Gợi ý:** {suggestion}")
            with st.expander("Giải thích"):
                st.write(explanation)

    with right:
        st.markdown("### 📌 Tóm tắt")
        st.write(f"**Dataset:** {st.session_state['active_name']}")
        st.write(f"**Biến phụ thuộc (Y):** {y}")
        st.write(f"**Biến độc lập (X):** {', '.join(xs) if xs else '-'}")

        st.markdown("---")
        if st.button("▶️ Run", type="primary", use_container_width=True, disabled=(len(xs) == 0)):
            try:
                _compute_and_store(y=y, xs=xs, y_force=y_force, x_force=x_force, y_event=y_event)
                st.session_state["active_step"] = 3
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi chạy: {e}")


# =========================
# STEP 3: Results
# =========================
else:
    st.subheader("📌 Kết quả")

    meta = st.session_state.get("last_run_meta")
    res = st.session_state.get("last_result")

    if not meta or not res:
        st.info("Chưa có kết quả. Vào **2) Chọn biến** → chọn Y/X → bấm **Run**.")
    else:
        y_name = meta.get("y", "-")
        x_list = meta.get("xs", [])
        x_text = ", ".join(x_list) if x_list else "-"

        st.markdown(
            f"""
            <div style="border:1px solid rgba(0,0,0,0.08); border-radius:12px; padding:10px;">
              <div style="display:flex; gap:12px; flex-wrap:wrap;">
                <div style="min-width:200px;">
                  <div style="color:#6b7280; font-size:12px;">Dataset</div>
                  <div style="font-size:15px; font-weight:800;">{meta.get('dataset','-')}</div>
                </div>
                <div style="min-width:220px;">
                  <div style="color:#6b7280; font-size:12px;">Biến phụ thuộc (Y)</div>
                  <div style="font-size:15px; font-weight:800;">{y_name}</div>
                </div>
                <div style="min-width:320px; flex:1;">
                  <div style="color:#6b7280; font-size:12px;">Biến độc lập (X)</div>
                  <div style="font-size:15px; font-weight:800;">{x_text}</div>
                </div>
              </div>
              <div style="margin-top:6px; color:#6b7280; font-size:12px;">Gợi ý</div>
              <div style="font-size:15px; font-weight:800;">{meta.get('suggestion','-')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        left, right = st.columns([1.45, 1.0], gap="small")

        with left:
            if meta["mode"] == "test":
                st.markdown("### 📊 Kết quả kiểm định")
                st.dataframe(res["table"], use_container_width=True, height=260)

                st.markdown("### 🔎 Diễn giải")
                st.write(res["interp"])

            else:
                kind = res["kind"]
                fit = res["fit"]
                table = res["table"]
                st.caption(meta.get("note", ""))

                if kind == "ols" and table is not None:
                    st.markdown("### 📊 Bảng kết quả mô hình (OLS)")
                    st.dataframe(table, use_container_width=True, height=270)

                    st.markdown("### 🧮 Phương trình hồi quy")
                    st.write(format_ols_equation(fit, y_name))

                    st.markdown("### 🔎 Diễn giải chi tiết (mỗi biến)")
                    st.write("\n".join(explain_ols_effects(fit, y_name, alpha=0.05)))

                elif kind == "logit" and table is not None:
                    st.markdown("### 📊 Bảng kết quả logistic (OR)")
                    st.dataframe(table, use_container_width=True, height=270)
                    st.write(
                        "🔎 **Gợi ý diễn giải:**\n"
                        "- OR > 1: tăng odds xảy ra sự kiện (Y=1)\n"
                        "- OR < 1: giảm odds\n"
                        "- p-value < 0.05 và CI 95% không chứa 1: thường có ý nghĩa"
                    )

                else:
                    st.markdown("### 📄 MNLogit Summary")
                    st.write(fit.summary())
                    st.info("Multinomial: nếu bạn muốn bảng RRR = exp(coef) theo từng nhóm, mình có thể bổ sung tiếp.")

        with right:
            st.markdown("### 📈 Biểu đồ minh hoạ")
            try:
                if meta["mode"] == "test":
                    y = meta["y"]
                    x1 = meta["xs"][0]
                    y_force = meta.get("y_force", "Tự động")
                    x_force = meta.get("x_force", "Tự động")

                    yk = var_kind(df[y], y_force)
                    xk = var_kind(df[x1], x_force)
                    tmp = df[[y, x1]].dropna().copy()

                    if yk == "num" and xk == "cat":
                        tmp[y] = coerce_numeric(tmp[y])
                        tmp = tmp.dropna()
                        fig = px.box(tmp, x=x1, y=y, points="all", title=f"{y} theo nhóm {x1}", height=320)
                        st.plotly_chart(fig, use_container_width=True)

                    elif yk == "cat" and xk == "num":
                        tmp[x1] = coerce_numeric(tmp[x1])
                        tmp = tmp.dropna()
                        fig = px.box(tmp, x=y, y=x1, points="all", title=f"{x1} theo nhóm {y}", height=320)
                        st.plotly_chart(fig, use_container_width=True)

                    elif yk == "cat" and xk == "cat":
                        tab = pd.crosstab(tmp[y].astype(str), tmp[x1].astype(str))
                        tab2 = tab.div(tab.sum(axis=1), axis=0).reset_index().melt(
                            id_vars=[y], var_name=x1, value_name="Tỷ lệ"
                        )
                        fig = px.bar(tab2, x=y, y="Tỷ lệ", color=x1, barmode="stack", title="Tỷ lệ theo nhóm", height=320)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        tmp[y] = coerce_numeric(tmp[y])
                        tmp[x1] = coerce_numeric(tmp[x1])
                        tmp = tmp.dropna()
                        fig = px.scatter(tmp, x=x1, y=y, trendline="ols", title=f"{y} ~ {x1}", height=320)
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    kind = res["kind"]
                    data_used = res["data_used"]
                    y = meta["y"]
                    xs = meta["xs"]

                    if kind == "ols":
                        x1 = xs[0]
                        if (not is_categorical(data_used[x1])) and (not is_categorical(data_used[y])):
                            fig = px.scatter(data_used, x=x1, y=y, trendline="ols", title=f"{y} ~ {x1}", height=320)
                        else:
                            fig = (
                                px.box(data_used, x=x1, y=y, points="all", title=f"{y} theo nhóm {x1}", height=320)
                                if is_categorical(data_used[x1])
                                else px.scatter(data_used, x=x1, y=y, title=f"{y} theo {x1}", height=320)
                            )
                        st.plotly_chart(fig, use_container_width=True)

                    elif kind == "logit":
                        p = res["fit"].predict()
                        fig = px.histogram(p, nbins=22, title="Xác suất dự đoán (p)", height=320)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.info("Multinomial: biểu đồ sẽ được bổ sung theo nhu cầu.")
            except Exception as e:
                st.warning(f"Không vẽ được biểu đồ: {e}")

    st.divider()
    st.caption(
        "⚠️ Lưu ý: Công cụ hỗ trợ gợi ý và chạy kiểm định/mô hình cơ bản. "
        "Người dùng cần kiểm tra giả định, thiết kế nghiên cứu và mã hoá biến để diễn giải đúng."
    )
