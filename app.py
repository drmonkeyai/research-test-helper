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
# UI CSS: big step buttons
# =========================
st.markdown(
    """
    <style>
    /* Làm nút to hơn */
    div.stButton > button {
        width: 100%;
        padding: 16px 14px !important;
        border-radius: 14px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border: 1px solid rgba(0,0,0,0.12) !important;
        box-shadow: 0 1px 8px rgba(0,0,0,0.06) !important;
    }
    /* Caption nhỏ dưới nút */
    .step-caption {
        color: #6b7280;
        font-size: 13px;
        margin-top: -6px;
        margin-bottom: 4px;
        line-height: 1.25rem;
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
# Helpers: read files
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
    """
    Nhiều hàm (read_spss/read_stata/pyreadr) cần path -> dùng file tạm.
    """
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
    """
    Return dict {table_name: df}

    Supported:
      - .csv
      - .xlsx (openpyxl)
      - .xls  (xlrd)
      - .sav/.zsav (SPSS) via pandas.read_spss(path)
      - .dta (STATA) via pandas.read_stata(path)
      - .rds (R) via pyreadr (optional)
    """
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
        xls = pd.ExcelFile(io.BytesIO(raw), engine="xlrd")
        out: Dict[str, pd.DataFrame] = {}
        for sh in xls.sheet_names:
            out[str(sh)] = pd.read_excel(xls, sheet_name=sh, engine="xlrd")
        return out

    if ext in [".sav", ".zsav"]:
        df = _read_via_tempfile(raw, ext, pd.read_spss)
        return {"data": df}

    if ext == ".dta":
        df = _read_via_tempfile(raw, ".dta", pd.read_stata)
        return {"data": df}

    if ext == ".rds":
        try:
            import pyreadr  # type: ignore
        except Exception as e:
            raise RuntimeError("Thiếu thư viện pyreadr để đọc .rds. Hãy cài: pip install pyreadr") from e

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
# Helpers: type detection
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
        return {
            "Tên biến": col,
            "Đặc tính biến": f"Phân loại | mức={nunique} | thiếu={miss}/{n} | top: {top}",
        }

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
    return {
        "levels": levels,
        "n": ns,
        "normality_p": norm_p,
        "levene_p": lev_p,
        "total_n": int(tmp.shape[0]),
    }


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

    # cat-cat
    if yk == "cat" and xk == "cat":
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        if tab.shape == (2, 2) and (tab.values < 5).any():
            return ("Fisher exact (2x2)", "Bảng 2x2 và có ô nhỏ → ưu tiên Fisher exact.", "fisher_2x2")
        return ("Chi-bình phương (Chi-square)", "X và Y đều phân loại → kiểm định độc lập bằng Chi-square.", "chisq")

    # y num, x cat
    if yk == "num" and xk == "cat":
        rep = assumption_report_num_by_group(df, y_num=y, group_cat=x)
        n_levels = len(rep["levels"])
        norm_ok = _norm_ok(rep)
        var_ok = _var_ok(rep)

        if n_levels == 2:
            if norm_ok and var_ok:
                return ("t-test (Student)", "2 nhóm, đạt chuẩn & phương sai tương đương → Student t-test.", "ttest_student")
            if norm_ok and (not var_ok):
                return ("t-test (Welch)", "2 nhóm, chuẩn nhưng phương sai khác → Welch t-test.", "ttest_welch")
            return ("Mann–Whitney U", "2 nhóm nhưng không đạt giả định chuẩn → Mann–Whitney.", "mwu")

        if norm_ok and var_ok:
            return ("ANOVA một yếu tố", "Nhiều nhóm, đạt chuẩn & đồng nhất phương sai → one-way ANOVA.", "anova")
        return ("Kruskal–Wallis", "Nhiều nhóm nhưng không đạt giả định → Kruskal–Wallis.", "kruskal")

    # y cat, x num (swap)
    if yk == "cat" and xk == "num":
        rep = assumption_report_num_by_group(df, y_num=x, group_cat=y)
        n_levels = len(rep["levels"])
        norm_ok = _norm_ok(rep)
        var_ok = _var_ok(rep)

        if n_levels == 2:
            if norm_ok and var_ok:
                return ("t-test (Student)", "2 nhóm, đạt chuẩn & phương sai tương đương → Student t-test.", "ttest_student_swapped")
            if norm_ok and (not var_ok):
                return ("t-test (Welch)", "2 nhóm, chuẩn nhưng phương sai khác → Welch t-test.", "ttest_welch_swapped")
            return ("Mann–Whitney U", "2 nhóm nhưng không đạt giả định chuẩn → Mann–Whitney.", "mwu_swapped")

        if norm_ok and var_ok:
            return ("ANOVA một yếu tố", "Nhiều nhóm, đạt chuẩn & đồng nhất phương sai → one-way ANOVA.", "anova_swapped")
        return ("Kruskal–Wallis", "Nhiều nhóm nhưng không đạt giả định → Kruskal–Wallis.", "kruskal_swapped")

    # num-num: Pearson vs Spearman
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
            return ("Tương quan Pearson", "X và Y gần chuẩn → Pearson correlation.", "corr_pearson")
        return ("Tương quan Spearman", "X hoặc Y không chuẩn/ordinal → Spearman correlation.", "corr_spearman")

    return ("Không xác định", "Không xác định được phép kiểm phù hợp từ kiểu biến hiện tại.", "none")


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
    # cat-cat
    if test_kind == "chisq":
        tmp = df[[y, x]].dropna()
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        chi2, p, dof, exp = stats.chi2_contingency(tab.values)
        v = _cramers_v(tab)
        out = pd.DataFrame({"Chỉ số": ["Chi2", "df", "p-value", "Cramer's V"], "Giá trị": [chi2, dof, p, v]})
        interp = "Diễn giải: p-value nhỏ gợi ý X và Y có liên quan. Cramer's V đánh giá độ mạnh liên quan."
        return out, interp

    if test_kind == "fisher_2x2":
        tmp = df[[y, x]].dropna()
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        if tab.shape != (2, 2):
            raise ValueError("Fisher exact chỉ áp dụng bảng 2x2.")
        oddsratio, p = stats.fisher_exact(tab.values)
        out = pd.DataFrame({"Chỉ số": ["Odds ratio", "p-value"], "Giá trị": [oddsratio, p]})
        interp = "Diễn giải: p-value nhỏ gợi ý có liên quan giữa 2 biến phân loại. Odds ratio diễn giải theo nhóm tham chiếu."
        return out, interp

    # y numeric, x categorical
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
            interp = f"{assump}\nDiễn giải: p-value nhỏ gợi ý trung bình Y khác nhau giữa 2 nhóm. Cohen’s d là effect size."
            return out, interp

        if test_kind == "mwu":
            if len(levels) != 2:
                raise ValueError("Mann–Whitney cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            out = pd.DataFrame({"Chỉ số": ["U", "p-value"], "Giá trị": [u, p]})
            interp = f"{assump}\nDiễn giải: Mann–Whitney dùng khi dữ liệu không đạt chuẩn."
            return out, interp

        if test_kind == "anova":
            f, p = stats.f_oneway(*arrays)
            out = pd.DataFrame({"Chỉ số": ["F", "p-value"], "Giá trị": [f, p]})
            interp = f"{assump}\nDiễn giải: p-value nhỏ gợi ý có ít nhất 1 nhóm khác trung bình; nên làm post-hoc."
            return out, interp

        if test_kind == "kruskal":
            h, p = stats.kruskal(*arrays)
            out = pd.DataFrame({"Chỉ số": ["H (Kruskal)", "p-value"], "Giá trị": [h, p]})
            interp = f"{assump}\nDiễn giải: Kruskal–Wallis dùng khi không đạt giả định; nếu có ý nghĩa nên làm post-hoc."
            return out, interp

    # swapped: x numeric by y groups
    if test_kind in ("ttest_student_swapped", "ttest_welch_swapped", "mwu_swapped", "anova_swapped", "kruskal_swapped"):
        tmp = df[[y, x]].dropna().copy()
        tmp[x] = coerce_numeric(tmp[x])
        tmp = tmp.dropna()
        groups = tmp[y].astype(str)
        levels = sorted(groups.unique().tolist())
        arrays = [tmp.loc[groups == lv, x].to_numpy() for lv in levels]
        rep = assumption_report_num_by_group(df, y_num=x, group_cat=y)
        assump = _assumption_text(rep)

        base_kind = test_kind.replace("_swapped", "")
        if base_kind in ("ttest_student", "ttest_welch"):
            if len(levels) != 2:
                raise ValueError("t-test cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            equal_var = (base_kind == "ttest_student")
            tstat, p = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
            d = _cohens_d(a, b)
            out = pd.DataFrame({"Chỉ số": ["t", "p-value", "Cohen's d"], "Giá trị": [tstat, p, d]})
            interp = f"{assump}\nDiễn giải: p-value nhỏ gợi ý trung bình X khác nhau giữa 2 nhóm Y."
            return out, interp

        if base_kind == "mwu":
            if len(levels) != 2:
                raise ValueError("Mann–Whitney cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            out = pd.DataFrame({"Chỉ số": ["U", "p-value"], "Giá trị": [u, p]})
            interp = f"{assump}\nDiễn giải: Mann–Whitney dùng khi dữ liệu không đạt chuẩn."
            return out, interp

        if base_kind == "anova":
            f, p = stats.f_oneway(*arrays)
            out = pd.DataFrame({"Chỉ số": ["F", "p-value"], "Giá trị": [f, p]})
            interp = f"{assump}\nDiễn giải: p-value nhỏ gợi ý có ít nhất 1 nhóm khác trung bình; nên làm post-hoc."
            return out, interp

        if base_kind == "kruskal":
            h, p = stats.kruskal(*arrays)
            out = pd.DataFrame({"Chỉ số": ["H (Kruskal)", "p-value"], "Giá trị": [h, p]})
            interp = f"{assump}\nDiễn giải: Kruskal–Wallis dùng khi không đạt giả định; nên làm post-hoc."
            return out, interp

    # correlation
    if test_kind == "corr_pearson":
        tmp = df[[y, x]].copy()
        tmp[y] = coerce_numeric(tmp[y])
        tmp[x] = coerce_numeric(tmp[x])
        tmp = tmp.dropna()
        r, p = stats.pearsonr(tmp[x].to_numpy(), tmp[y].to_numpy())
        pny = normality_pvalue(tmp[y].to_numpy())
        pnx = normality_pvalue(tmp[x].to_numpy())
        out = pd.DataFrame(
            {"Chỉ số": ["Pearson r", "p-value", "n", "Shapiro p(Y)", "Shapiro p(X)"], "Giá trị": [r, p, tmp.shape[0], pny, pnx]}
        )
        interp = "Diễn giải: r gần 0 → yếu; gần ±1 → mạnh. p-value nhỏ gợi ý liên quan tuyến tính có ý nghĩa."
        return out, interp

    if test_kind == "corr_spearman":
        tmp = df[[y, x]].copy()
        tmp[y] = coerce_numeric(tmp[y])
        tmp[x] = coerce_numeric(tmp[x])
        tmp = tmp.dropna()
        rho, p = stats.spearmanr(tmp[x].to_numpy(), tmp[y].to_numpy())
        pny = normality_pvalue(tmp[y].to_numpy())
        pnx = normality_pvalue(tmp[x].to_numpy())
        out = pd.DataFrame(
            {"Chỉ số": ["Spearman rho", "p-value", "n", "Shapiro p(Y)", "Shapiro p(X)"], "Giá trị": [rho, p, tmp.shape[0], pny, pnx]}
        )
        interp = "Diễn giải: Spearman đánh giá liên quan đơn điệu (không cần chuẩn), phù hợp khi dữ liệu không chuẩn/ordinal."
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
            return ("Không đủ dữ liệu", "Biến phụ thuộc chỉ có 0–1 mức sau khi loại thiếu. Hãy kiểm tra dữ liệu.")
        if n_levels == 2:
            return (
                "Hồi quy Logistic nhị phân (Binary Logistic)",
                "Y phân loại 2 mức → logistic nhị phân để ước lượng OR và p-value khi có nhiều biến độc lập.",
            )
        return (
            "Hồi quy Logistic đa danh (Multinomial Logistic)",
            f"Y phân loại >2 mức (mức={n_levels}) → logistic đa danh (multinomial).",
        )
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
    out = pd.DataFrame({"Hệ số": fit.params, "CI 2.5%": conf[0], "CI 97.5%": conf[1], "p-value": fit.pvalues})
    out.index.name = "Biến"
    return out.sort_values("p-value")


def logit_or_table(fit) -> pd.DataFrame:
    conf = fit.conf_int()
    out = pd.DataFrame(
        {"OR": np.exp(fit.params), "CI 2.5%": np.exp(conf[0]), "CI 97.5%": np.exp(conf[1]), "p-value": fit.pvalues}
    )
    out.index.name = "Biến"
    return out.sort_values("p-value")


# =========================
# Session state: datasets + dedupe + stepper
# =========================
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}  # key -> df
if "active_name" not in st.session_state:
    st.session_state["active_name"] = None

if "pending_tables" not in st.session_state:
    st.session_state["pending_tables"] = None
if "pending_fname" not in st.session_state:
    st.session_state["pending_fname"] = None
if "pending_file_hash" not in st.session_state:
    st.session_state["pending_file_hash"] = None

if "hash_to_key" not in st.session_state:
    st.session_state["hash_to_key"] = {}  # hash -> dataset key
if "key_to_hashes" not in st.session_state:
    st.session_state["key_to_hashes"] = {}  # dataset key -> set(hashes)
if "last_upload_hash" not in st.session_state:
    st.session_state["last_upload_hash"] = None

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_run_meta" not in st.session_state:
    st.session_state["last_run_meta"] = None

if "active_step" not in st.session_state:
    st.session_state["active_step"] = 1  # 1=Data,2=Choose,3=Results


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
# UI Header
# =========================
st.markdown(
    f"""
    <div style="padding: 0.25rem 0 0.5rem 0;">
      <h1 style="margin:0;">{APP_TITLE}</h1>
      <div style="color:#6b7280;">
        Upload dữ liệu → chọn biến → kiểm định (1 X) hoặc mô hình (nhiều X) → kết quả + giải thích
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Sidebar: Upload & Dataset manager
# =========================
with st.sidebar:
    st.markdown("## 🧪 Dữ liệu")

    up = st.file_uploader(
        "⬆️ Upload dữ liệu",
        type=["csv", "xlsx", "xls", "sav", "zsav", "dta", "rds"],
        accept_multiple_files=False,
    )

    if up is not None:
        try:
            raw = up.getvalue()
            file_hash = _file_sha256(raw)

            # tránh rerun add lại
            if st.session_state["last_upload_hash"] != file_hash:
                st.session_state["last_upload_hash"] = file_hash

                # file đã có
                if file_hash in st.session_state["hash_to_key"]:
                    existed_key = st.session_state["hash_to_key"][file_hash]
                    st.session_state["active_name"] = existed_key
                    st.info(f"Đã có trước đó → chuyển sang: {existed_key}")
                else:
                    tables = read_file_safely(up)

                    # nhiều sheet/object
                    if len(tables) > 1:
                        st.session_state["pending_tables"] = tables
                        st.session_state["pending_fname"] = up.name
                        st.session_state["pending_file_hash"] = file_hash
                        st.info(f"File có {len(tables)} bảng. Chọn 1 bảng để nhập.")
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
                        st.success(f"Đã tải: {key} ({df_new.shape[0]}x{df_new.shape[1]})")

        except Exception as e:
            st.error(f"Không đọc được file: {e}")

    # pending sheet/object
    if st.session_state["pending_tables"] is not None:
        st.markdown("### Chọn sheet/object")
        tables = st.session_state["pending_tables"]
        fname = st.session_state["pending_fname"] or "file"
        pending_file_hash = st.session_state["pending_file_hash"]

        chosen_table = st.selectbox("Sheet/Object", options=list(tables.keys()))
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("✅ Nhập", use_container_width=True):
                df_new = tables[chosen_table]
                table_hash = _df_sha256(df_new)

                if table_hash in st.session_state["hash_to_key"]:
                    existed_key = st.session_state["hash_to_key"][table_hash]
                    st.session_state["active_name"] = existed_key
                    st.info(f"Bảng đã nhập → {existed_key}")
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
                    st.success(f"Đã nhập: {key} ({df_new.shape[0]}x{df_new.shape[1]})")

                st.session_state["pending_tables"] = None
                st.session_state["pending_fname"] = None
                st.session_state["pending_file_hash"] = None
                st.rerun()

        with c2:
            if st.button("❌ Huỷ", use_container_width=True):
                st.session_state["pending_tables"] = None
                st.session_state["pending_fname"] = None
                st.session_state["pending_file_hash"] = None
                st.rerun()

    st.markdown("---")
    st.markdown("## 📁 Dataset")

    names_all = list(st.session_state["datasets"].keys())
    if not names_all:
        st.info("Chưa có dữ liệu. Upload để bắt đầu.")
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
    summ = overall_summary(df_active)
    st.caption(f"rows={summ['Số dòng']} | biến={summ['Số biến']} | thiếu={summ['Ô thiếu (NA)']}")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🗑️ Xoá dataset", use_container_width=True):
            _delete_dataset(chosen)
            remaining = list(st.session_state["datasets"].keys())
            st.session_state["active_name"] = remaining[0] if remaining else None
            st.session_state["last_result"] = None
            st.session_state["last_run_meta"] = None
            st.session_state["active_step"] = 1
            st.rerun()

    with c2:
        if st.button("🧹 Xoá tất cả", use_container_width=True):
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
# Data in main
# =========================
df = st.session_state["datasets"][st.session_state["active_name"]]
cols = df.columns.tolist()


# =========================
# Stepper Buttons (BIG)
# =========================
st.markdown("### 🧭 Các bước thao tác")

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    t = "primary" if st.session_state["active_step"] == 1 else "secondary"
    if st.button("1) 📄 Dữ liệu", type=t, use_container_width=True):
        st.session_state["active_step"] = 1
        st.rerun()
    st.markdown('<div class="step-caption">Tổng quan • xem bảng • danh sách biến</div>', unsafe_allow_html=True)

with c2:
    t = "primary" if st.session_state["active_step"] == 2 else "secondary"
    if st.button("2) 🎯 Chọn biến", type=t, use_container_width=True):
        st.session_state["active_step"] = 2
        st.rerun()
    st.markdown('<div class="step-caption">Chọn Y/X • xem gợi ý • bấm Run</div>', unsafe_allow_html=True)

with c3:
    t = "primary" if st.session_state["active_step"] == 3 else "secondary"
    if st.button("3) 📌 Kết quả", type=t, use_container_width=True):
        st.session_state["active_step"] = 3
        st.rerun()
    st.markdown('<div class="step-caption">Bảng kết quả • biểu đồ • diễn giải</div>', unsafe_allow_html=True)

st.divider()


# =========================
# Compute and store results
# =========================
def _compute_and_store(y: str, xs: List[str], y_force: str, x_force: str, y_event: Optional[str]):
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

    # model
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
# STEP 1: Data view
# =========================
if st.session_state["active_step"] == 1:
    st.subheader("📄 Dữ liệu")

    summ = overall_summary(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Số dòng", summ["Số dòng"])
    m2.metric("Số biến", summ["Số biến"])
    m3.metric("Định lượng", summ["Biến định lượng"])
    m4.metric("Phân loại", summ["Biến phân loại"])
    m5.metric("Ô thiếu (NA)", summ["Ô thiếu (NA)"])

    st.markdown("### 👀 Xem nhanh dữ liệu")
    st.dataframe(df.head(30), use_container_width=True, height=260)

    st.markdown("### 🧾 Liệt kê biến & đặc tính")
    var_rows = [summarize_variable(df, c) for c in cols]
    var_df = pd.DataFrame(var_rows)

    v1, v2 = st.columns([1.2, 1.0])
    with v1:
        q = st.text_input("Tìm nhanh tên biến", value="", placeholder="vd: age, weight...")
    with v2:
        filter_opt = st.selectbox("Lọc nhanh", ["Tất cả", "Chỉ định lượng", "Chỉ phân loại"], index=0)

    if q.strip():
        var_df = var_df[var_df["Tên biến"].str.contains(q.strip(), case=False, na=False)].copy()

    if filter_opt == "Chỉ định lượng":
        var_df = var_df[var_df["Đặc tính biến"].str.contains("Định lượng", na=False)]
    elif filter_opt == "Chỉ phân loại":
        var_df = var_df[var_df["Đặc tính biến"].str.contains("Phân loại", na=False)]

    st.dataframe(var_df, use_container_width=True, height=420)

    st.info("👉 Bấm **2) Chọn biến** để chọn Y/X và chạy kiểm định/mô hình.")


# =========================
# STEP 2: Choose variables
# =========================
elif st.session_state["active_step"] == 2:
    st.subheader("🎯 Chọn biến phân tích")

    left, right = st.columns([2.2, 1.0], gap="large")

    with left:
        vq = st.text_input("Tìm biến", value="", placeholder="gõ tên biến...")
        cols_show = [c for c in cols if vq.lower() in c.lower()] if vq.strip() else cols
        if not cols_show:
            cols_show = cols

        y = st.selectbox("Biến phụ thuộc (Y)", options=cols_show, index=0)
        xs = st.multiselect("Biến độc lập (X) (có thể chọn nhiều)", options=[c for c in cols_show if c != y])

        st.markdown("**Ép kiểu nếu cần**")
        force_opts = ["Tự động", "Định lượng (numeric)", "Phân loại (categorical)"]
        y_force = st.selectbox("Kiểu Y", options=force_opts, index=0)
        x_force = "Tự động"
        if len(xs) == 1:
            x_force = st.selectbox("Kiểu X (chỉ áp dụng khi chọn 1 X)", options=force_opts, index=0)

        y_event = None
        if var_kind(df[y], y_force) == "cat":
            levels = sorted(df[y].dropna().astype(str).unique().tolist())
            if len(levels) == 2:
                y_event = st.selectbox("Chọn mức coi là 'Sự kiện' (Y=1) (logistic)", options=levels, index=1)

        st.markdown("#### ✅ Gợi ý")
        if len(xs) == 0:
            st.info("Chọn ít nhất 1 biến X.")
            suggestion = None
            explanation = None
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
        st.markdown("#### Tóm tắt lựa chọn")
        st.write(f"**Dataset:** {st.session_state['active_name']}")
        st.write(f"**Y:** {y} ({'định lượng' if var_kind(df[y], y_force)=='num' else 'phân loại'})")

        if len(xs) == 0:
            st.write("**X:** -")
            st.button("▶️ Run", type="primary", use_container_width=True, disabled=True)
        else:
            if len(xs) == 1:
                x1 = xs[0]
                xk = var_kind(df[x1], x_force)
                st.write(f"**X:** {x1} ({'định lượng' if xk=='num' else 'phân loại'})")
            else:
                st.write(f"**X:** {len(xs)} biến")

            st.markdown("---")
            if st.button("▶️ Run", type="primary", use_container_width=True):
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
        st.info("Chưa có kết quả. Bấm **2) Chọn biến** → chọn Y/X → bấm **Run**.")
    else:
        st.markdown("#### Tóm tắt lần chạy")
        st.write(f"- **Dataset:** {meta.get('dataset')}")
        st.write(f"- **Y:** {meta.get('y')}")
        st.write(f"- **X:** {', '.join(meta.get('xs', []))}")
        st.write(f"- **Gợi ý:** {meta.get('suggestion')}")
        st.divider()

        left, right = st.columns([1.4, 1.0], gap="large")

        with left:
            if meta["mode"] == "test":
                st.markdown("### 📊 Kết quả kiểm định")
                st.dataframe(res["table"], use_container_width=True)
                st.markdown("### 🔎 Diễn giải")
                st.write(res["interp"])
            else:
                st.caption(meta.get("note", ""))
                kind = res["kind"]
                if kind in ("ols", "logit") and res["table"] is not None:
                    st.markdown("### 📊 Bảng kết quả mô hình")
                    st.dataframe(res["table"], use_container_width=True)
                    st.markdown("### 🔎 Diễn giải")
                    if kind == "ols":
                        st.write(
                            "- Hệ số > 0: Y tăng khi X tăng (giữ các biến khác).\n"
                            "- p-value < 0.05: thường có ý nghĩa.\n"
                            "- CI 95% không chứa 0: thường có ý nghĩa."
                        )
                    else:
                        st.write(
                            "- OR > 1: tăng odds sự kiện.\n"
                            "- OR < 1: giảm odds.\n"
                            "- CI 95% không chứa 1 và p<0.05: thường có ý nghĩa."
                        )
                else:
                    st.markdown("### 📄 MNLogit Summary")
                    st.write(res["fit"].summary())

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
                        fig = px.box(tmp, x=x1, y=y, points="all", title=f"{y} theo nhóm {x1}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif yk == "cat" and xk == "num":
                        tmp[x1] = coerce_numeric(tmp[x1])
                        tmp = tmp.dropna()
                        fig = px.box(tmp, x=y, y=x1, points="all", title=f"{x1} theo nhóm {y}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif yk == "cat" and xk == "cat":
                        tab = pd.crosstab(tmp[y].astype(str), tmp[x1].astype(str))
                        tab2 = tab.div(tab.sum(axis=1), axis=0).reset_index().melt(
                            id_vars=[y], var_name=x1, value_name="Tỷ lệ"
                        )
                        fig = px.bar(tab2, x=y, y="Tỷ lệ", color=x1, barmode="stack", title=f"Tỷ lệ {x1} theo {y}")
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        tmp[y] = coerce_numeric(tmp[y])
                        tmp[x1] = coerce_numeric(tmp[x1])
                        tmp = tmp.dropna()
                        fig = px.scatter(tmp, x=x1, y=y, trendline="ols", title=f"{y} ~ {x1}")
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    kind = res["kind"]
                    data_used = res["data_used"]
                    fit = res["fit"]
                    y = meta["y"]
                    xs = meta["xs"]

                    if kind == "ols":
                        x1 = xs[0]
                        if (not is_categorical(data_used[x1])) and (not is_categorical(data_used[y])):
                            fig = px.scatter(data_used, x=x1, y=y, trendline="ols", title=f"{y} ~ {x1} (trendline)")
                        else:
                            fig = (
                                px.box(data_used, x=x1, y=y, points="all", title=f"{y} theo nhóm {x1}")
                                if is_categorical(data_used[x1])
                                else px.scatter(data_used, x=x1, y=y, title=f"{y} theo {x1}")
                            )
                        st.plotly_chart(fig, use_container_width=True)

                        pred = fit.fittedvalues
                        tmp_plot = pd.DataFrame({"Thực tế": data_used[y], "Dự đoán": pred})
                        fig2 = px.scatter(tmp_plot, x="Thực tế", y="Dự đoán", title="Dự đoán vs Thực tế")
                        st.plotly_chart(fig2, use_container_width=True)

                    elif kind == "logit":
                        p = fit.predict()
                        fig = px.histogram(p, nbins=25, title="Phân bố xác suất dự đoán (p)")
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.info("Multinomial: có thể bổ sung biểu đồ RRR / xác suất dự đoán theo nhu cầu.")
            except Exception as e:
                st.warning(f"Không vẽ được biểu đồ: {e}")

    st.divider()
    st.caption(
        "⚠️ Lưu ý: Công cụ hỗ trợ gợi ý và chạy kiểm định/mô hình cơ bản. "
        "Người dùng cần kiểm tra giả định, thiết kế nghiên cứu và cách mã hoá biến để diễn giải đúng."
    )
