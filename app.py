import io
import re
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

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
# Helpers: safe name / hashing
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
    Some readers (SPSS/STATA) may require a file path, not BytesIO.
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
      - .sav/.zsav (SPSS) (pyreadstat backend)
      - .dta (STATA) (pyreadstat backend)
      - .rds (pyreadr)
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

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            res = pyreadr.read_r(tmp_path)
            out: Dict[str, pd.DataFrame] = {}
            for k, v in res.items():
                if isinstance(v, pd.DataFrame):
                    out[str(k) if k else "data"] = v
            if not out:
                raise RuntimeError("File .rds không chứa DataFrame (hoặc object không hỗ trợ).")
            return out
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

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
    """
    - n < 3: nan
    - n <= 5000: Shapiro-Wilk
    - n > 5000: D’Agostino K2
    """
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
    """
    Levene test (robust)
    """
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
    # if any group has <3, normality can't be assessed -> treat as not ok
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


# =========================
# Single-X: suggest + run (with assumptions)
# =========================
def suggest_single_x_test(
    df: pd.DataFrame,
    y: str,
    x: str,
    y_forced: str = "Tự động",
    x_forced: str = "Tự động",
) -> Tuple[str, str, str]:
    """
    Returns (suggestion, explanation, test_kind)
    """
    yk = var_kind(df[y], y_forced)
    xk = var_kind(df[x], x_forced)

    tmp = df[[y, x]].dropna()
    if tmp.shape[0] < 3:
        return ("Không đủ dữ liệu", "Sau khi loại NA, số dòng quá ít để kiểm định.", "none")

    # cat - cat
    if yk == "cat" and xk == "cat":
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        if tab.shape == (2, 2) and (tab.values < 5).any():
            return ("Fisher exact (2x2)", "Bảng 2x2 và có ô nhỏ → ưu tiên Fisher exact.", "fisher_2x2")
        return ("Chi-bình phương (Chi-square)", "X và Y đều phân loại → kiểm định độc lập bằng Chi-square.", "chisq")

    # num - cat: compare numeric across groups
    if yk == "num" and xk == "cat":
        rep = assumption_report_num_by_group(df, y_num=y, group_cat=x)
        n_levels = len(rep["levels"])
        norm_ok = _norm_ok(rep)
        var_ok = _var_ok(rep)

        if n_levels == 2:
            if norm_ok and var_ok:
                return ("t-test độc lập (Student)", "2 nhóm, chuẩn & phương sai tương đương → t-test Student.", "ttest_student")
            if norm_ok and (not var_ok):
                return ("t-test độc lập (Welch)", "2 nhóm, chuẩn nhưng phương sai khác → Welch t-test.", "ttest_welch")
            return ("Mann–Whitney U", "2 nhóm nhưng không đạt giả định chuẩn → Mann–Whitney (phi tham số).", "mwu")

        if norm_ok and var_ok:
            return ("ANOVA một yếu tố", "Nhiều nhóm, chuẩn & đồng nhất phương sai → one-way ANOVA.", "anova")
        return ("Kruskal–Wallis", "Nhiều nhóm nhưng không đạt giả định → Kruskal–Wallis (phi tham số).", "kruskal")

    # cat - num: swap roles (compare numeric X across Y groups)
    if yk == "cat" and xk == "num":
        rep = assumption_report_num_by_group(df, y_num=x, group_cat=y)
        n_levels = len(rep["levels"])
        norm_ok = _norm_ok(rep)
        var_ok = _var_ok(rep)

        if n_levels == 2:
            if norm_ok and var_ok:
                return ("t-test độc lập (Student)", "2 nhóm, chuẩn & phương sai tương đương → t-test Student.", "ttest_student_swapped")
            if norm_ok and (not var_ok):
                return ("t-test độc lập (Welch)", "2 nhóm, chuẩn nhưng phương sai khác → Welch t-test.", "ttest_welch_swapped")
            return ("Mann–Whitney U", "2 nhóm nhưng không đạt giả định chuẩn → Mann–Whitney (phi tham số).", "mwu_swapped")

        if norm_ok and var_ok:
            return ("ANOVA một yếu tố", "Nhiều nhóm, chuẩn & đồng nhất phương sai → one-way ANOVA.", "anova_swapped")
        return ("Kruskal–Wallis", "Nhiều nhóm nhưng không đạt giả định → Kruskal–Wallis (phi tham số).", "kruskal_swapped")

    # num - num: correlation; choose Pearson if both ~normal else Spearman
    if yk == "num" and xk == "num":
        yv = coerce_numeric(df[y]).dropna().to_numpy()
        xv = coerce_numeric(df[x]).dropna().to_numpy()
        # align quickly with pairwise dropna
        tmp2 = df[[y, x]].copy()
        tmp2[y] = coerce_numeric(tmp2[y])
        tmp2[x] = coerce_numeric(tmp2[x])
        tmp2 = tmp2.dropna()
        if tmp2.shape[0] < 3:
            return ("Không đủ dữ liệu", "Không đủ dòng số để tính tương quan.", "none")

        pny = normality_pvalue(tmp2[y].to_numpy())
        pnx = normality_pvalue(tmp2[x].to_numpy())
        if (not np.isnan(pny)) and (not np.isnan(pnx)) and (pny >= 0.05) and (pnx >= 0.05):
            return ("Tương quan Pearson", "X và Y gần chuẩn → dùng Pearson correlation.", "corr_pearson")
        return ("Tương quan Spearman", "X hoặc Y không chuẩn/ordinal → dùng Spearman correlation.", "corr_spearman")

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


def _assumption_text(rep: dict) -> str:
    norm = ", ".join([f"{k}: p={rep['normality_p'][k]:.4f}" if not np.isnan(rep['normality_p'][k]) else f"{k}: p=NA"
                      for k in rep["levels"]])
    lev = rep.get("levene_p", float("nan"))
    lev_s = f"{lev:.4f}" if not np.isnan(lev) else "NA"
    return f"Kiểm tra giả định: Shapiro theo nhóm [{norm}]; Levene p={lev_s}."


def run_single_x_test(df: pd.DataFrame, y: str, x: str, test_kind: str) -> Tuple[pd.DataFrame, str]:
    """
    Return (result_table, interpretation_text)
    """
    # --- cat-cat ---
    if test_kind == "chisq":
        tmp = df[[y, x]].dropna()
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        chi2, p, dof, exp = stats.chi2_contingency(tab.values)
        v = _cramers_v(tab)
        out = pd.DataFrame({"Chỉ số": ["Chi2", "df", "p-value", "Cramer's V"], "Giá trị": [chi2, dof, p, v]})
        interp = (
            "Diễn giải: p-value nhỏ gợi ý X và Y có liên quan (không độc lập). "
            "Cramer's V cho biết độ mạnh liên quan (≈0.1 nhỏ, 0.3 vừa, 0.5 lớn – tuỳ bối cảnh)."
        )
        return out, interp

    if test_kind == "fisher_2x2":
        tmp = df[[y, x]].dropna()
        tab = pd.crosstab(tmp[y].astype(str), tmp[x].astype(str))
        if tab.shape != (2, 2):
            raise ValueError("Fisher exact chỉ áp dụng bảng 2x2.")
        oddsratio, p = stats.fisher_exact(tab.values)
        out = pd.DataFrame({"Chỉ số": ["Odds ratio", "p-value"], "Giá trị": [oddsratio, p]})
        interp = (
            "Diễn giải: p-value nhỏ gợi ý có liên quan giữa 2 biến phân loại. "
            "Odds ratio >1 cho thấy odds cao hơn ở một nhóm (xem nhóm tham chiếu từ bảng 2x2)."
        )
        return out, interp

    # --- num-cat (Y numeric, X group) ---
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
            out = pd.DataFrame(
                {"Chỉ số": ["t", "p-value", "Cohen's d"], "Giá trị": [tstat, p, d]}
            )
            interp = (
                f"{assump}\n"
                "Diễn giải: p-value nhỏ gợi ý trung bình Y khác nhau giữa 2 nhóm. "
                "Cohen’s d: ~0.2 nhỏ, 0.5 vừa, 0.8 lớn (tham khảo)."
            )
            return out, interp

        if test_kind == "mwu":
            if len(levels) != 2:
                raise ValueError("Mann–Whitney cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            out = pd.DataFrame({"Chỉ số": ["U", "p-value"], "Giá trị": [u, p]})
            interp = (
                f"{assump}\n"
                "Diễn giải: Mann–Whitney so sánh phân bố/median giữa 2 nhóm khi không đạt giả định chuẩn."
            )
            return out, interp

        if test_kind == "anova":
            f, p = stats.f_oneway(*arrays)
            out = pd.DataFrame({"Chỉ số": ["F", "p-value"], "Giá trị": [f, p]})
            interp = (
                f"{assump}\n"
                "Diễn giải: p-value nhỏ gợi ý có ít nhất 1 nhóm khác trung bình; nếu có ý nghĩa nên làm post-hoc."
            )
            return out, interp

        if test_kind == "kruskal":
            h, p = stats.kruskal(*arrays)
            out = pd.DataFrame({"Chỉ số": ["H (Kruskal)", "p-value"], "Giá trị": [h, p]})
            interp = (
                f"{assump}\n"
                "Diễn giải: Kruskal–Wallis dùng khi không đạt giả định; nếu có ý nghĩa nên làm post-hoc phi tham số."
            )
            return out, interp

    # --- swapped: X numeric, Y group ---
    if test_kind in ("ttest_student_swapped", "ttest_welch_swapped", "mwu_swapped", "anova_swapped", "kruskal_swapped"):
        # swap y <-> x in the numeric-by-group framework:
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
            interp = (
                f"{assump}\n"
                "Diễn giải: p-value nhỏ gợi ý trung bình X khác nhau giữa 2 nhóm Y."
            )
            return out, interp

        if base_kind == "mwu":
            if len(levels) != 2:
                raise ValueError("Mann–Whitney cần đúng 2 nhóm.")
            a, b = arrays[0], arrays[1]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            out = pd.DataFrame({"Chỉ số": ["U", "p-value"], "Giá trị": [u, p]})
            interp = f"{assump}\nDiễn giải: Mann–Whitney dùng khi không đạt giả định chuẩn."
            return out, interp

        if base_kind == "anova":
            f, p = stats.f_oneway(*arrays)
            out = pd.DataFrame({"Chỉ số": ["F", "p-value"], "Giá trị": [f, p]})
            interp = f"{assump}\nDiễn giải: p-value nhỏ gợi ý có ít nhất 1 nhóm khác trung bình; nên làm post-hoc."
            return out, interp

        if base_kind == "kruskal":
            h, p = stats.kruskal(*arrays)
            out = pd.DataFrame({"Chỉ số": ["H (Kruskal)", "p-value"], "Giá trị": [h, p]})
            interp = f"{assump}\nDiễn giải: Kruskal–Wallis dùng khi không đạt giả định; nên làm post-hoc phi tham số."
            return out, interp

    # --- correlation ---
    if test_kind == "corr_pearson":
        tmp = df[[y, x]].copy()
        tmp[y] = coerce_numeric(tmp[y])
        tmp[x] = coerce_numeric(tmp[x])
        tmp = tmp.dropna()
        r, p = stats.pearsonr(tmp[x].to_numpy(), tmp[y].to_numpy())
        pny = normality_pvalue(tmp[y].to_numpy())
        pnx = normality_pvalue(tmp[x].to_numpy())
        out = pd.DataFrame({"Chỉ số": ["Pearson r", "p-value", "n", "Shapiro p(Y)", "Shapiro p(X)"], "Giá trị": [r, p, tmp.shape[0], pny, pnx]})
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
        out = pd.DataFrame({"Chỉ số": ["Spearman rho", "p-value", "n", "Shapiro p(Y)", "Shapiro p(X)"], "Giá trị": [rho, p, tmp.shape[0], pny, pnx]})
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
                "Y phân loại 2 mức → phù hợp logistic nhị phân để ước lượng OR và p-value khi có nhiều biến độc lập.",
            )
        return (
            "Hồi quy Logistic đa danh (Multinomial Logistic)",
            f"Y phân loại >2 mức (mức={n_levels}) → phù hợp logistic đa danh.",
        )
    return ("Hồi quy tuyến tính (OLS)", "Y định lượng → phù hợp hồi quy tuyến tính (OLS).")


def build_formula(
    df: pd.DataFrame,
    y: str,
    xs: List[str],
    y_binary_event: str | None = None,
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
        return formula, tmp, "mnlogit||Multinomial: hệ số theo nhóm tham chiếu (mã hoá category)"

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
# Session state: datasets + dedupe
# =========================
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}  # key -> df
if "active_name" not in st.session_state:
    st.session_state["active_name"] = None

# pending tables (Excel/RDS multiple)
if "pending_tables" not in st.session_state:
    st.session_state["pending_tables"] = None
if "pending_fname" not in st.session_state:
    st.session_state["pending_fname"] = None
if "pending_file_hash" not in st.session_state:
    st.session_state["pending_file_hash"] = None

# dedupe maps
if "hash_to_key" not in st.session_state:
    st.session_state["hash_to_key"] = {}  # hash -> key
if "key_to_hashes" not in st.session_state:
    st.session_state["key_to_hashes"] = {}  # key -> set(hash)
if "last_upload_hash" not in st.session_state:
    st.session_state["last_upload_hash"] = None


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
      <div style="color:#6b7280;">Upload dữ liệu → chọn biến → (1 X: kiểm định có kiểm tra giả định) | (nhiều X: mô hình hồi quy)</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()


# =========================
# Top row (Overview | Upload | File list)
# =========================
col_left, col_mid, col_right = st.columns([2.2, 1.6, 2.2], gap="large")

with col_mid:
    st.subheader("⬆️ Upload file")
    up = st.file_uploader(
        "Tải lên dữ liệu",
        type=["csv", "xlsx", "xls", "sav", "zsav", "dta", "rds"],
        accept_multiple_files=False,
    )

    if up is not None:
        try:
            raw = up.getvalue()
            file_hash = _file_sha256(raw)

            # prevent rerun duplicate
            if st.session_state["last_upload_hash"] != file_hash:
                st.session_state["last_upload_hash"] = file_hash

                if file_hash in st.session_state["hash_to_key"]:
                    existed = st.session_state["hash_to_key"][file_hash]
                    st.session_state["active_name"] = existed
                    st.info(f"File này đã upload trước đó → chuyển sang: {existed}")
                else:
                    tables = read_file_safely(up)

                    if len(tables) > 1:
                        st.session_state["pending_tables"] = tables
                        st.session_state["pending_fname"] = up.name
                        st.session_state["pending_file_hash"] = file_hash
                        st.info(f"File có {len(tables)} bảng (sheet/object). Chọn 1 bảng để nhập.")
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
                        st.success(f"Đã tải: {key} (rows={df_new.shape[0]}, cols={df_new.shape[1]})")

        except Exception as e:
            st.error(f"Không đọc được file: {e}")

    # pending import
    if st.session_state["pending_tables"] is not None:
        tables = st.session_state["pending_tables"]
        fname = st.session_state["pending_fname"] or "file"
        pending_file_hash = st.session_state["pending_file_hash"]

        chosen_table = st.selectbox("Chọn sheet/object", options=list(tables.keys()))
        c1, c2 = st.columns([1, 1])

        with c1:
            if st.button("✅ Nhập bảng đã chọn", use_container_width=True):
                df_new = tables[chosen_table]
                table_hash = _df_sha256(df_new)

                if table_hash in st.session_state["hash_to_key"]:
                    existed = st.session_state["hash_to_key"][table_hash]
                    st.session_state["active_name"] = existed
                    st.info(f"Bảng này đã nhập trước đó → chuyển sang: {existed}")
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
                    st.success(f"Đã nhập: {key} (rows={df_new.shape[0]}, cols={df_new.shape[1]})")

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

with col_right:
    st.subheader("📁 Danh sách file đã upload")
    names = list(st.session_state["datasets"].keys())
    if len(names) == 0:
        st.info("Chưa có file nào. Hãy upload ở cột giữa.")
    else:
        active = st.session_state["active_name"] or names[0]
        chosen = st.radio(
            "Click để chọn file",
            options=names,
            index=names.index(active) if active in names else 0,
            label_visibility="collapsed",
        )
        st.session_state["active_name"] = chosen

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🗑️ Xóa file đang chọn", use_container_width=True):
                _delete_dataset(chosen)
                remaining = list(st.session_state["datasets"].keys())
                st.session_state["active_name"] = remaining[0] if remaining else None
                st.rerun()
        with c2:
            if st.button("🧹 Xóa tất cả", use_container_width=True):
                st.session_state["datasets"] = {}
                st.session_state["active_name"] = None
                st.session_state["pending_tables"] = None
                st.session_state["pending_fname"] = None
                st.session_state["pending_file_hash"] = None
                st.session_state["hash_to_key"] = {}
                st.session_state["key_to_hashes"] = {}
                st.session_state["last_upload_hash"] = None
                st.rerun()

with col_left:
    st.subheader("📌 Tổng quan dữ liệu")
    if st.session_state["active_name"] is None:
        st.info("Upload và chọn 1 file để xem tổng quan.")
    else:
        df = st.session_state["datasets"][st.session_state["active_name"]]
        summ = overall_summary(df)
        st.write(
            f"- **Số dòng:** {summ['Số dòng']}\n"
            f"- **Số biến:** {summ['Số biến']}\n"
            f"- **Biến định lượng:** {summ['Biến định lượng']}\n"
            f"- **Biến phân loại:** {summ['Biến phân loại']}\n"
            f"- **Ô thiếu (NA):** {summ['Ô thiếu (NA)']}"
        )

st.divider()


# =========================
# Main area
# =========================
if st.session_state["active_name"] is None:
    st.stop()

df = st.session_state["datasets"][st.session_state["active_name"]]
cols = df.columns.tolist()

main_left, main_right = st.columns([2.4, 1.6], gap="large")

with main_left:
    st.subheader("🧾 Liệt kê biến & đặc tính")
    var_rows = [summarize_variable(df, c) for c in cols]
    var_df = pd.DataFrame(var_rows)

    q = st.text_input("Tìm nhanh tên biến", value="")
    if q.strip():
        mask = var_df["Tên biến"].str.contains(q.strip(), case=False, na=False)
        var_df = var_df.loc[mask].copy()

    st.dataframe(var_df, use_container_width=True, height=420)

with main_right:
    st.subheader("🎯 Chọn biến phân tích")
    y = st.selectbox("Chọn biến phụ thuộc (Y)", options=cols, index=0)
    x = st.multiselect("Chọn biến độc lập (có thể chọn nhiều biến)", options=[c for c in cols if c != y])

    st.markdown("**Ép kiểu nếu cần** (để tránh nhận sai 0/1 thành số đo):")
    force_opts = ["Tự động", "Định lượng (numeric)", "Phân loại (categorical)"]
    y_force = st.selectbox("Kiểu Y", options=force_opts, index=0)

    x_force = "Tự động"
    if len(x) == 1:
        x_force = st.selectbox("Kiểu X (chỉ áp dụng khi chọn 1 biến X)", options=force_opts, index=0)

    # logistic event if binary categorical Y
    y_is_cat = var_kind(df[y], y_force) == "cat"
    y_event = None
    if y_is_cat:
        levels = sorted(df[y].dropna().astype(str).unique().tolist())
        if len(levels) == 2:
            y_event = st.selectbox("Chọn mức coi là 'Sự kiện' (Y=1) cho logistic", options=levels, index=1)

    if len(x) == 0:
        st.info("Chọn ít nhất 1 biến độc lập để phần mềm gợi ý và chạy kết quả.")
        st.stop()

    # Decide mode
    if len(x) == 1:
        suggestion, explanation, test_kind = suggest_single_x_test(df, y, x[0], y_forced=y_force, x_forced=x_force)
        analysis_mode = "test"
    else:
        tmp_for_suggest = df.copy()
        if y_force == "Định lượng (numeric)":
            tmp_for_suggest[y] = coerce_numeric(tmp_for_suggest[y])
        elif y_force == "Phân loại (categorical)":
            tmp_for_suggest[y] = tmp_for_suggest[y].astype("string")

        suggestion, explanation = suggest_model(tmp_for_suggest, y, x)
        test_kind = "none"
        analysis_mode = "model"

    st.divider()
    st.subheader("✅ Phép kiểm / mô hình gợi ý")
    st.write("**Chế độ:** " + ("Kiểm định (1 biến độc lập)" if analysis_mode == "test" else "Mô hình hồi quy (nhiều biến độc lập)"))
    st.write(f"**Gợi ý:** {suggestion}")

    with st.expander("Giải thích tại sao chọn phương pháp này"):
        st.write(explanation)
        st.write(
            "- Nếu chỉ chọn **1 biến độc lập**, app kiểm tra **giả định** (phân phối chuẩn, đồng nhất phương sai) để chọn t-test/ANOVA hay Mann–Whitney/Kruskal.\n"
            "- Nếu chọn **nhiều biến độc lập**, app ưu tiên **mô hình hồi quy** để **hiệu chỉnh (adjust)** đồng thời.\n"
            "- Dữ liệu dùng để chạy sẽ **loại dòng thiếu (NA)** theo các biến đã chọn."
        )

    model_formula = None
    model_data_used = None
    model_kind = None

    if analysis_mode == "model":
        df_model = df.copy()
        if y_force == "Định lượng (numeric)":
            df_model[y] = coerce_numeric(df_model[y])
        elif y_force == "Phân loại (categorical)":
            df_model[y] = df_model[y].astype("string")

        model_formula, model_data_used, model_kind = build_formula(df_model, y, x, y_binary_event=y_event)

        with st.expander("Xem công thức mô hình (formula)"):
            st.code(model_formula)
            st.caption(f"Số dòng dùng cho mô hình (sau khi loại NA): {model_data_used.shape[0]}")

    run = st.button("▶️ Chạy kiểm định / mô hình", type="primary", use_container_width=True)


# =========================
# Results area
# =========================
st.divider()
res_left, res_right = st.columns([1.35, 1.0], gap="large")

with res_left:
    st.subheader("📌 Kết quả")
    if not run:
        st.info("Nhấn **Chạy kiểm định / mô hình** để xem kết quả.")
    else:
        try:
            if analysis_mode == "test":
                x1 = x[0]
                result_df, interp = run_single_x_test(df, y, x1, test_kind=test_kind)
                st.dataframe(result_df, use_container_width=True)
                st.write("🔎 **Gợi ý diễn giải:**")
                st.write(interp)
            else:
                fit, note = run_model(model_formula, model_data_used, model_kind)
                kind = model_kind.split("||", 1)[0]
                st.caption(note)

                if kind == "ols":
                    out = ols_table(fit)
                    st.dataframe(out, use_container_width=True)
                    st.write(
                        "🔎 **Gợi ý diễn giải:**\n"
                        "- Hệ số > 0: Y tăng khi X tăng (giữ các biến khác không đổi).\n"
                        "- p-value < 0.05: liên quan có ý nghĩa thống kê (tuỳ ngưỡng nghiên cứu).\n"
                        "- CI 95% không chứa 0: thường tương ứng có ý nghĩa."
                    )
                elif kind == "logit":
                    out = logit_or_table(fit)
                    st.dataframe(out, use_container_width=True)
                    st.write(
                        "🔎 **Gợi ý diễn giải:**\n"
                        "- OR > 1: tăng odds xảy ra sự kiện (Y=1).\n"
                        "- OR < 1: giảm odds.\n"
                        "- p-value < 0.05 và CI 95% không chứa 1: thường có ý nghĩa."
                    )
                else:
                    st.write(fit.summary())
                    st.write(
                        "🔎 **Gợi ý diễn giải (Multinomial):**\n"
                        "- Hệ số được ước lượng theo **nhóm tham chiếu**.\n"
                        "- Nếu muốn RRR = exp(coef) theo từng nhóm, có thể bổ sung bảng riêng."
                    )
        except Exception as e:
            st.error(f"Lỗi khi chạy: {e}")
            st.info("Mẹo: kiểm tra dữ liệu (NA), biến phân loại quá nhiều mức, hoặc cỡ mẫu quá nhỏ.")

with res_right:
    st.subheader("📈 Biểu đồ minh hoạ")
    if not run:
        st.info("Chạy xong app sẽ vẽ biểu đồ minh hoạ.")
    else:
        try:
            if analysis_mode == "test":
                x1 = x[0]
                yk = var_kind(df[y], y_force)
                xk = var_kind(df[x1], x_force)

                tmp = df[[y, x1]].dropna().copy()
                if tmp.shape[0] < 3:
                    st.info("Không đủ dữ liệu để vẽ biểu đồ.")
                else:
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
                        fig = px.scatter(tmp, x=x1, y=y, trendline="ols", title=f"{y} ~ {x1} (scatter + trendline)")
                        st.plotly_chart(fig, use_container_width=True)

            else:
                kind = model_kind.split("||", 1)[0]
                fit, _ = run_model(model_formula, model_data_used, model_kind)

                if kind == "ols":
                    x1 = x[0]
                    if (not is_categorical(model_data_used[x1])) and (not is_categorical(model_data_used[y])):
                        fig = px.scatter(model_data_used, x=x1, y=y, trendline="ols", title=f"{y} ~ {x1} (kèm trendline)")
                    else:
                        if is_categorical(model_data_used[x1]):
                            fig = px.box(model_data_used, x=x1, y=y, points="all", title=f"{y} theo nhóm {x1}")
                        else:
                            fig = px.scatter(model_data_used, x=x1, y=y, title=f"{y} theo {x1}")
                    st.plotly_chart(fig, use_container_width=True)

                    pred = fit.fittedvalues
                    tmp_plot = pd.DataFrame({"Thực tế": model_data_used[y], "Dự đoán": pred})
                    fig2 = px.scatter(tmp_plot, x="Thực tế", y="Dự đoán", title="Dự đoán vs Thực tế")
                    st.plotly_chart(fig2, use_container_width=True)

                elif kind == "logit":
                    p = fit.predict()
                    fig = px.histogram(p, nbins=25, title="Phân bố xác suất dự đoán (p)")
                    st.plotly_chart(fig, use_container_width=True)

                    y_true = model_data_used["_y01_"].astype(int)
                    y_pred = (p >= 0.5).astype(int)
                    tp = int(((y_true == 1) & (y_pred == 1)).sum())
                    tn = int(((y_true == 0) & (y_pred == 0)).sum())
                    fp = int(((y_true == 0) & (y_pred == 1)).sum())
                    fn = int(((y_true == 1) & (y_pred == 0)).sum())
                    st.write("**Bảng nhầm lẫn (ngưỡng 0.5):**")
                    st.table(
                        pd.DataFrame(
                            {"Dự đoán 0": [tn, fn], "Dự đoán 1": [fp, tp]},
                            index=["Thực tế 0", "Thực tế 1"],
                        )
                    )

                else:
                    st.info("Multinomial: biểu đồ minh hoạ có thể bổ sung (RRR, xác suất dự đoán).")

        except Exception as e:
            st.warning(f"Không vẽ được biểu đồ: {e}")

st.divider()
st.caption(
    "⚠️ Lưu ý: Công cụ hỗ trợ gợi ý và chạy kiểm định/mô hình cơ bản. "
    "Người dùng cần kiểm tra giả định, thiết kế nghiên cứu và cách mã hoá biến để diễn giải đúng."
)
