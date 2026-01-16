import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

st.set_page_config(page_title="Research Test Helper", layout="wide")
st.title("🔬 Research Test Helper (Upload CSV → chọn biến → gợi ý kiểm định)")

# ---------- Helpers ----------
def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def is_categorical(s: pd.Series, cat_unique_threshold: int = 10, cat_unique_ratio: float = 0.05) -> bool:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    if is_numeric(s):
        nunique = s.dropna().nunique()
        n = s.dropna().shape[0]
        if n == 0:
            return False
        if nunique <= cat_unique_threshold:
            return True
        if (nunique / n) <= cat_unique_ratio:
            return True
    return False

def normality_pvalue(x: pd.Series, max_n: int = 5000) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 3:
        return np.nan
    if len(x) > max_n:
        x = x.sample(max_n, random_state=1)
    try:
        _, p = stats.shapiro(x)
        return float(p)
    except Exception:
        return np.nan

def cramer_v(contingency: np.ndarray) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    r, k = contingency.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1) + 1e-12)))

def suggest_test(df: pd.DataFrame, x: str, y: str, x_force: str, y_force: str):
    sX, sY = df[x], df[y]

    def resolve_type(series, forced):
        if forced == "Tự động":
            return "categorical" if is_categorical(series) else "numeric"
        return "numeric" if forced == "Định lượng (numeric)" else "categorical"

    tx = resolve_type(sX, x_force)
    ty = resolve_type(sY, y_force)

    out = {"tx": tx, "ty": ty, "test": None, "rationale": [], "notes": [], "runner": None}

    tmp = df[[x, y]].copy().dropna()
    if tmp.shape[0] < 3:
        out["test"] = "Không đủ dữ liệu (sau khi loại NA)"
        return out, tmp

    # numeric - numeric
    if tx == "numeric" and ty == "numeric":
        tmp2 = tmp.copy()
        tmp2[x] = pd.to_numeric(tmp2[x], errors="coerce")
        tmp2[y] = pd.to_numeric(tmp2[y], errors="coerce")
        tmp2 = tmp2.dropna()
        if tmp2.shape[0] < 3:
            out["test"] = "Không đủ dữ liệu numeric hợp lệ"
            return out, tmp2

        px_norm = normality_pvalue(tmp2[x])
        py_norm = normality_pvalue(tmp2[y])
        out["rationale"].append("X và Y đều là biến định lượng → xét tương quan.")
        out["notes"].append(f"Shapiro p(X)≈{px_norm:.4g} | Shapiro p(Y)≈{py_norm:.4g}")

        if (not np.isnan(px_norm) and px_norm >= 0.05) and (not np.isnan(py_norm) and py_norm >= 0.05):
            out["test"] = "Tương quan Pearson"
            out["runner"] = ("pearson", x, y)
            out["rationale"].append("Cả hai gần chuẩn → gợi ý Pearson.")
        else:
            out["test"] = "Tương quan Spearman"
            out["runner"] = ("spearman", x, y)
            out["rationale"].append("Ít nhất một biến không chuẩn/không chắc → gợi ý Spearman.")
        return out, tmp2

    # categorical - categorical
    if tx == "categorical" and ty == "categorical":
        ct = pd.crosstab(tmp[x], tmp[y])
        out["rationale"].append("Cả hai biến phân loại → xét độc lập (bảng chéo).")

        if ct.shape == (2, 2):
            chi2, p_chi, dof, expected = stats.chi2_contingency(ct.values, correction=False)
            if (expected < 5).any():
                out["test"] = "Fisher exact (2x2)"
                out["runner"] = ("fisher_2x2", x, y)
                out["rationale"].append("Bảng 2x2 và có ô kỳ vọng < 5 → gợi ý Fisher.")
            else:
                out["test"] = "Chi-square độc lập (2x2)"
                out["runner"] = ("chi2", x, y)
                out["rationale"].append("Kỳ vọng đủ lớn → gợi ý Chi-square.")
            return out, tmp

        chi2, p, dof, expected = stats.chi2_contingency(ct.values, correction=False)
        out["test"] = "Chi-square độc lập"
        out["runner"] = ("chi2", x, y)
        if (expected < 5).any():
            out["notes"].append("Có ô kỳ vọng < 5 → cân nhắc gộp nhóm / mô phỏng.")
        return out, tmp

    # numeric - categorical
    if tx == "categorical" and ty == "numeric":
        g, v = x, y
    else:
        g, v = y, x

    tmp2 = tmp.copy()
    tmp2[v] = pd.to_numeric(tmp2[v], errors="coerce")
    tmp2 = tmp2.dropna()
    tmp2[g] = tmp2[g].astype("category")

    k = tmp2[g].nunique()
    out["rationale"].append("Một biến phân loại + một biến định lượng → so sánh giữa các nhóm.")
    if k < 2:
        out["test"] = "Biến nhóm chỉ có 1 mức"
        return out, tmp2

    cats = tmp2[g].cat.categories
    vecs = [tmp2.loc[tmp2[g] == c, v].dropna() for c in cats]
    ns = [len(a) for a in vecs]
    out["notes"].append("Cỡ mẫu theo nhóm: " + ", ".join([f"{c} (n={n})" for c, n in zip(cats, ns)]))

    norm_ps = [normality_pvalue(a) for a in vecs]
    any_valid = any(not np.isnan(p) for p in norm_ps)
    all_normal = any_valid and all((np.isnan(p) or p >= 0.05) for p in norm_ps)

    if k == 2:
        out["rationale"].append("2 nhóm → t-test hoặc Mann–Whitney.")
        if all_normal:
            out["test"] = "Welch t-test (mặc định an toàn)"
            out["runner"] = ("welch_t", g, v)
        else:
            out["test"] = "Mann–Whitney U"
            out["runner"] = ("mannwhitney", g, v)
        return out, tmp2

    # k >= 3
    out["rationale"].append(f"{k} nhóm → ANOVA hoặc Kruskal–Wallis.")
    if all_normal:
        out["test"] = "ANOVA một yếu tố"
        out["runner"] = ("anova", g, v)
    else:
        out["test"] = "Kruskal–Wallis"
        out["runner"] = ("kruskal", g, v)
    return out, tmp2

# ---------- UI ----------
uploaded = st.file_uploader("Tải lên file CSV", type=["csv"])
if uploaded is None:
    st.info("Chọn một file CSV để bắt đầu.")
    st.stop()

# đọc CSV (fallback encoding)
try:
    df = pd.read_csv(uploaded)
except UnicodeDecodeError:
    df = pd.read_csv(uploaded, encoding="latin1")

st.subheader("Xem trước dữ liệu")
st.dataframe(df.head(50), use_container_width=True)

cols = df.columns.tolist()
st.sidebar.header("Chọn biến")
x = st.sidebar.selectbox("Biến X", cols, index=0)
y = st.sidebar.selectbox("Biến Y", cols, index=1 if len(cols) > 1 else 0)

type_options = ["Tự động", "Định lượng (numeric)", "Phân loại (categorical)"]
st.sidebar.markdown("**Ép kiểu nếu cần**")
x_force = st.sidebar.selectbox("Kiểu X", type_options, index=0)
y_force = st.sidebar.selectbox("Kiểu Y", type_options, index=0)

if x == y:
    st.warning("Hãy chọn 2 biến khác nhau.")
    st.stop()

suggestion, data_used = suggest_test(df, x, y, x_force, y_force)

st.subheader("Gợi ý kiểm định")
st.write(f"**X:** `{x}` → **{suggestion['tx']}**")
st.write(f"**Y:** `{y}` → **{suggestion['ty']}**")
st.success(f"✅ **Kiểm định gợi ý:** {suggestion['test']}")
st.caption(f"Số dòng dùng phân tích (sau khi loại NA theo X,Y): {data_used.shape[0]}")

with st.expander("Giải thích gợi ý"):
    for r in suggestion["rationale"]:
        st.write("- " + r)
    for n in suggestion["notes"]:
        st.write("- " + n)

runner = suggestion.get("runner")
if runner is None:
    st.stop()

st.subheader("Kết quả & biểu đồ")

kind = runner[0]

if kind in ("pearson", "spearman"):
    fig = px.scatter(data_used, x=runner[1], y=runner[2], trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    if kind == "pearson":
        r, p = stats.pearsonr(data_used[runner[1]], data_used[runner[2]])
        st.write(f"**Pearson r = {r:.4f}**, **p = {p:.4g}**")
    else:
        r, p = stats.spearmanr(data_used[runner[1]], data_used[runner[2]])
        st.write(f"**Spearman ρ = {r:.4f}**, **p = {p:.4g}**")

elif kind in ("chi2", "fisher_2x2"):
    a, b = runner[1], runner[2]
    ct = pd.crosstab(data_used[a], data_used[b])
    st.write("**Bảng chéo:**")
    st.dataframe(ct, use_container_width=True)

    if kind == "fisher_2x2":
        odds, p = stats.fisher_exact(ct.values)
        st.write(f"**Fisher exact**: OR = {odds:.4f}, **p = {p:.4g}**")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(ct.values, correction=False)
        v = cramer_v(ct.values)
        st.write(f"**Chi-square**: χ² = {chi2:.4f}, dof = {dof}, **p = {p:.4g}**")
        st.write(f"**Cramer's V ≈ {v:.4f}**")

elif kind in ("welch_t", "mannwhitney", "anova", "kruskal"):
    g, v = runner[1], runner[2]
    data_used[g] = data_used[g].astype("category")
    fig = px.box(data_used, x=g, y=v, points="all")
    st.plotly_chart(fig, use_container_width=True)

    cats = data_used[g].cat.categories
    vecs = [data_used.loc[data_used[g] == c, v].dropna() for c in cats]

    if kind == "welch_t":
        t, p = stats.ttest_ind(vecs[0], vecs[1], equal_var=False, nan_policy="omit")
        st.write(f"**Welch t-test**: t = {t:.4f}, **p = {p:.4g}**")
    elif kind == "mannwhitney":
        u, p = stats.mannwhitneyu(vecs[0], vecs[1], alternative="two-sided")
        st.write(f"**Mann–Whitney U**: U = {u:.4f}, **p = {p:.4g}**")
    elif kind == "anova":
        f, p = stats.f_oneway(*vecs)
        st.write(f"**ANOVA**: F = {f:.4f}, **p = {p:.4g}**")
    elif kind == "kruskal":
        h, p = stats.kruskal(*vecs)
        st.write(f"**Kruskal–Wallis**: H = {h:.4f}, **p = {p:.4g}**")

st.subheader("Tải dữ liệu đã dùng (lọc NA theo X,Y)")
st.download_button(
    "⬇️ Download data_used.csv",
    data=data_used.to_csv(index=False).encode("utf-8"),
    file_name="data_used.csv",
    mime="text/csv",
)
import statsmodels.api as sm
import statsmodels.formula.api as smf

st.divider()
st.header("📌 Phân tích đa biến (nhiều biến độc lập)")

all_cols = df.columns.tolist()

y_multi = st.sidebar.selectbox("Biến phụ thuộc (Y)", all_cols, key="y_multi")
x_multi = st.sidebar.multiselect("Biến độc lập (X1, X2, ...)", [c for c in all_cols if c != y_multi], key="x_multi")

type_options2 = ["Tự động", "Định lượng (numeric)", "Phân loại (categorical)"]
st.sidebar.markdown("**Ép kiểu nếu cần (đa biến)**")
y_force2 = st.sidebar.selectbox("Kiểu Y (đa biến)", type_options2, index=0, key="y_force2")

if len(x_multi) == 0:
    st.info("Chọn ít nhất 1 biến độc lập ở thanh bên trái để chạy mô hình.")
    st.stop()

# --- helper: auto detect type ---
def auto_type(series: pd.Series, forced: str):
    if forced == "Định lượng (numeric)":
        return "numeric"
    if forced == "Phân loại (categorical)":
        return "categorical"
    # auto
    return "categorical" if is_categorical(series) else "numeric"

y_type = auto_type(df[y_multi], y_force2)

# Build a temp dataframe with selected columns only + drop NA
use_cols = [y_multi] + x_multi
tmp = df[use_cols].copy()

# Convert Y numeric if needed
if y_type == "numeric":
    tmp[y_multi] = pd.to_numeric(tmp[y_multi], errors="coerce")

tmp = tmp.dropna()
st.caption(f"Số dòng dùng cho mô hình (sau khi loại NA theo {len(use_cols)} biến): {tmp.shape[0]}")

# Decide model based on Y
model_name = None

if y_type == "numeric":
    model_name = "Hồi quy tuyến tính (OLS)"
else:
    # treat Y as categorical
    y_nunique = tmp[y_multi].nunique()
    if y_nunique == 2:
        model_name = "Hồi quy logistic nhị phân (Logit)"
    else:
        model_name = "Hồi quy logistic đa danh (Multinomial - MNLogit)"

st.success(f"✅ Gợi ý mô hình: **{model_name}**")

# Build formula: use C(var) for categorical predictors (auto)
terms = []
for x in x_multi:
    if is_categorical(tmp[x]):
        terms.append(f"C(Q('{x}'))")  # safe for spaces/special chars
    else:
        terms.append(f"Q('{x}')")

y_term = f"Q('{y_multi}')" if y_type == "numeric" else f"C(Q('{y_multi}'))"
formula = f"{y_term} ~ " + " + ".join(terms)

with st.expander("Xem công thức mô hình (formula)"):
    st.code(formula)

run_model = st.button("Chạy mô hình")

if run_model:
    try:
        if model_name.startswith("Hồi quy tuyến tính"):
            fit = smf.ols(formula=formula, data=tmp).fit()
            st.subheader("Kết quả OLS")
            st.write(fit.summary())

        elif "nhị phân" in model_name:
            # For binary logit, we need Y numeric 0/1
            y_cat = tmp[y_multi].astype("category")
            if len(y_cat.cat.categories) != 2:
                st.error("Y không phải nhị phân sau khi xử lý. Hãy kiểm tra dữ liệu.")
                st.stop()
            # Map to 0/1
            mapping = {y_cat.cat.categories[0]: 0, y_cat.cat.categories[1]: 1}
            tmp2 = tmp.copy()
            tmp2["_y01_"] = tmp2[y_multi].map(mapping)

            # rebuild formula with _y01_ numeric
            formula2 = f"_y01_ ~ " + " + ".join(terms)
            fit = smf.logit(formula=formula2, data=tmp2).fit(disp=0)

            st.subheader("Kết quả Logistic (OR)")
            params = fit.params
            conf = fit.conf_int()
            or_table = pd.DataFrame({
                "OR": np.exp(params),
                "CI 2.5%": np.exp(conf[0]),
                "CI 97.5%": np.exp(conf[1]),
                "p-value": fit.pvalues
            }).sort_values("p-value")
            st.dataframe(or_table, use_container_width=True)

        else:
            # Multinomial
            # MNLogit requires numeric codes for Y categories
            tmp2 = tmp.copy()
            tmp2["_ycat_"] = tmp2[y_multi].astype("category")
            tmp2["_ycode_"] = tmp2["_ycat_"].cat.codes

            formula2 = f"_ycode_ ~ " + " + ".join(terms)
            fit = smf.mnlogit(formula=formula2, data=tmp2).fit(disp=0)

            st.subheader("Kết quả Multinomial (hệ số)")
            st.write(fit.summary())

            st.caption("Gợi ý: Multinomial thường diễn giải theo nhóm tham chiếu; nếu bạn muốn bảng RRR (exp(coef)), mình có thể bổ sung.")

    except Exception as e:
        st.error(f"Lỗi khi chạy mô hình: {e}")
        st.info("Mẹo: kiểm tra biến phân loại có quá nhiều mức, dữ liệu bị ký tự lạ, hoặc cỡ mẫu quá nhỏ.")

