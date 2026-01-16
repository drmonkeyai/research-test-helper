import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
# Helpers
# =========================
def _safe_name(name: str) -> str:
    """Safe key name for session_state keys."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip())[:80] or "file"


def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """Try reading CSV with common encodings; fall back gracefully."""
    raw = uploaded_file.getvalue()

    # Try utf-8-sig first (Excel-friendly), then utf-8, then cp1258, then latin1
    encodings = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def is_categorical(s: pd.Series) -> bool:
    """Heuristic: object/category/bool, or low unique count."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        nunique = s.dropna().nunique()
        # if numeric but very few levels, likely categorical (e.g., 0/1/2)
        if nunique <= 10:
            return True
    return False


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Try convert to numeric, keep NaN on errors."""
    return pd.to_numeric(s, errors="coerce")


def summarize_variable(df: pd.DataFrame, col: str) -> Dict[str, str]:
    s = df[col]
    miss = int(s.isna().sum())
    n = int(len(s))
    nunique = int(s.dropna().nunique())

    if is_categorical(s):
        # Top levels
        vc = s.astype("string").value_counts(dropna=True).head(3)
        top = ", ".join([f"{idx} ({val})" for idx, val in vc.items()]) if len(vc) else "-"
        return {
            "Tên biến": col,
            "Đặc tính biến": f"Phân loại | mức={nunique} | thiếu={miss}/{n} | top: {top}",
        }

    # numeric
    x = coerce_numeric(s)
    x_non = x.dropna()
    if len(x_non) == 0:
        return {
            "Tên biến": col,
            "Đặc tính biến": f"Định lượng | thiếu={miss}/{n} | (không đọc được số)",
        }

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


def suggest_model(df: pd.DataFrame, y: str, xs: List[str]) -> Tuple[str, str]:
    """
    Return (suggestion_name, explanation).
    """
    y_s = df[y]
    # Determine Y type
    if is_categorical(y_s):
        n_levels = int(y_s.dropna().nunique())
        if n_levels <= 1:
            return ("Không đủ dữ liệu", "Biến phụ thuộc chỉ có 0–1 mức sau khi loại thiếu. Hãy kiểm tra dữ liệu.")
        if n_levels == 2:
            return (
                "Hồi quy Logistic nhị phân (Binary Logistic)",
                "Y là biến phân loại 2 mức → phù hợp mô hình logistic nhị phân để ước lượng OR và p-value khi có nhiều biến độc lập.",
            )
        return (
            "Hồi quy Logistic đa danh (Multinomial Logistic)",
            f"Y là biến phân loại >2 mức (mức={n_levels}) → phù hợp logistic đa danh (multinomial).",
        )

    # numeric Y
    return (
        "Hồi quy tuyến tính (OLS)",
        "Y là biến định lượng liên tục → phù hợp hồi quy tuyến tính (OLS) để ước lượng hệ số, p-value và khoảng tin cậy.",
    )


def build_formula(df: pd.DataFrame, y: str, xs: List[str], y_binary_event: str | None = None) -> Tuple[str, pd.DataFrame, str]:
    """
    Build formula for statsmodels with safe quoting.
    For logistic binary, we map Y categories to 0/1 using y_binary_event as event=1.
    Returns (formula, data_used, model_kind)
    model_kind in {"ols","logit","mnlogit"}
    """
    tmp = df[[y] + xs].copy()

    # Drop missing across selected vars
    tmp = tmp.dropna()

    # Determine Y type
    if is_categorical(tmp[y]):
        n_levels = int(tmp[y].nunique())

        if n_levels == 2:
            # Map to 0/1 with event
            y_cat = tmp[y].astype("category")
            cats = list(y_cat.cat.categories)

            if y_binary_event is None or y_binary_event not in cats:
                # default: choose 2nd category as event
                event = cats[1]
            else:
                event = y_binary_event

            tmp["_y01_"] = (tmp[y] == event).astype(int)

            terms = []
            for x in xs:
                if is_categorical(tmp[x]):
                    terms.append(f"C(Q('{x}'))")
                else:
                    terms.append(f"Q('{x}')")

            formula = "_y01_ ~ " + " + ".join(terms)
            note = f"Logistic nhị phân: sự kiện (Y=1) = '{event}'"
            return (formula, tmp, "logit" + "||" + note)

        # multinomial
        tmp["_ycat_"] = tmp[y].astype("category")
        tmp["_ycode_"] = tmp["_ycat_"].cat.codes

        terms = []
        for x in xs:
            if is_categorical(tmp[x]):
                terms.append(f"C(Q('{x}'))")
            else:
                terms.append(f"Q('{x}')")
        formula = "_ycode_ ~ " + " + ".join(terms)
        note = "Multinomial: hệ số theo nhóm tham chiếu (mã hoá category)"
        return (formula, tmp, "mnlogit" + "||" + note)

    # OLS
    tmp[y] = coerce_numeric(tmp[y])
    tmp = tmp.dropna()

    terms = []
    for x in xs:
        if is_categorical(tmp[x]):
            terms.append(f"C(Q('{x}'))")
        else:
            terms.append(f"Q('{x}')")

    formula = f"Q('{y}') ~ " + " + ".join(terms)
    return (formula, tmp, "ols||OLS")


def run_model(formula: str, data_used: pd.DataFrame, model_kind: str):
    """
    model_kind: 'ols||...' or 'logit||...' or 'mnlogit||...'
    """
    kind, note = model_kind.split("||", 1)

    if kind == "ols":
        fit = smf.ols(formula=formula, data=data_used).fit()
        return fit, note

    if kind == "logit":
        fit = smf.logit(formula=formula, data=data_used).fit(disp=0)
        return fit, note

    if kind == "mnlogit":
        fit = smf.mnlogit(formula=formula, data=data_used).fit(disp=0)
        return fit, note

    raise ValueError("Unknown model kind")


def ols_table(fit) -> pd.DataFrame:
    conf = fit.conf_int()
    out = pd.DataFrame(
        {
            "Hệ số": fit.params,
            "CI 2.5%": conf[0],
            "CI 97.5%": conf[1],
            "p-value": fit.pvalues,
        }
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
# Session state
# =========================
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}  # name -> df

if "active_name" not in st.session_state:
    st.session_state["active_name"] = None


# =========================
# UI: Header
# =========================
st.markdown(
    f"""
    <div style="padding: 0.25rem 0 0.5rem 0;">
      <h1 style="margin:0;">{APP_TITLE}</h1>
      <div style="color:#6b7280;">Upload CSV → chọn biến → gợi ý kiểm định / mô hình phù hợp</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =========================
# UI: Top row (Overview | Upload | File list)
# =========================
col_left, col_mid, col_right = st.columns([2.2, 1.6, 2.2], gap="large")

with col_mid:
    st.subheader("⬆️ Upload file")
    up = st.file_uploader("Tải lên file CSV", type=["csv"], accept_multiple_files=False)
    if up is not None:
        try:
            df_new = read_csv_safely(up)
            fname = up.name
            # Ensure unique key
            key = fname
            if key in st.session_state["datasets"]:
                base = _safe_name(fname)
                i = 2
                while f"{base}_{i}" in st.session_state["datasets"]:
                    i += 1
                key = f"{base}_{i}"
            st.session_state["datasets"][key] = df_new
            st.session_state["active_name"] = key
            st.success(f"Đã tải: {key} (rows={df_new.shape[0]}, cols={df_new.shape[1]})")
        except Exception as e:
            st.error(f"Không đọc được CSV: {e}")

with col_right:
    st.subheader("📁 Danh sách file đã upload")
    names = list(st.session_state["datasets"].keys())
    if len(names) == 0:
        st.info("Chưa có file nào. Hãy upload CSV ở cột giữa.")
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
                st.session_state["datasets"].pop(chosen, None)
                remaining = list(st.session_state["datasets"].keys())
                st.session_state["active_name"] = remaining[0] if remaining else None
                st.rerun()
        with c2:
            if st.button("🧹 Xóa tất cả", use_container_width=True):
                st.session_state["datasets"] = {}
                st.session_state["active_name"] = None
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
# Main area: Variable table + Choose X/Y
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

    # Quick search
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

    # If Y is categorical and binary, allow choosing event
    y_series = df[y]
    y_is_cat = True if y_force == "Phân loại (categorical)" else (False if y_force == "Định lượng (numeric)" else is_categorical(y_series))

    y_event = None
    if y_is_cat:
        levels = sorted(df[y].dropna().astype(str).unique().tolist())
        if len(levels) == 2:
            y_event = st.selectbox("Chọn mức được coi là 'Sự kiện' (Y=1) cho logistic", options=levels, index=1)

    if len(x) == 0:
        st.info("Chọn ít nhất 1 biến độc lập để phần mềm gợi ý mô hình và chạy kết quả.")
        st.stop()

    # Suggest model & explain
    # Respect forced Y type
    tmp_for_suggest = df.copy()
    if y_force == "Định lượng (numeric)":
        # create a fake numeric series for decision
        tmp_for_suggest[y] = coerce_numeric(tmp_for_suggest[y])
    elif y_force == "Phân loại (categorical)":
        tmp_for_suggest[y] = tmp_for_suggest[y].astype("string")

    suggestion, explanation = suggest_model(tmp_for_suggest, y, x)

    st.divider()
    st.subheader("✅ Phép kiểm / mô hình gợi ý")
    st.write(f"**Gợi ý:** {suggestion}")
    with st.expander("Giải thích tại sao chọn mô hình này"):
        st.write(explanation)
        st.write(
            "- App dựa vào **kiểu biến Y** (định lượng / phân loại 2 mức / phân loại >2 mức).\n"
            "- Với nhiều biến độc lập, mô hình hồi quy giúp **hiệu chỉnh (adjust)** các biến đồng thời.\n"
            "- Dữ liệu dùng cho mô hình sẽ **loại dòng thiếu (NA)** theo các biến đã chọn."
        )

    # Build formula and show
    # Enforce y type by converting if needed
    df_model = df.copy()
    if y_force == "Định lượng (numeric)":
        df_model[y] = coerce_numeric(df_model[y])
    elif y_force == "Phân loại (categorical)":
        df_model[y] = df_model[y].astype("string")

    formula, data_used, model_kind = build_formula(df_model, y, x, y_binary_event=y_event)

    with st.expander("Xem công thức mô hình (formula)"):
        st.code(formula)
        st.caption(f"Số dòng dùng cho mô hình (sau khi loại NA): {data_used.shape[0]}")

    run = st.button("▶️ Chạy mô hình", type="primary", use_container_width=True)


# =========================
# Results area
# =========================
st.divider()
res_left, res_right = st.columns([1.35, 1.0], gap="large")

with res_left:
    st.subheader("📌 Kết quả chạy mô hình")
    if not run:
        st.info("Nhấn **Chạy mô hình** để xem kết quả.")
    else:
        try:
            fit, note = run_model(formula, data_used, model_kind)

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

            else:  # mnlogit
                st.write(fit.summary())
                st.write(
                    "🔎 **Gợi ý diễn giải (Multinomial):**\n"
                    "- Hệ số được ước lượng theo **nhóm tham chiếu**.\n"
                    "- Nếu bạn muốn bảng RRR = exp(coef) theo từng nhóm, mình có thể bổ sung."
                )

        except Exception as e:
            st.error(f"Lỗi khi chạy mô hình: {e}")
            st.info("Mẹo: kiểm tra dữ liệu (NA), biến phân loại quá nhiều mức, hoặc cỡ mẫu quá nhỏ.")


with res_right:
    st.subheader("📈 Biểu đồ minh hoạ")
    if not run:
        st.info("Chạy mô hình xong app sẽ vẽ biểu đồ minh hoạ.")
    else:
        try:
            kind = model_kind.split("||", 1)[0]

            # If only 1 numeric X and numeric Y, show scatter with OLS trendline
            if kind == "ols":
                # choose 1 variable to visualize (priority: first X)
                x1 = x[0]
                if (not is_categorical(data_used[x1])) and (not is_categorical(data_used[y])):
                    fig = px.scatter(data_used, x=x1, y=y, trendline="ols", title=f"{y} ~ {x1} (kèm trendline)")
                else:
                    # if categorical X -> boxplot
                    if is_categorical(data_used[x1]):
                        fig = px.box(data_used, x=x1, y=y, points="all", title=f"{y} theo nhóm {x1}")
                    else:
                        fig = px.scatter(data_used, x=x1, y=y, title=f"{y} theo {x1}")
                st.plotly_chart(fig, use_container_width=True)

                # predicted vs actual
                pred = fit.fittedvalues
                tmp_plot = pd.DataFrame({"Thực tế": data_used[y], "Dự đoán": pred})
                fig2 = px.scatter(tmp_plot, x="Thực tế", y="Dự đoán", title="Dự đoán vs Thực tế")
                st.plotly_chart(fig2, use_container_width=True)

            elif kind == "logit":
                # predicted probability histogram
                p = fit.predict()
                fig = px.histogram(p, nbins=25, title="Phân bố xác suất dự đoán (p)")
                st.plotly_chart(fig, use_container_width=True)

                # Simple confusion at 0.5
                y_true = data_used["_y01_"].astype(int)
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
                st.info("Multinomial: biểu đồ minh hoạ sẽ được bổ sung theo nhu cầu (RRR theo nhóm, xác suất dự đoán).")

        except Exception as e:
            st.warning(f"Không vẽ được biểu đồ: {e}")


# =========================
# Footer
# =========================
st.divider()
st.caption(
    "⚠️ Lưu ý: Công cụ hỗ trợ gợi ý và chạy mô hình cơ bản. "
    "Người dùng cần kiểm tra giả định, thiết kế nghiên cứu và cách mã hoá biến để diễn giải đúng."
)
