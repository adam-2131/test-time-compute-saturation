from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from io_utils import (
    discover_bootstrap_ci_files,
    discover_extraction_audit_files,
    get_repo_paths,
    load_bootstrap_ci_json,
    load_extraction_audit_csv,
    load_scaling_results_csv,
    safe_new_results_path,
)


st.set_page_config(page_title="Results & Audit Dashboard", layout="wide")

paths = get_repo_paths()

st.title("Results & Audit Dashboard (read-only)")
st.caption("Loads local artifacts from `./results` (and optionally `./data`). No model/API calls are made.")

tab_a, tab_b, tab_fig, tab_c = st.tabs(["Run Summary", "Bootstrap CI Viewer", "Figures", "Extraction Audit Viewer"])


def _as_str(x) -> str:
    return "" if x is None else str(x)


# -----------------------------
# Shared scaling results loader
# -----------------------------

st.sidebar.header("Scaling results source")

# By default, show the canonical/curated file, but allow inspection of raw/clean artifacts too.
available_sources = []
for name in ["scaling_results.csv", "scaling_results_clean.csv", "scaling_results_raw.csv"]:
    p = paths.results_dir / name
    if p.exists():
        available_sources.append(name)

if not available_sources:
    st.error(f"No scaling results CSVs found under `{paths.results_dir}`.")
    st.stop()

source_name = st.sidebar.selectbox(
    "Source file",
    options=available_sources,
    index=available_sources.index("scaling_results.csv") if "scaling_results.csv" in available_sources else 0,
    help="`scaling_results.csv` is the canonical output; `*_raw/clean` are optional helper artifacts.",
)
scaling_path = paths.results_dir / source_name

try:
    df = load_scaling_results_csv(scaling_path)
except Exception as e:
    st.exception(e)
    st.stop()

if df.empty:
    st.warning(
        f"`{scaling_path.name}` has 0 rows. If you only ran toy/sanity checks, those may only appear in "
        "`scaling_results_raw.csv`. For real runs, execute `python main.py ...` (with a real dataset / subset) "
        "or rebuild the curated file via `python scripts/rebuild_scaling_results.py`."
    )


def _sanitize_filename_part(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "all"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    # Collapse repeats
    cleaned = "".join(out)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "all"


def _df_filtered(
    df_in: pd.DataFrame,
    *,
    dataset: str | None,
    method: str | None,
    run_id: str | None,
    seed: str | None,
) -> pd.DataFrame:
    filt = df_in.copy()
    if dataset and dataset != "All":
        filt = filt[filt["dataset"].astype(str) == dataset]
    if method and method != "All":
        filt = filt[filt["method"].astype(str) == method]
    if run_id and run_id != "All":
        filt = filt[filt["run_id"].astype(str) == run_id]
    if seed and seed != "All":
        filt = filt[filt["seed"].astype(str) == seed]
    return filt


def _plot_lines(
    df_in: pd.DataFrame,
    *,
    x_col: str,
    y_cols: list[str],
    title: str,
    overlay_seeds: bool,
    ylabel: str,
) -> plt.Figure:
    """
    Create a simple line plot. If overlay_seeds, plot each seed as its own series;
    otherwise, aggregate across seeds with mean at each multiplier.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    d = df_in.copy()
    d = d.dropna(subset=[x_col])
    if d.empty:
        ax.text(0.5, 0.5, "No data for current filters.", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return fig

    # Robustly extract multipliers from pandas Int64/float/str.
    xs = []
    for v in d[x_col].dropna().tolist():
        try:
            xs.append(int(v))
        except Exception:
            continue
    xs = sorted(set(xs))
    if xs and all(x > 0 for x in xs):
        ax.set_xscale("log")
    ax.set_xlabel("Compute multiplier (× baseline budget)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    present_y_cols = [c for c in y_cols if c in d.columns and d[c].notna().any()]
    if not present_y_cols:
        ax.text(0.5, 0.5, "Requested metric columns not available for this selection.", transform=ax.transAxes, ha="center", va="center")
        return fig

    if overlay_seeds and "seed" in d.columns and d["seed"].notna().any():
        for seed_val, g in d.groupby("seed", dropna=True):
            g2 = g.sort_values(by=[x_col], ascending=True, na_position="last")
            for c in present_y_cols:
                gg = g2.dropna(subset=[c])
                if gg.empty:
                    continue
                ax.plot(pd.to_numeric(gg[x_col], errors="coerce"), pd.to_numeric(gg[c], errors="coerce"), label=f"seed={seed_val} · {c}")
    else:
        g2 = d.groupby(x_col, dropna=True)[present_y_cols].mean(numeric_only=True).reset_index()
        g2 = g2.sort_values(by=[x_col], ascending=True, na_position="last")
        for c in present_y_cols:
            gg = g2.dropna(subset=[c])
            if gg.empty:
                continue
            ax.plot(pd.to_numeric(gg[x_col], errors="coerce"), pd.to_numeric(gg[c], errors="coerce"), label=c)

    ax.legend(fontsize=8)
    ax.grid(True, which="both")
    return fig


def _plot_paired_diff_ci(
    ci: dict,
    *,
    dataset: str | None,
    method: str | None,
    seed: str | None,
    prefer_key: str = "100-10",
) -> tuple[plt.Figure, bool]:
    """
    Plot paired diff CI from bootstrap_ci JSON.
    Returns: (fig, shown)
    """
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    groups = ci.get("groups", [])
    if not isinstance(groups, list) or not groups:
        ax.text(0.5, 0.5, "No groups in bootstrap_ci file.", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return fig, False

    # Filter to selected dataset/method/seed if provided.
    rows = []
    for g in groups:
        ds = _as_str(g.get("dataset"))
        md = _as_str(g.get("method"))
        sd = _as_str(g.get("seed"))
        if dataset and dataset != "All" and ds != dataset:
            continue
        if method and method != "All" and md != method:
            continue
        if seed and seed != "All" and sd != seed:
            continue
        diffs = g.get("diffs", {}) or {}
        if not isinstance(diffs, dict) or not diffs:
            continue
        diff_key = prefer_key if prefer_key in diffs else sorted(diffs.keys())[0]
        d = diffs.get(diff_key, {}) or {}
        ci95 = d.get("ci95", None)
        if not (isinstance(ci95, list) and len(ci95) >= 2):
            continue
        try:
            point = float(d.get("point"))
            lo = float(ci95[0])
            hi = float(ci95[1])
        except Exception:
            continue
        rows.append({"label": f"{ds} | {md} | seed={sd}", "diff_key": diff_key, "point": point, "lo": lo, "hi": hi})

    if not rows:
        ax.text(0.5, 0.5, "No paired diff CI found for current selection.", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return fig, False

    # Plot as horizontal CI bars (one per row).
    y = list(range(len(rows)))
    points = [r["point"] for r in rows]
    xerr = [[r["point"] - r["lo"] for r in rows], [r["hi"] - r["point"] for r in rows]]
    ax.errorbar(points, y, xerr=xerr, fmt="o")
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel(f"Paired diff ({rows[0]['diff_key']}) with 95% CI (bootstrap)")
    ax.grid(True, axis="x")

    # Add a small textual note about significance (CI includes 0) without custom colors.
    # Small inline label for CI containment of 0 (avoid custom colors).
    x_min, x_max = ax.get_xlim()
    x_text = x_max - 0.02 * (x_max - x_min)
    for i, r in enumerate(rows):
        includes_0 = (r["lo"] <= 0.0 <= r["hi"])
        ax.text(x_text, i, "includes 0" if includes_0 else "excludes 0", va="center", ha="right", fontsize=8)

    fig.tight_layout()
    return fig, True


with tab_a:
    st.subheader("Run Summary")

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    datasets = sorted([d for d in df["dataset"].dropna().unique().tolist() if str(d).strip()])
    methods = sorted([m for m in df["method"].dropna().unique().tolist() if str(m).strip()])
    run_ids = sorted([r for r in df["run_id"].dropna().unique().tolist() if str(r).strip()], reverse=True)
    seeds = sorted([s for s in df["seed"].dropna().unique().tolist() if _as_str(s).strip() != ""])

    with c1:
        dataset_sel = st.selectbox("dataset", options=["All"] + datasets, index=0)
    with c2:
        method_sel = st.selectbox("method", options=["All"] + methods, index=0)
    with c3:
        run_id_sel = st.selectbox("run_id", options=["All"] + run_ids, index=0)
    with c4:
        seed_sel = st.selectbox("seed (from notes)", options=["All"] + [str(s) for s in seeds], index=0)

    filt = df.copy()
    if dataset_sel != "All":
        filt = filt[filt["dataset"].astype(str) == dataset_sel]
    if method_sel != "All":
        filt = filt[filt["method"].astype(str) == method_sel]
    if run_id_sel != "All":
        filt = filt[filt["run_id"].astype(str) == run_id_sel]
    if seed_sel != "All":
        filt = filt[filt["seed"].astype(str) == seed_sel]

    # Table
    cols = [
        "compute_multiplier",
        "accuracy",
        "pass_at_k_accuracy",
        "majority_vote_accuracy",
        "total_decoded_tokens",
        "num_problems",
        "cap_hit_rate",
    ]
    out = filt[cols].copy()
    out = out.sort_values(by=["compute_multiplier"], ascending=True, na_position="last")
    st.dataframe(out, use_container_width=True, hide_index=True)

    st.caption("Note: accuracy may be method-dependent; pass@k and majority are logged separately.")


with tab_b:
    st.subheader("Bootstrap CI Viewer")

    ci_files = discover_bootstrap_ci_files(paths.results_dir)
    if not ci_files:
        st.info("No `bootstrap_ci_*.json` files found in `results/`. Run `python scripts/bootstrap_ci.py` first.")
    else:
        file_labels = [p.name for p in ci_files]
        selected = st.selectbox("bootstrap_ci file", options=file_labels, index=0)
        selected_path = paths.results_dir / selected

        try:
            ci = load_bootstrap_ci_json(selected_path)
        except Exception as e:
            st.exception(e)
        else:
            st.caption(f"Loaded: `{selected_path}`")

            groups = ci.get("groups", [])
            if not isinstance(groups, list) or not groups:
                st.warning("File has no groups to display.")
            else:
                for g in groups:
                    dataset = _as_str(g.get("dataset"))
                    method = _as_str(g.get("method"))
                    seed = _as_str(g.get("seed"))
                    num_problems = _as_str(g.get("num_problems"))

                    st.markdown(f"**dataset={dataset} | method={method} | seed={seed} | n={num_problems}**")

                    mults = g.get("multipliers", {}) or {}
                    diffs = g.get("diffs", {}) or {}

                    rows: list[dict[str, object]] = []
                    for m_str, v in sorted(mults.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0]):
                        ci95 = v.get("ci95", ["", ""])
                        rows.append(
                            {
                                "type": "multiplier",
                                "key": m_str,
                                "point": v.get("point", ""),
                                "lo": (ci95[0] if isinstance(ci95, list) and len(ci95) >= 2 else ""),
                                "hi": (ci95[1] if isinstance(ci95, list) and len(ci95) >= 2 else ""),
                                "ci_includes_0": "",
                            }
                        )

                    for diff_key, v in diffs.items():
                        ci95 = v.get("ci95", ["", ""])
                        lo = (ci95[0] if isinstance(ci95, list) and len(ci95) >= 2 else None)
                        hi = (ci95[1] if isinstance(ci95, list) and len(ci95) >= 2 else None)
                        includes_0 = ""
                        try:
                            if lo is not None and hi is not None:
                                includes_0 = bool(float(lo) <= 0.0 <= float(hi))
                        except Exception:
                            includes_0 = ""
                        rows.append(
                            {
                                "type": "paired_diff",
                                "key": diff_key,
                                "point": v.get("point", ""),
                                "lo": lo if lo is not None else "",
                                "hi": hi if hi is not None else "",
                                "ci_includes_0": includes_0,
                            }
                        )

                    df_ci = pd.DataFrame(rows)

                    def _style_row(r: pd.Series):
                        if r.get("type") == "paired_diff" and r.get("ci_includes_0") is True:
                            return ["background-color: #fff3cd"] * len(r)
                        if r.get("type") == "paired_diff" and r.get("ci_includes_0") is False:
                            return ["background-color: #d1e7dd"] * len(r)
                        return [""] * len(r)

                    st.dataframe(df_ci.style.apply(_style_row, axis=1), use_container_width=True, hide_index=True)

                    st.divider()


with tab_fig:
    st.subheader("Figures")
    st.caption("Paper-ready matplotlib figures from `results/scaling_results*.csv` and optional `results/bootstrap_ci_*.json` (read-only).")

    if df.empty:
        st.info("No scaling rows available to plot.")
    else:
        # Controls
        c1, c2, c3, c4 = st.columns(4)
        datasets = sorted([d for d in df["dataset"].dropna().unique().tolist() if str(d).strip()])
        methods = sorted([m for m in df["method"].dropna().unique().tolist() if str(m).strip()])
        run_ids = sorted([r for r in df["run_id"].dropna().unique().tolist() if str(r).strip()], reverse=True)
        seeds = sorted([s for s in df["seed"].dropna().unique().tolist() if _as_str(s).strip() != ""])

        with c1:
            dataset_sel = st.selectbox("dataset", options=["All"] + datasets, index=0, key="fig_dataset")
        with c2:
            method_sel = st.selectbox("method", options=["All"] + methods, index=0, key="fig_method")
        with c3:
            run_id_sel = st.selectbox("run_id", options=["All"] + run_ids, index=0, key="fig_runid")
        with c4:
            seed_sel = st.selectbox("seed (optional)", options=["All"] + [str(s) for s in seeds], index=0, key="fig_seed")

        overlay_seeds = st.checkbox("Overlay seeds as separate lines (when available)", value=False)
        export_svg = st.checkbox("Also export SVG", value=False)

        filt = _df_filtered(df, dataset=dataset_sel, method=method_sel, run_id=run_id_sel, seed=(None if overlay_seeds else seed_sel))
        if filt.empty:
            st.warning("No rows match the current selection.")
        else:
            # A) Accuracy vs multiplier
            fig_acc = _plot_lines(
                filt,
                x_col="compute_multiplier",
                y_cols=["accuracy", "pass_at_k_accuracy", "majority_vote_accuracy"],
                title="Accuracy vs Compute Multiplier",
                overlay_seeds=overlay_seeds,
                ylabel="Accuracy",
            )
            st.pyplot(fig_acc, use_container_width=True)

            # B) Tokens vs multiplier
            token_y = "total_decoded_tokens"
            if token_y not in filt.columns or not filt[token_y].notna().any():
                if "total_completion_tokens_used" in filt.columns and filt["total_completion_tokens_used"].notna().any():
                    token_y = "total_completion_tokens_used"

            fig_tok = _plot_lines(
                filt,
                x_col="compute_multiplier",
                y_cols=[token_y],
                title="Total Decoded Tokens vs Multiplier",
                overlay_seeds=overlay_seeds,
                ylabel=token_y,
            )
            st.pyplot(fig_tok, use_container_width=True)

            # C) Cap-hit rate vs multiplier
            if "cap_hit_rate" not in filt.columns or not filt["cap_hit_rate"].notna().any():
                st.info("Cap-hit rate not available for this run/selection (missing `cap_hit_rate=` in notes).")
                fig_cap = None
            else:
                fig_cap = _plot_lines(
                    filt,
                    x_col="compute_multiplier",
                    y_cols=["cap_hit_rate"],
                    title="Cap-hit Rate vs Multiplier",
                    overlay_seeds=overlay_seeds,
                    ylabel="cap_hit_rate",
                )
                st.pyplot(fig_cap, use_container_width=True)

            # D) Bootstrap CI paired diff plot (optional)
            ci_files = discover_bootstrap_ci_files(paths.results_dir)
            fig_ci = None
            ci_shown = False
            if not ci_files:
                st.info("No `bootstrap_ci_*.json` files found in `results/` (diff CI figure not available).")
            else:
                file_labels = [p.name for p in ci_files]
                selected = st.selectbox("bootstrap_ci file (optional)", options=["(none)"] + file_labels, index=0)
                if selected != "(none)":
                    selected_path = paths.results_dir / selected
                    try:
                        ci = load_bootstrap_ci_json(selected_path)
                    except Exception as e:
                        st.exception(e)
                    else:
                        fig_ci, ci_shown = _plot_paired_diff_ci(
                            ci, dataset=dataset_sel, method=method_sel, seed=(seed_sel if not overlay_seeds else "All")
                        )
                        st.pyplot(fig_ci, use_container_width=True)

            # Export current figures
            if st.button("Export current figures"):
                out_dir = paths.results_dir / "figures"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                ds = _sanitize_filename_part(dataset_sel)
                md = _sanitize_filename_part(method_sel)
                rid = _sanitize_filename_part(run_id_sel)

                def _save(fig: plt.Figure, suffix: str) -> None:
                    base = f"{ts}_{ds}_{md}_{rid}_{suffix}"
                    png_path = safe_new_results_path(out_dir, base + ".png")
                    fig.savefig(png_path, dpi=220, bbox_inches="tight")
                    if export_svg:
                        svg_path = safe_new_results_path(out_dir, base + ".svg")
                        fig.savefig(svg_path, bbox_inches="tight")

                _save(fig_acc, "accuracy")
                _save(fig_tok, "tokens")
                if fig_cap is not None:
                    _save(fig_cap, "caphit")
                if fig_ci is not None and ci_shown:
                    _save(fig_ci, "diffci")

                st.success(f"Exported figures to `{out_dir}`")


with tab_c:
    st.subheader("Extraction Audit Viewer")

    audit_files = discover_extraction_audit_files(paths.results_dir)
    if not audit_files:
        st.info("No `extraction_audit_*.csv` files found in `results/`. Run `python scripts/audit_extraction.py --n 50` first.")
    else:
        audit_labels = [p.name for p in audit_files]
        selected_audit = st.selectbox("extraction_audit file", options=audit_labels, index=0)
        audit_path = paths.results_dir / selected_audit

        try:
            audit_df = load_extraction_audit_csv(audit_path)
        except Exception as e:
            st.exception(e)
        else:
            # Construct a stable row id (prefer cache_path since it's unique).
            def _row_uid(row: pd.Series, i: int) -> str:
                cache_path = str(row.get("cache_path", "") or "").strip()
                if cache_path:
                    return cache_path
                problem_id = str(row.get("problem_id", "") or "").strip()
                rnd = str(row.get("round", "") or "").strip()
                draw = str(row.get("draw_index", "") or "").strip()
                return f"{problem_id}:{rnd}:{draw}:{i}"

            uids = [_row_uid(audit_df.iloc[i], i) for i in range(len(audit_df))]
            audit_df = audit_df.copy()
            audit_df["_uid"] = uids

            if "audit_labels" not in st.session_state:
                st.session_state["audit_labels"] = {}  # uid -> "True"/"False"/"Unknown"

            label_map: dict[str, str] = st.session_state["audit_labels"]

            key_cols = [
                "problem_id",
                "gold_answer",
                "extracted_answer",
                "round",
                "draw_index",
                "completion_tokens",
                "cache_path",
            ]
            display = audit_df[key_cols + ["_uid"]].copy()
            display["manual_label"] = display["_uid"].map(lambda uid: label_map.get(uid, "Unknown"))
            st.dataframe(display.drop(columns=["_uid"]), use_container_width=True, hide_index=True)

            # Row selector + raw output
            options = [
                f"{i:04d} | pid={_as_str(audit_df.iloc[i].get('problem_id'))} | round={_as_str(audit_df.iloc[i].get('round'))} | draw={_as_str(audit_df.iloc[i].get('draw_index'))}"
                for i in range(len(audit_df))
            ]
            sel = st.selectbox("Select a row", options=options, index=0)
            sel_idx = int(sel.split("|", 1)[0].strip())
            sel_row = audit_df.iloc[sel_idx]
            sel_uid = str(sel_row["_uid"])

            c1, c2, c3 = st.columns([2, 2, 3])
            with c1:
                st.write("**gold_answer**")
                st.code(_as_str(sel_row.get("gold_answer")), language=None)
            with c2:
                st.write("**extracted_answer**")
                st.code(_as_str(sel_row.get("extracted_answer")), language=None)
            with c3:
                st.write("**manual_label (session-only)**")
                current = label_map.get(sel_uid, "Unknown")
                new_label = st.radio(
                    "Set label for selected row",
                    options=["Unknown", "True", "False"],
                    index=["Unknown", "True", "False"].index(current),
                    horizontal=True,
                )
                if new_label != current:
                    label_map[sel_uid] = new_label
                    st.session_state["audit_labels"] = label_map

            with st.expander("raw_output"):
                st.text(_as_str(sel_row.get("raw_output")))

            # Export
            labeled_counts = pd.Series(list(label_map.values())).value_counts(dropna=False).to_dict() if label_map else {}
            st.caption(f"Labels set this session: {labeled_counts or {'Unknown': 0}}")

            if st.button("Export labeled copy (new CSV under results/)"):
                out_df = audit_df.drop(columns=["_uid"]).copy()
                out_df["manual_label"] = audit_df["_uid"].map(lambda uid: label_map.get(uid, "Unknown"))

                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                out_name = f"extraction_audit_labeled_{ts}.csv"
                out_path = safe_new_results_path(paths.results_dir, out_name)
                out_df.to_csv(out_path, index=False, encoding="utf-8")
                st.success(f"Wrote `{out_path}`")


