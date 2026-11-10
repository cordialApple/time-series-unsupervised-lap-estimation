from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIGURE_DPI = 200
BLUE, ORANGE, GREEN = sns.color_palette("colorblind", 3).as_hex()
INK = "#2b2b2b"
MUTED = "#8c8c8c"
IDLE = "#dcdcdc"
COHORT_TITLES = {"drive_day_2026": "2026 motor speed", "aim_2023": "2023 yaw rate"}


def _style():
    return sns.axes_style("ticks", {"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK})


def _panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", fontsize=10.5, pad=9)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _recording_timeline(ax, runs):
    recordings = runs.drop_duplicates("recording_id").sort_values("recording_duration_s")
    order = {recording_id: row for row, recording_id in enumerate(recordings["recording_id"])}
    ax.barh(list(order.values()), recordings["recording_duration_s"], color=IDLE, height=0.55)
    ax.barh(runs["recording_id"].map(order), runs["end_s"] - runs["start_s"], left=runs["start_s"], color=BLUE, height=0.55)
    ax.set_yticks(list(order.values()), [recording_id[:6] for recording_id in order])
    ax.set(xlabel="Elapsed time in recording (s)", ylabel="Recording")
    longest = recordings.iloc[-1]
    first_run = runs.loc[runs["recording_id"].eq(longest["recording_id"])].sort_values("start_s").iloc[0]
    top = order[longest["recording_id"]]
    ax.annotate("driving run", (first_run["start_s"] + 4, top + 0.42), color=BLUE, fontsize=9, fontweight="bold", va="bottom")
    ax.annotate("idle", (first_run["end_s"] + 60, top + 0.42), color=MUTED, fontsize=9, fontweight="bold", va="bottom")
    ax.set_ylim(-0.6, len(order) + 0.1)


def _aligned_headings(ax, profiles, run_count):
    sns.lineplot(data=profiles, x="reference_time_s", y="heading_turns", units="run_id", estimator=None, color=BLUE, alpha=0.45, linewidth=1.0, ax=ax)
    coverage = profiles.groupby("reference_time_s")["heading_turns"].agg(["median", "size"])
    median = coverage.loc[coverage["size"] >= run_count / 2, "median"]
    ax.plot(median.index, median, color=INK, linewidth=2.0)
    ax.set(xlabel="Time along reference run (s)", ylabel="Heading (turns)")
    peak = median.idxmax()
    ax.annotate("median", (peak, median.max()), xytext=(6, 8), textcoords="offset points", color=INK, fontsize=9, fontweight="bold")
    ax.annotate(f"{run_count} runs, pace-aligned", (0.02, 0.92), xycoords="axes fraction", color=BLUE, fontsize=9, fontweight="bold")


def _similarity_against_null(ax, pairs, null):
    values = pd.concat(
        [
            pd.DataFrame({"correlation": null["correlation"], "source": f"shuffled run (n={len(null)})"}),
            pd.DataFrame({"correlation": pairs["correlation"], "source": f"run pair (n={len(pairs)})"}),
        ],
        ignore_index=True,
    )
    sns.histplot(data=values, x="correlation", hue="source", palette=[MUTED, BLUE], binwidth=0.04, binrange=(0, 1), stat="probability", common_norm=False, element="step", alpha=0.55, linewidth=1.2, ax=ax)
    sns.move_legend(ax, "upper center", title=None, frameon=False, fontsize=9)
    ax.set(xlabel="Best yaw-rate correlation after pace and offset alignment", ylabel="Share of comparisons", xlim=(0, 1))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)


def _lap_times(ax, runs):
    ordered = runs.sort_values("aligned_lap_time_s")
    median = ordered["aligned_lap_time_s"].median()
    ax.axvline(median, color=MUTED, linewidth=1.0, linestyle="--")
    ax.hlines(ordered["segment_id"], median, ordered["aligned_lap_time_s"], color=IDLE, linewidth=2.0)
    sns.scatterplot(data=ordered, x="aligned_lap_time_s", y="segment_id", color=BLUE, s=70, edgecolor="white", linewidth=1.5, zorder=3, legend=False, ax=ax)
    for _, row in ordered.iterrows():
        outward = -1 if row["aligned_lap_time_s"] < median else 1
        ax.annotate(f"{row['aligned_lap_time_s']:.1f}", (row["aligned_lap_time_s"], row["segment_id"]), xytext=(8 * outward, 0), textcoords="offset points", va="center", ha="right" if outward < 0 else "left", fontsize=8.5, color=INK)
    ax.annotate(f"median {median:.1f} s", (median, -0.75), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=9, va="center")
    ax.set(xlabel="Lap time over reference span (s)", ylabel="Run")
    ax.set_xlim(ordered["aligned_lap_time_s"].min() - 4, ordered["aligned_lap_time_s"].max() + 7)
    ax.set_ylim(len(ordered) - 0.4, -1.2)


def render_course_runs(method_root):
    method_root = Path(method_root)
    runs = pd.read_csv(method_root / "predictions.csv")
    pairs = pd.read_csv(method_root / "pairs.csv")
    null = pd.read_csv(method_root / "null.csv")
    profiles = pd.read_csv(method_root / "aligned_profiles.csv")
    with _style(), sns.plotting_context("paper", font_scale=1.1):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.25]})
        _recording_timeline(axes[0, 0], runs)
        _panel_title(axes[0, 0], "a", f"{len(runs)} driving runs in {runs['recording_id'].nunique()} recordings")
        _aligned_headings(axes[0, 1], profiles, len(runs))
        _panel_title(axes[0, 1], "b", "Runs trace one heading profile: same course")
        _similarity_against_null(axes[1, 0], pairs, null)
        _panel_title(axes[1, 0], "c", "Every run pair beats the shuffled null")
        _lap_times(axes[1, 1], runs)
        _panel_title(axes[1, 1], "d", "Lap time per run")
        sns.despine(fig)
    return _save(fig, method_root / "course_runs.png")


def _calibration_panel(ax, real, benchmark, alpha, show_legend):
    repeating = benchmark["repeating"].astype(bool)
    series = [
        (benchmark.loc[repeating, "p_value"], GREEN, "planted laps"),
        (benchmark.loc[~repeating, "p_value"], ORANGE, "unrelated segments"),
        (real["p_value"], BLUE, "real driving recordings"),
    ]
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle="--", label="calibrated null")
    for values, color, label in series:
        sns.ecdfplot(x=values, color=color, linewidth=2.0, label=f"{label} (n={len(values)})", ax=ax)
    ax.axvline(alpha, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set(xlabel="p-value against block-shuffle null", ylabel="Cumulative share", xlim=(0, 1), ylim=(0, 1.02))
    if show_legend:
        ax.annotate(f"p = {alpha:g}", (alpha, 0.02), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")


def render_null_calibration(artifact_root, output_path):
    artifact_root = Path(artifact_root)
    panels = []
    for cohort_key, title in COHORT_TITLES.items():
        method_root = artifact_root / cohort_key / "10_physics_laps"
        predictions = pd.read_csv(method_root / "predictions.csv")
        benchmark = pd.read_csv(method_root / "synthetic_benchmark.csv")
        driving = predictions.loc[~predictions["control_candidate"].astype(bool)]
        for domain in sorted(driving["domain"].unique()):
            panels.append((f"{title}, by {domain}", driving.loc[driving["domain"].eq(domain)], benchmark.loc[benchmark["domain"].eq(domain)]))
    with _style(), sns.plotting_context("paper", font_scale=1.1):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.2), constrained_layout=True, sharey=True)
        for index, (ax, (title, real, benchmark)) in enumerate(zip(axes, panels)):
            _calibration_panel(ax, real, benchmark, 0.05, index == 0)
            _panel_title(ax, "abcdef"[index], title)
        sns.despine(fig)
    return _save(fig, output_path)
