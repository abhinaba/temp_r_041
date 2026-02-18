#!/usr/bin/env python3
"""
Generate high-quality Nature/Science magazine style figures comparing
delete vs retrieval operators for ICE-NSR experiments.

Reads from multiple source directories and produces four publication-ready figures.
"""

import json
import glob
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# Matplotlib setup (Agg backend for headless rendering)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# Nature/Science style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Color palette (Nature-style muted)
COLOR_DELETE    = '#4878CF'   # steel blue
COLOR_RETRIEVAL = '#E8553B'   # terra cotta
COLOR_SIG_LINE  = '#888888'   # dashed significance threshold
COLOR_GRID      = '#E0E0E0'   # light gray gridlines

# Extended palette for multi-extractor plots
EXTRACTOR_COLORS = {
    'attention':             '#4878CF',
    'gradient':              '#E8553B',
    'integrated_gradients':  '#6ACC65',
    'lime':                  '#D65F5F',
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Source directories (relative to repo root)
DIR_RETRIEVAL        = os.path.join(BASE_DIR, 'results', 'retrieval')
DIR_ENCODER          = os.path.join(BASE_DIR, 'results', 'encoder')
DIR_ENCODER_ATTN_FIX = os.path.join(BASE_DIR, 'results', 'encoder')
DIR_MULTILINGUAL     = os.path.join(BASE_DIR, 'results', 'multilingual')
DIR_DELETE_BASELINE  = os.path.join(BASE_DIR, 'results', 'legacy_delete_baseline')

# Short model names for display
MODEL_SHORT_NAMES = {
    'gpt2':                                 'GPT-2',
    'LiquidAI/LFM2-2.6B-Exp':              'LFM2-2.6B',
    'meta-llama/Llama-3.2-3B-Instruct':     'Llama-3.2-3B',
    'deepseek-ai/deepseek-llm-7b-chat':     'DeepSeek-7B',
    'mistralai/Mistral-7B-Instruct-v0.3':   'Mistral-7B',
    'meta-llama/Llama-3.1-8B':              'Llama-3.1-8B',
    'Qwen/Qwen2.5-7B-Instruct':            'Qwen2.5-7B',
}

DATASET_DISPLAY = {
    'sst2':   'SST-2',
    'esnli':  'e-SNLI',
    'agnews': 'AG News',
    'imdb':   'IMDB',
}

# Order for consistent plotting
MODEL_ORDER = [
    'GPT-2', 'LFM2-2.6B', 'Llama-3.2-3B', 'DeepSeek-7B',
    'Mistral-7B', 'Llama-3.1-8B', 'Qwen2.5-7B'
]

DATASET_ORDER = ['SST-2', 'e-SNLI', 'AG News', 'IMDB']


# ===========================================================================
# Data loading utilities
# ===========================================================================

def safe_load_json(path):
    """Load JSON robustly, handling files with extra trailing data."""
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        try:
            with open(path) as f:
                text = f.read()
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text)
            return obj
        except Exception:
            return None
    except Exception:
        return None


def shorten_model(full_name):
    """Map full HuggingFace model path to short display name."""
    return MODEL_SHORT_NAMES.get(full_name, full_name.split('/')[-1])


def shorten_dataset(ds):
    """Map dataset key to display name."""
    return DATASET_DISPLAY.get(ds, ds)


def normalize_extractor(ext_key):
    """Strip operator suffixes like '/retrieval' from extractor key."""
    return ext_key.split('/')[0]


# -- Load delete baseline data ------------------------------------------------
def load_delete_data():
    """
    Load delete (NSR) baseline results.
    Returns dict: (short_model, dataset_display, extractor) -> win_rate
    When duplicates exist, keep the latest (highest timestamp in filename).
    """
    records = {}
    files = sorted(glob.glob(os.path.join(DIR_DELETE_BASELINE, 'ice_llm_nsr_*.json')))
    for fpath in files:
        d = safe_load_json(fpath)
        if d is None:
            continue
        cfg = d.get('config', {})
        model = cfg.get('model', '')
        dataset = cfg.get('dataset', '')
        summary = d.get('summary', {})
        fname = os.path.basename(fpath)
        ts = fname.rsplit('.', 1)[0].rsplit('_', 1)[-1] if '_' in fname else ''

        model_short = shorten_model(model)
        ds_display = shorten_dataset(dataset)

        for ext_key, ext_data in summary.items():
            if not isinstance(ext_data, dict):
                continue
            wr = ext_data.get('win_rate')
            if wr is None:
                continue
            ext_clean = normalize_extractor(ext_key)
            key = (model_short, ds_display, ext_clean)
            if key not in records or ts > records[key][0]:
                records[key] = (ts, float(wr))

    return {k: v[1] for k, v in records.items()}


# -- Load retrieval data -------------------------------------------------------
def load_retrieval_data():
    """
    Load LLM retrieval results.
    Returns dict: (short_model, dataset_display, extractor) -> win_rate
    """
    records = {}
    files = sorted(glob.glob(os.path.join(DIR_RETRIEVAL, 'ice_llm_retrieval_*.json')))
    for fpath in files:
        d = safe_load_json(fpath)
        if d is None:
            continue
        cfg = d.get('config', {})
        model = cfg.get('model', '')
        dataset = cfg.get('dataset', '')
        summary = d.get('summary', {})

        fname = os.path.basename(fpath)
        ts = fname.rsplit('.', 1)[0].rsplit('_', 1)[-1] if '_' in fname else ''

        model_short = shorten_model(model)
        ds_display = shorten_dataset(dataset)

        for ext_key, ext_data in summary.items():
            if not isinstance(ext_data, dict):
                continue
            wr = ext_data.get('win_rate')
            if wr is None:
                continue
            ext_clean = normalize_extractor(ext_key)
            key = (model_short, ds_display, ext_clean)
            if key not in records or ts > records[key][0]:
                records[key] = (ts, float(wr))

    return {k: v[1] for k, v in records.items()}


# -- Load encoder data (merge base + attn_fix) --------------------------------
def load_encoder_data():
    """
    Load encoder retrieval results from both encoder_retrieval/ and
    encoder_retrieval_attn_fix/. The attn_fix directory overrides
    attention results from the base directory.
    Returns dict: (dataset_display, extractor) -> win_rate
    """
    records = {}

    for d_dir in [DIR_ENCODER, DIR_ENCODER_ATTN_FIX]:
        files = sorted(glob.glob(os.path.join(d_dir, '*.json')))
        for fpath in files:
            d = safe_load_json(fpath)
            if d is None:
                continue
            cfg = d.get('config', {})
            dataset = cfg.get('dataset', '')
            ds_display = shorten_dataset(dataset)

            results = d.get('results', {})
            for op_key, op_data in results.items():
                if not isinstance(op_data, dict):
                    continue
                for ext_key, ext_data in op_data.items():
                    if not isinstance(ext_data, dict):
                        continue
                    if 'error' in ext_data:
                        continue
                    wr = ext_data.get('win_rate')
                    if wr is None:
                        continue
                    key = (ds_display, ext_key)
                    fname = os.path.basename(fpath)
                    ts = fname.rsplit('.', 1)[0].rsplit('_', 1)[-1]
                    if key not in records or ts > records[key][0]:
                        records[key] = (ts, float(wr))

    return {k: v[1] for k, v in records.items()}


# -- Load multilingual data ----------------------------------------------------
def load_multilingual_data():
    """
    Load multilingual retrieval results.
    Returns dict: (short_model, language, extractor) -> win_rate
    """
    records = {}
    files = sorted(glob.glob(os.path.join(DIR_MULTILINGUAL, '*.json')))
    for fpath in files:
        d = safe_load_json(fpath)
        if d is None:
            continue
        cfg = d.get('config', {})
        model = cfg.get('model', '')
        extractor = cfg.get('extractor', '')
        model_short = shorten_model(model)

        fname = os.path.basename(fpath)
        ts = fname.rsplit('.', 1)[0].rsplit('_', 1)[-1]

        results = d.get('results', {})
        for op_key, op_data in results.items():
            if not isinstance(op_data, dict):
                continue
            for lang, lang_data in op_data.items():
                if not isinstance(lang_data, dict):
                    continue
                wr = lang_data.get('win_rate')
                if wr is None:
                    continue
                key = (model_short, lang, extractor)
                if key not in records or ts > records[key][0]:
                    records[key] = (ts, float(wr))

    return {k: v[1] for k, v in records.items()}


# ===========================================================================
# Figure generation
# ===========================================================================

def save_figure(fig, name):
    """Save figure as both PDF and PNG, stripping identifying metadata."""
    pdf_path = os.path.join(FIGURES_DIR, f'{name}.pdf')
    png_path = os.path.join(FIGURES_DIR, f'{name}.png')
    metadata_clean = {"Software": "", "Author": "", "Creator": ""}
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white',
                metadata={"Creator": "", "Producer": "", "Author": ""})
    fig.savefig(png_path, format='png', bbox_inches='tight', facecolor='white',
                metadata=metadata_clean)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    plt.close(fig)


# -- FIGURE 1: Delete vs Retrieval Win Rate Comparison -------------------------
def figure1_delete_vs_retrieval(delete_data, retrieval_data):
    """
    Grouped bar chart: 7 models x 4 datasets, faceted 2x2.
    Uses llm_attention extractor for comparison (most complete coverage).
    Falls back to llm_gradient if attention unavailable.
    """
    print("\n[Figure 1] Delete vs Retrieval Win Rate Comparison")

    def get_wr(data, model, dataset):
        for ext in ['llm_attention', 'llm_gradient']:
            key = (model, dataset, ext)
            if key in data:
                return data[key]
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    fig.suptitle('Win Rate Comparison: Delete vs. Retrieval Infill Operators',
                 fontsize=13, fontweight='bold', y=0.98)

    bar_width = 0.32
    x = np.arange(len(MODEL_ORDER))

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx // 2][idx % 2]

        del_vals = []
        ret_vals = []
        for model in MODEL_ORDER:
            del_vals.append(get_wr(delete_data, model, dataset))
            ret_vals.append(get_wr(retrieval_data, model, dataset))

        # Plot bars (skip None values)
        for i, (dv, rv) in enumerate(zip(del_vals, ret_vals)):
            if dv is not None:
                bar = ax.bar(x[i] - bar_width/2, dv, bar_width,
                       color=COLOR_DELETE, edgecolor='white', linewidth=0.5,
                       zorder=3)
            if rv is not None:
                bar = ax.bar(x[i] + bar_width/2, rv, bar_width,
                       color=COLOR_RETRIEVAL, edgecolor='white', linewidth=0.5,
                       zorder=3)

        # Significance threshold line
        ax.axhline(y=0.55, color=COLOR_SIG_LINE, linestyle='--',
                   linewidth=1.0, alpha=0.8, zorder=2)

        # Chance level line
        ax.axhline(y=0.50, color='#BBBBBB', linestyle=':',
                   linewidth=0.7, alpha=0.6, zorder=1)

        # Formatting
        ax.set_title(dataset, fontsize=11, fontweight='bold', pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0.15, 1.0)
        ax.set_ylabel('Win Rate' if idx % 2 == 0 else '')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.5, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

    # Shared legend at the bottom
    legend_elements = [
        Patch(facecolor=COLOR_DELETE, edgecolor='white', label='Delete'),
        Patch(facecolor=COLOR_RETRIEVAL, edgecolor='white', label='Retrieval'),
        plt.Line2D([0], [0], color=COLOR_SIG_LINE, linestyle='--',
                   linewidth=1.0, label='Significance (0.55)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, 'fig1_delete_vs_retrieval')


# -- FIGURE 2: Encoder Model Retrieval Results ---------------------------------
def figure2_encoder_results(encoder_data):
    """
    Grouped bar chart: 3 datasets x 4 extractors.
    """
    print("\n[Figure 2] Encoder Model Retrieval Results")

    extractors_order = ['attention', 'gradient', 'integrated_gradients', 'lime']
    extractor_labels = ['Attention', 'Gradient', 'Integ. Grad.', 'LIME']
    datasets = ['SST-2', 'e-SNLI', 'IMDB']

    fig, ax = plt.subplots(figsize=(8, 5))

    n_datasets = len(datasets)
    n_extractors = len(extractors_order)
    bar_width = 0.18
    group_width = n_extractors * bar_width + 0.1
    x = np.arange(n_datasets) * (group_width + 0.3)

    plotted_labels = set()
    for j, ext in enumerate(extractors_order):
        vals = []
        for ds in datasets:
            key = (ds, ext)
            vals.append(encoder_data.get(key, None))

        positions = x + (j - n_extractors/2 + 0.5) * bar_width
        for i, v in enumerate(vals):
            if v is not None:
                lbl = extractor_labels[j] if extractor_labels[j] not in plotted_labels else ''
                if lbl:
                    plotted_labels.add(extractor_labels[j])
                ax.bar(positions[i], v, bar_width,
                       color=EXTRACTOR_COLORS.get(ext, '#999999'),
                       edgecolor='white', linewidth=0.5, zorder=3,
                       label=lbl)

    # Significance threshold
    ax.axhline(y=0.55, color=COLOR_SIG_LINE, linestyle='--',
               linewidth=1.0, alpha=0.8, zorder=2)
    ax.axhline(y=0.50, color='#BBBBBB', linestyle=':',
               linewidth=0.7, alpha=0.6, zorder=1)

    ax.set_title('Encoder Model (BERT) Retrieval Win Rates by Extractor',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('Win Rate', fontsize=11)
    ax.set_ylim(0.35, 0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Add value annotations on bars
    for rect in ax.patches:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, color='#333333')

    ax.legend(frameon=False, loc='upper left', fontsize=9)
    fig.tight_layout()
    save_figure(fig, 'fig2_encoder_retrieval')


# -- FIGURE 3: Multilingual Win Rates Heatmap ----------------------------------
def figure3_multilingual_heatmap(multilingual_data):
    """
    Heatmap: models (rows) x languages (columns).
    Separate panels for attention vs gradient extractors.
    """
    print("\n[Figure 3] Multilingual Win Rates Heatmap")

    lang_display = {
        'de_native': 'German',
        'fr_native': 'French',
        'tr_native': 'Turkish',
        'ar_native': 'Arabic',
        'hi_native': 'Hindi',
        'cn_native': 'Chinese',
    }

    preferred_langs = ['de_native', 'fr_native', 'tr_native', 'ar_native',
                       'hi_native', 'cn_native']
    extractors = ['attention', 'gradient']

    # Collect models and languages that have data
    models_seen = set()
    langs_seen = set()
    for (model, lang, ext), wr in multilingual_data.items():
        models_seen.add(model)
        langs_seen.add(lang)

    # Filter to models in our standard order, add any extras
    models_list = [m for m in MODEL_ORDER if m in models_seen]
    extra = sorted(models_seen - set(models_list))
    models_list.extend(extra)

    # Filter languages: use preferred order
    langs_list = [l for l in preferred_langs if l in langs_seen]

    if not models_list or not langs_list:
        print("  WARNING: No multilingual data found. Skipping Figure 3.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                              gridspec_kw={'wspace': 0.35})
    fig.suptitle('Multilingual Retrieval Win Rates',
                 fontsize=13, fontweight='bold', y=1.02)

    im = None
    for panel_idx, ext in enumerate(extractors):
        ax = axes[panel_idx]

        # Build matrix
        matrix = np.full((len(models_list), len(langs_list)), np.nan)
        for i, model in enumerate(models_list):
            for j, lang in enumerate(langs_list):
                key = (model, lang, ext)
                if key in multilingual_data:
                    matrix[i, j] = multilingual_data[key]

        # Plot heatmap with diverging colormap centered at 0.55
        vmin, vmax = 0.1, 0.85
        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        # Annotate cells
        for i in range(len(models_list)):
            for j in range(len(langs_list)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, '--', ha='center', va='center',
                            fontsize=8, color='#999999')
                else:
                    text_color = 'white' if val < 0.3 or val > 0.75 else '#222222'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color=text_color)

        # Axis labels
        lang_labels = [lang_display.get(l, l) for l in langs_list]
        ax.set_xticks(range(len(langs_list)))
        ax.set_xticklabels(lang_labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(models_list)))
        ax.set_yticklabels(models_list, fontsize=9)
        ax.set_title(f'{ext.capitalize()} Extractor', fontsize=11,
                     fontweight='bold', pad=8)

        # Draw grid lines between cells
        for edge in range(len(langs_list) + 1):
            ax.axvline(edge - 0.5, color='white', linewidth=1.5)
        for edge in range(len(models_list) + 1):
            ax.axhline(edge - 0.5, color='white', linewidth=1.5)

    # Colorbar
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.04,
                             label='Win Rate')
        cbar.ax.axhline(y=0.55, color='black', linewidth=1.5, linestyle='--')
        cbar.ax.text(1.6, 0.55, '0.55', transform=cbar.ax.get_yaxis_transform(),
                     ha='left', va='center', fontsize=8, color='black')

    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    save_figure(fig, 'fig3_multilingual_heatmap')


# -- FIGURE 4: Dataset-level Aggregation (Violin) ------------------------------
def figure4_dataset_aggregation(delete_data, retrieval_data):
    """
    Violin plot showing distribution of win rates per dataset,
    split by Delete vs Retrieval.
    """
    print("\n[Figure 4] Dataset-level Aggregation")

    datasets = DATASET_ORDER

    # Collect all win rates per (dataset, operator)
    del_by_ds = defaultdict(list)
    ret_by_ds = defaultdict(list)

    for (model, ds, ext), wr in delete_data.items():
        if ds in datasets:
            del_by_ds[ds].append(wr)
    for (model, ds, ext), wr in retrieval_data.items():
        if ds in datasets:
            ret_by_ds[ds].append(wr)

    fig, ax = plt.subplots(figsize=(8, 5))

    x_ticks = []

    for i, ds in enumerate(datasets):
        center = i * 2.5
        pos_del = center - 0.4
        pos_ret = center + 0.4

        del_vals = del_by_ds.get(ds, [])
        ret_vals = ret_by_ds.get(ds, [])

        if len(del_vals) >= 2:
            vp = ax.violinplot([del_vals], positions=[pos_del],
                               showmeans=True, showextrema=False, widths=0.65)
            for body in vp['bodies']:
                body.set_facecolor(COLOR_DELETE)
                body.set_alpha(0.6)
                body.set_edgecolor(COLOR_DELETE)
                body.set_linewidth(0.8)
            vp['cmeans'].set_color(COLOR_DELETE)
            vp['cmeans'].set_linewidth(1.5)
            # Overlay individual points
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(del_vals))
            ax.scatter(pos_del + jitter, del_vals, s=18, color=COLOR_DELETE,
                      alpha=0.7, edgecolors='white', linewidths=0.4, zorder=5)
        elif len(del_vals) == 1:
            ax.scatter([pos_del], del_vals, s=40, color=COLOR_DELETE,
                      edgecolors='white', linewidths=0.5, zorder=5, marker='D')

        if len(ret_vals) >= 2:
            vp = ax.violinplot([ret_vals], positions=[pos_ret],
                               showmeans=True, showextrema=False, widths=0.65)
            for body in vp['bodies']:
                body.set_facecolor(COLOR_RETRIEVAL)
                body.set_alpha(0.6)
                body.set_edgecolor(COLOR_RETRIEVAL)
                body.set_linewidth(0.8)
            vp['cmeans'].set_color(COLOR_RETRIEVAL)
            vp['cmeans'].set_linewidth(1.5)
            jitter = np.random.RandomState(43).uniform(-0.12, 0.12, len(ret_vals))
            ax.scatter(pos_ret + jitter, ret_vals, s=18, color=COLOR_RETRIEVAL,
                      alpha=0.7, edgecolors='white', linewidths=0.4, zorder=5)
        elif len(ret_vals) == 1:
            ax.scatter([pos_ret], ret_vals, s=40, color=COLOR_RETRIEVAL,
                      edgecolors='white', linewidths=0.5, zorder=5, marker='D')

        x_ticks.append(center)

    # Significance threshold
    ax.axhline(y=0.55, color=COLOR_SIG_LINE, linestyle='--',
               linewidth=1.0, alpha=0.8, zorder=2)
    ax.axhline(y=0.50, color='#BBBBBB', linestyle=':',
               linewidth=0.7, alpha=0.6, zorder=1)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('Win Rate', fontsize=11)
    ax.set_title('Distribution of Win Rates by Dataset and Operator',
                 fontsize=12, fontweight='bold', pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0.1, 1.0)

    # Legend
    legend_elements = [
        Patch(facecolor=COLOR_DELETE, alpha=0.6, edgecolor=COLOR_DELETE,
              label='Delete'),
        Patch(facecolor=COLOR_RETRIEVAL, alpha=0.6, edgecolor=COLOR_RETRIEVAL,
              label='Retrieval'),
        plt.Line2D([0], [0], color=COLOR_SIG_LINE, linestyle='--',
                   linewidth=1.0, label='Significance (0.55)'),
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='upper right',
              fontsize=9)

    fig.tight_layout()
    save_figure(fig, 'fig4_dataset_aggregation')


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("ICE-NSR Figure Generator")
    print("Nature/Science magazine style -- Publication quality")
    print("=" * 70)

    # Load all data
    print("\nLoading delete baseline data...")
    delete_data = load_delete_data()
    print(f"  Loaded {len(delete_data)} (model, dataset, extractor) entries")

    print("\nLoading retrieval data...")
    retrieval_data = load_retrieval_data()
    print(f"  Loaded {len(retrieval_data)} (model, dataset, extractor) entries")

    print("\nLoading encoder data...")
    encoder_data = load_encoder_data()
    print(f"  Loaded {len(encoder_data)} (dataset, extractor) entries")

    print("\nLoading multilingual data...")
    multilingual_data = load_multilingual_data()
    print(f"  Loaded {len(multilingual_data)} (model, language, extractor) entries")

    # Debug: print summary tables
    print("\n--- Delete Baseline Summary ---")
    for k in sorted(delete_data.keys()):
        print(f"  {k[0]:18s} | {k[1]:8s} | {k[2]:18s} | {delete_data[k]:.4f}")

    print("\n--- Retrieval Summary ---")
    for k in sorted(retrieval_data.keys()):
        print(f"  {k[0]:18s} | {k[1]:8s} | {k[2]:18s} | {retrieval_data[k]:.4f}")

    print("\n--- Encoder Summary ---")
    for k in sorted(encoder_data.keys()):
        print(f"  {k[0]:8s} | {k[1]:22s} | {encoder_data[k]:.4f}")

    print("\n--- Multilingual Summary ---")
    for k in sorted(multilingual_data.keys()):
        print(f"  {k[0]:18s} | {k[1]:12s} | {k[2]:12s} | {multilingual_data[k]:.4f}")

    # Generate figures
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)

    try:
        figure1_delete_vs_retrieval(delete_data, retrieval_data)
    except Exception as e:
        print(f"  ERROR in Figure 1: {e}")
        import traceback; traceback.print_exc()

    try:
        figure2_encoder_results(encoder_data)
    except Exception as e:
        print(f"  ERROR in Figure 2: {e}")
        import traceback; traceback.print_exc()

    try:
        figure3_multilingual_heatmap(multilingual_data)
    except Exception as e:
        print(f"  ERROR in Figure 3: {e}")
        import traceback; traceback.print_exc()

    try:
        figure4_dataset_aggregation(delete_data, retrieval_data)
    except Exception as e:
        print(f"  ERROR in Figure 4: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
