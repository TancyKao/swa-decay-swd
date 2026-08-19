# Figure scripts — SWA decay and sleep-wake state discrepancy

Analysis and figure-generation code for the manuscript
*"Feeling awake while asleep: Overnight slow wave dissipation associates with
sleep-wake state discrepancy."*

## Contents

| Script | Generates | Cohort |
|---|---|---|
| `fig1_S2_S3_correlations.py` | Figure 1, Figure S2, Figure S3 | Study 1 (clinical insomnia) |
| `fig2_subgroup_comparisons.py` | Figure 2 (SWD subgroup comparisons) | Study 1 (clinical insomnia) |
| `fig3_fig4_S7_SHHS_groups.py` | Figure 3, Figure 4, Figures S7–S8 | Study 2 (SHHS) |

Outputs produced by each script:

- `fig1_S2_S3_correlations.py`
  - `frontal_central_SSM_correlation.png` → **Figure 1**
  - `individual_channels_SSM_correlations.png` → **Figure S2**
  - `sleep_time_correlations.png` → **Figure S3**
  - `individual_channel_SSM_correlations.csv` (statistics)
- `fig2_subgroup_comparisons.py`
  - `mean_comparison_by_group.png` → **Figure 2**
  - `peak_comparison_by_group.png`, `RisingRate_comparison_by_group.png`,
    `RisingTime_comparison_by_group.png` (supplementary/exploratory boxplots)
- `fig3_fig4_S7_SHHS_groups.py`
  - `bar_plot_Central_HC_INS.png` → **Figure 3**
  - `bar_plot_Central_area_combined.png` → **Figure 4**
  - `bar_plot_EEG1_EEG2_combined.png` → **Figure S7**
  - `bar_plot_EEG1_EEG2_combined_SSM.png` → **Figure S8**

(Filenames carry a legacy `SSM` token; legends map to figures by content.)

## Environment

```
python >= 3.10
pip install -r requirements.txt
```

## Input data (not included)

The scripts read pre-processed tabular data, not raw EEG. These files are **not**
in this repository — Study 1 data are available from the lead contact on
reasonable request; Study 2 uses the public Sleep Heart Health Study (SHHS) from
the National Sleep Research Resource (https://sleepdata.org).

- `fig1_S2_S3_correlations.py` expects an Excel workbook with sheets
  `variables` and `abspow` (per-subject decay rates `F3/F4/C3/C4_DECAY_RATE` and
  SWD / sleep-time variables).
- `fig3_fig4_S7_SHHS_groups.py` expects the CSVs listed in its header docstring
  (`df_macro_clean_age_gender_bmi_matched.csv`, `SWA_decay_df.csv`, etc.).

## Running

Paths are configured centrally in `config.py`. Put the input files (see above)
in a `data/` folder next to the scripts; figures are written to `outputs/`.
Both locations can be overridden with environment variables:

```
# default: ./data in, ./outputs out
python fig1_S2_S3_correlations.py

# custom locations
DATA_DIR=/path/to/data OUT_DIR=/path/to/figures python fig3_fig4_S7_SHHS_groups.py
```

Scripts call `plt.show()`; run headless with `MPLBACKEND=Agg`:

```
MPLBACKEND=Agg python fig1_S2_S3_correlations.py
MPLBACKEND=Agg python fig2_subgroup_comparisons.py
MPLBACKEND=Agg python fig3_fig4_S7_SHHS_groups.py
```
