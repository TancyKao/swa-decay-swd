# %% import libraries
"""
Sleep Architecture Analysis Script
This script performs comprehensive analysis of sleep wave activity (SWA) decay patterns
and compares them between insomnia (INS) and healthy control (HC) groups, with special
focus on sleep state misperception (SSM) subgroups.
Main Components:
----------------
1. Data Loading:
    - Loads preprocessed sleep data including macro sleep parameters, power spectral
      density (PSD), sleep cycles, SWA decay, sleep oscillations (SO), and spindles data
2. Data Preparation:
    - Calculates Sleep Perception Index (SPI)
    - Merges SWA decay data with SSM measurements
    - Filters data to matched subjects
3. SSM Subgroup Selection:
    - Categorizes subjects into three groups based on SSM percentiles:
      * Overestimators: SSM <= 10th percentile
      * Underestimators: SSM >= 90th percentile
      * Normal estimators: SSM within ±0.5 standard deviation of mean
4. Statistical Analysis:
    - Performs independent t-tests comparing SWA decay between INS and HC groups
      for each EEG channel
    - Reports means, standard deviations, t-statistics, and p-values
5. Visualizations:
    - Generates bar plots comparing SWA decay between INS and HC groups
    - Creates separate plots for different SSM subgroups (overestimators, underestimators,
      normal estimators)
    - Produces combined plots showing all SSM types together
Input Files:
-----------
- df_macro_clean_age_gender_bmi_matched.csv: Matched demographic and macro sleep data
- df_PSD_STA_no_outliers.csv: Power spectral density statistics without outliers
- df_PSD_CYC_clean_long.csv: Clean PSD cycle data in long format
- SWA_decay_df.csv: Slow wave activity decay measurements
- df_SO_clean_long.csv: Clean sleep oscillation data in long format
- df_Spindles_clean_long.csv: Clean spindle data in long format
Output Files:
------------
- Bar plots saved as PNG files for each channel and SSM subgroup combination
- Combined bar plots showing all SSM types together
Dependencies:
------------
- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- scipy, statsmodels: Statistical analysis
- warnings, logging: Error handling and logging
Author: Tancy Kao
Project: PRJ-11_Oracle_2023
Organization: Woolcock Institute
"""
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
import pandas as pd
import shutil
import tempfile
import warnings
import seaborn as sns
from scipy import stats
#from ads.dataset.factory import DatasetFactory
from os import path
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannotations.Annotator import Annotator

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


# %% load data
from config import DATA_DIR, OUT_DIR
filedir = DATA_DIR + os.sep
df_macro_clean = pd.read_csv(filedir + 'df_macro_clean_age_gender_bmi_matched.csv')
df_PSD_STA_no_outliers = pd.read_csv(filedir + 'df_PSD_STA_no_outliers.csv')
df_PSD_CYC_clean_long = pd.read_csv(filedir + 'df_PSD_CYC_clean_long.csv')
SWA_decay_df = pd.read_csv(filedir + 'SWA_decay_df.csv')
df_SO_clean_long = pd.read_csv(filedir + 'df_SO_clean_long.csv')
df_Spindles_clean_long = pd.read_csv(filedir + 'df_Spindles_clean_long.csv')

savefolder = OUT_DIR


# %% merge data for SWA decay and SSM
df_PSD_STA_no_outliers['SPI'] = df_PSD_STA_no_outliers['SUBJECTIVE_SLEEP_TIME']/df_PSD_STA_no_outliers['SLPPRDP']*100
df_SSM_all = df_PSD_STA_no_outliers[['NSRRID','GENDER','AGE_S1','HC_INS','SUBJECTIVE_SLEEP_TIME','SSM','SLPPRDP','oTST_sTST','SPI']]
df_SSM_all = df_SSM_all.drop_duplicates()

df_SSM = df_SSM_all[df_SSM_all['NSRRID'].isin(df_macro_clean['NSRRID'])] # filter to matched subjects
df_SSM = pd.merge(df_SSM, df_macro_clean[['NSRRID', 'BMI_S1']], on='NSRRID', how='left')

df_SWA_decay_SSM = pd.merge(SWA_decay_df, df_SSM, on=(['NSRRID','HC_INS']), how='left')
# filter out RELPSD
df_SWA_decay_SSM = df_SWA_decay_SSM[df_SWA_decay_SSM['PSD_TYPE'] != 'RELPSD']

# %% select SSM subgroups
target_y = 'SSM' # SPI has same result with SSM, SSM will be neg as overestimators
# Calculate the 90th and 10th percentiles for SSM within each group
percentiles_HC = df_SWA_decay_SSM[df_SWA_decay_SSM.HC_INS=='HC'].groupby('HC_INS')[target_y].quantile([0.10, 0.90]).unstack()
percentiles_INS = df_SWA_decay_SSM[df_SWA_decay_SSM.HC_INS=='INS'].groupby('HC_INS')[target_y].quantile([0.10, 0.90]).unstack()
percentiles_HC.columns = ['lower_percentile', 'Upper_percentile']
percentiles_INS.columns = ['lower_percentile', 'Upper_percentile']


percentiles = pd.concat([percentiles_HC, percentiles_INS], ignore_index=False)

grouped = df_SWA_decay_SSM.groupby('HC_INS')[target_y]
mean_std = grouped.agg(['mean', 'std'])

mean_std['lower_05std'] = mean_std['mean'] - 0.5 * mean_std['std']
mean_std['upper_05std'] = mean_std['mean'] + 0.5 * mean_std['std']

# Select only the necessary columns
mean_std = mean_std[['lower_05std', 'upper_05std']]

df_SWA_decay_SSM = df_SWA_decay_SSM.merge(percentiles, on='HC_INS', how='left')
df_SWA_decay_SSM = df_SWA_decay_SSM.merge(mean_std, on='HC_INS', how='left') 

# Subselect rows
df_SSM_under = df_SWA_decay_SSM[(df_SWA_decay_SSM[target_y] <= df_SWA_decay_SSM['lower_percentile'])]
df_SSM_under['SSM_TYPE'] = 'overest'

df_SSM_over = df_SWA_decay_SSM[(df_SWA_decay_SSM[target_y] >= df_SWA_decay_SSM['Upper_percentile'])]
df_SSM_over['SSM_TYPE'] = 'underest'

df_SSM_normal = df_SWA_decay_SSM[(df_SWA_decay_SSM[target_y] <= df_SWA_decay_SSM['upper_05std']) & (df_SWA_decay_SSM[target_y] >= df_SWA_decay_SSM['lower_05std'])]
df_SSM_normal['SSM_TYPE'] = 'normest'

### append all data
df_SSM_select = pd.concat([df_SSM_under, df_SSM_over], ignore_index=True)
df_SSM_select = pd.concat([df_SSM_select, df_SSM_normal], ignore_index=True)

# calculate mean and std of age, BMI, and proportion of gender in each group and SSM type
grouped_data = df_SSM_select.drop_duplicates(subset='NSRRID').groupby(['HC_INS', 'SSM_TYPE'])
age_bmi_stats = grouped_data[['AGE_S1', 'BMI_S1', 'SSM']].agg(['mean', 'std'])
gender_props = grouped_data['GENDER'].value_counts(normalize=True).unstack(fill_value=0)
sample_sizes = grouped_data.size().to_frame(name='N')
summary_stats = pd.concat([sample_sizes, age_bmi_stats, gender_props], axis=1)

# Add statistical tests comparing HC vs INS for each SSM_TYPE
print("\nStatistical Tests (HC vs INS) by SSM Type:")
print("=" * 80)

df_unique = df_SSM_select.drop_duplicates(subset='NSRRID')

for ssm_type in df_unique['SSM_TYPE'].unique():
    print(f"\nSSM Type: {ssm_type}")
    print("-" * 80)
    
    df_ssm = df_unique[df_unique['SSM_TYPE'] == ssm_type]
    hc_data = df_ssm[df_ssm['HC_INS'] == 'HC']
    ins_data = df_ssm[df_ssm['HC_INS'] == 'INS']
    
    # T-test for AGE_S1
    t_stat_age, p_val_age = stats.ttest_ind(hc_data['AGE_S1'].dropna(), 
                                             ins_data['AGE_S1'].dropna())
    # Cohen's d for AGE_S1
    mean_diff_age = hc_data['AGE_S1'].mean() - ins_data['AGE_S1'].mean()
    pooled_std_age = np.sqrt(((len(hc_data['AGE_S1']) - 1) * hc_data['AGE_S1'].std()**2 + 
                               (len(ins_data['AGE_S1']) - 1) * ins_data['AGE_S1'].std()**2) / 
                              (len(hc_data['AGE_S1']) + len(ins_data['AGE_S1']) - 2))
    cohens_d_age = mean_diff_age / pooled_std_age
    print(f"AGE_S1: t={t_stat_age:.3f}, p={p_val_age:.4f}, d={cohens_d_age:.3f}")
    
    # T-test for BMI_S1
    t_stat_bmi, p_val_bmi = stats.ttest_ind(hc_data['BMI_S1'].dropna(), 
                                             ins_data['BMI_S1'].dropna())
    # Cohen's d for BMI_S1
    mean_diff_bmi = hc_data['BMI_S1'].mean() - ins_data['BMI_S1'].mean()
    pooled_std_bmi = np.sqrt(((len(hc_data['BMI_S1']) - 1) * hc_data['BMI_S1'].std()**2 + 
                               (len(ins_data['BMI_S1']) - 1) * ins_data['BMI_S1'].std()**2) / 
                              (len(hc_data['BMI_S1']) + len(ins_data['BMI_S1']) - 2))
    cohens_d_bmi = mean_diff_bmi / pooled_std_bmi
    print(f"BMI_S1: t={t_stat_bmi:.3f}, p={p_val_bmi:.4f}, d={cohens_d_bmi:.3f}")
    
    # T-test for SSM
    t_stat_ssm, p_val_ssm = stats.ttest_ind(hc_data['SSM'].dropna(), 
                                             ins_data['SSM'].dropna())
    # Cohen's d for SSM
    mean_diff_ssm = hc_data['SSM'].mean() - ins_data['SSM'].mean()
    pooled_std_ssm = np.sqrt(((len(hc_data['SSM']) - 1) * hc_data['SSM'].std()**2 + 
                               (len(ins_data['SSM']) - 1) * ins_data['SSM'].std()**2) / 
                              (len(hc_data['SSM']) + len(ins_data['SSM']) - 2))
    cohens_d_ssm = mean_diff_ssm / pooled_std_ssm
    print(f"SSM: t={t_stat_ssm:.3f}, p={p_val_ssm:.4f}, d={cohens_d_ssm:.3f}")
    
    # Chi-square test for GENDER
    contingency_table = pd.crosstab(df_ssm['HC_INS'], df_ssm['GENDER'])
    chi2, p_val_gender, dof, expected = stats.chi2_contingency(contingency_table)
    # Cramér's V for GENDER
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    print(f"GENDER: χ²={chi2:.3f}, p={p_val_gender:.4f}, V={cramers_v:.3f}")

print("\nSummary Statistics by Group and SSM Type:")
print("=" * 80)

# Create formatted output with mean ± std
for (hc_ins, ssm_type), group_data in summary_stats.iterrows():
    print(f"\n{hc_ins} - {ssm_type}:")
    print(f"  N: {int(group_data['N'])}")
    print(f"  Age: {group_data[('AGE_S1', 'mean')]:.2f} ± {group_data[('AGE_S1', 'std')]:.2f}")
    print(f"  BMI: {group_data[('BMI_S1', 'mean')]:.2f} ± {group_data[('BMI_S1', 'std')]:.2f}")
    print(f"  SSM: {group_data[('SSM', 'mean')]:.2f} ± {group_data[('SSM', 'std')]:.2f}")

    gender_cols = [col for col in group_data.index if isinstance(col, (int, float)) and col in [1, 2]]
    if gender_cols:
        for gender_val in gender_cols:
            gender_prop = group_data[gender_val]
            gender_count = int(gender_prop * group_data['N'])
            gender_label = 'Male' if gender_val == 1 else 'Female'
            print(f"  {gender_label}: {gender_count} ({gender_prop*100:.1f}%)")



# %% compare SWA decay between INS and HC using t-test
results = []
for chan in df_SWA_decay_SSM['CH'].unique():
    
    # Filter the DataFrame for the current chan
    df_filtered = df_SWA_decay_SSM[(df_SWA_decay_SSM['CH'] == chan)]
    
    # Separate data by group
    ins_data = df_filtered[df_filtered['HC_INS'] == 'INS']['Decay']
    hc_data = df_filtered[df_filtered['HC_INS'] == 'HC']['Decay']
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(ins_data, hc_data)
    
    # Calculate Cohen's d (effect size)
    mean_diff = ins_data.mean() - hc_data.mean()
    pooled_std = np.sqrt(((len(ins_data) - 1) * ins_data.std()**2 + 
                          (len(hc_data) - 1) * hc_data.std()**2) / 
                         (len(ins_data) + len(hc_data) - 2))
    cohens_d = mean_diff / pooled_std
    
    # Store the results
    results.append({
        'chan': chan,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'INS_mean': ins_data.mean(),
        'HC_mean': hc_data.mean(),
        'INS_std': ins_data.std(),
        'HC_std': hc_data.std()
    })

# Print the results
for result in results:
    print(f"Channel: {result['chan']}")
    print(f"INS: mean={result['INS_mean']:.4f}, std={result['INS_std']:.4f}")
    print(f"HC: mean={result['HC_mean']:.4f}, std={result['HC_std']:.4f}")
    print(f"t-statistic: {result['t_statistic']:.4f}, p-value: {result['p_value']:.4f}, Cohen's d: {result['cohens_d']:.4f}")
    print("\n")

# %% plot bar plot between INS and HC
import seaborn as sns

# Combine EEG1 and EEG2 into one figure with two panels
# Filter data for EEG1 and EEG2 channels
df_eeg1_eeg2 = df_SWA_decay_SSM[df_SWA_decay_SSM['CH'].isin(['EEG1', 'EEG2'])].copy()

# Change order: HC first, then INS
df_eeg1_eeg2['HC_INS'] = pd.Categorical(df_eeg1_eeg2['HC_INS'], categories=['HC', 'INS'], ordered=True)
df_eeg1_eeg2 = df_eeg1_eeg2.sort_values(['CH', 'HC_INS'])
df_eeg1_eeg2['Decay'] = df_eeg1_eeg2['Decay'] * 100

# Create figure with two subplots (180mm = 7.087 inches)
fig, axes = plt.subplots(1, 2, figsize=(7.087, 3), dpi=600)

# Plot EEG2 (C3) on left panel
df_eeg2 = df_eeg1_eeg2[df_eeg1_eeg2['CH'] == 'EEG2']
sns.barplot(x='HC_INS', y='Decay', data=df_eeg2, ci='sd',
            palette=['#67a9cf', '#ef8a62'], ax=axes[0], capsize=0.1)
axes[0].set_ylabel('Decay Rate (%)', fontsize=10)
axes[0].set_xlabel(None)
axes[0].set_title('C3', fontsize=10)
axes[0].tick_params(labelsize=8)
sns.despine(ax=axes[0], top=True, right=True)

# Add statistical annotation for EEG2
pairs_eeg2 = [('HC', 'INS')]
eeg2_result = [r for r in results if r['chan'] == 'EEG2'][0]
annotator_eeg2 = Annotator(axes[0], pairs_eeg2, data=df_eeg2, x='HC_INS', y='Decay')
annotator_eeg2.configure(test=None, text_format='star', loc='inside')
annotator_eeg2.set_pvalues([eeg2_result['p_value']])
annotator_eeg2.annotate()
axes[0].set_xticklabels(['Good sleepers', 'Insomnia symptoms'])

# Plot EEG1 (C4) on right panel
df_eeg1 = df_eeg1_eeg2[df_eeg1_eeg2['CH'] == 'EEG1']
sns.barplot(x='HC_INS', y='Decay', data=df_eeg1, ci='sd',
            palette=['#67a9cf', '#ef8a62'], ax=axes[1], capsize=0.1)
axes[1].set_ylabel('Decay Rate (%)', fontsize=10)
axes[1].set_xlabel(None)
axes[1].set_title('C4', fontsize=10)
axes[1].tick_params(labelsize=8)
sns.despine(ax=axes[1], top=True, right=True)

# Add statistical annotation for EEG1
pairs_eeg1 = [('HC', 'INS')]
eeg1_result = [r for r in results if r['chan'] == 'EEG1'][0]
annotator_eeg1 = Annotator(axes[1], pairs_eeg1, data=df_eeg1, x='HC_INS', y='Decay')
annotator_eeg1.configure(test=None, text_format='star', loc='inside')
annotator_eeg1.set_pvalues([eeg1_result['p_value']])
annotator_eeg1.annotate()
axes[1].set_xticklabels(['Good sleepers', 'Insomnia symptoms'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save as PNG with 600 DPI
plt.savefig(os.path.join(savefolder, 'bar_plot_EEG1_EEG2_combined.png'),
            dpi=600, bbox_inches='tight')

# Show the plot
plt.show()

# %% Create merged central area plot for HC vs INS (whole groups)
# Merge EEG1 and EEG2 for whole HC and INS groups
df_central_whole = df_SWA_decay_SSM[df_SWA_decay_SSM['CH'].isin(['EEG1', 'EEG2'])].copy()

# Calculate mean Decay across EEG1 and EEG2 for each subject
df_central_whole_merged = df_central_whole.groupby(['NSRRID', 'HC_INS']).agg({
    'Decay': 'mean'
}).reset_index()

# Convert to percentage
df_central_whole_merged['Decay'] = df_central_whole_merged['Decay'] * 100

# Change order: HC first, then INS
df_central_whole_merged['HC_INS'] = pd.Categorical(df_central_whole_merged['HC_INS'],
                                                     categories=['HC', 'INS'], ordered=True)
df_central_whole_merged = df_central_whole_merged.sort_values('HC_INS')

# Convert mm to inches (90mm = half of 180mm = 3.543 inches)
fig_width = 90 / 25.4
fig_height = 3

# Perform statistical test for Central HC vs INS (before plotting)
hc_central = df_central_whole_merged[df_central_whole_merged['HC_INS'] == 'HC']['Decay']
ins_central = df_central_whole_merged[df_central_whole_merged['HC_INS'] == 'INS']['Decay']

# T-test
t_stat, p_value = stats.ttest_ind(hc_central, ins_central)

# Create the bar plot
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

sns.barplot(x='HC_INS', y='Decay', data=df_central_whole_merged, ci='sd',
            palette=['#67a9cf', '#ef8a62'], ax=ax, capsize=0.1)

ax.set_ylabel('Decay Rate (%)', fontsize=14)
ax.set_xlabel('', fontsize=14)
ax.set_title('Central Area', fontsize=14)
ax.tick_params(labelsize=12)
sns.despine(ax=ax, top=True, right=True)

# Add statistical annotation for Central area
pairs_central = [('HC', 'INS')]
annotator_central = Annotator(ax, pairs_central, data=df_central_whole_merged, x='HC_INS', y='Decay')
annotator_central.configure(test=None, text_format='star', loc='inside')
annotator_central.set_pvalues([p_value])
annotator_central.annotate()
ax.set_xticklabels(['Good\nsleepers', 'Insomnia\nsymptoms'])

plt.tight_layout()

# Save as PNG with 600 DPI
plt.savefig(os.path.join(savefolder, 'bar_plot_Central_HC_INS.png'),
            dpi=600, bbox_inches='tight')

print(f"\nPlot saved to: {os.path.join(savefolder, 'bar_plot_Central_HC_INS.png')}")

# Show the plot
plt.show()

# Calculate Cohen's d
mean_diff = hc_central.mean() - ins_central.mean()
pooled_std = np.sqrt(((len(hc_central) - 1) * hc_central.std()**2 +
                      (len(ins_central) - 1) * ins_central.std()**2) /
                     (len(hc_central) + len(ins_central) - 2))
cohens_d = mean_diff / pooled_std

# Print summary statistics
print("\n" + "="*80)
print("STATISTICAL SUMMARY: Central Area (EEG1+EEG2) - HC vs INS")
print("="*80)
print(f"HC: Mean±SD = {hc_central.mean():.2f}±{hc_central.std():.2f}, N = {len(hc_central)}")
print(f"INS: Mean±SD = {ins_central.mean():.2f}±{ins_central.std():.2f}, N = {len(ins_central)}")
print(f"t-value = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")
print(f"Effect Size (Cohen's d) = {cohens_d:.4f}")
print("="*80)



# %% plot bar plot in over, under, normestimators in HC and INS
for chan in df_SWA_decay_SSM['CH'].unique():
    for ssmt in df_SSM_select['SSM_TYPE'].unique():
        # Filter the DataFrame for the current channel
        df_filtered = df_SSM_select[(df_SSM_select['CH'] == chan) & (df_SSM_select['SSM_TYPE'] == ssmt)]
        df_filtered['HC_INS'] = pd.Categorical(df_filtered['HC_INS'], categories=['INS', 'HC'], ordered=True)
        df_filtered = df_filtered.sort_values('HC_INS')
        df_filtered['Decay'] = df_filtered['Decay'] * 100

        # Set the palette based on the SSM type
        if ssmt == 'overest':
            palette = ['#fee090', '#e0f3f8']
        elif ssmt == 'underest':
            palette = ['#fc8d59', '#91bfdb']
        else:
            palette = ['#ef8a62', '#67a9cf']  # Default palette

        # Create the bar plot with custom colors
        plt.figure(figsize=(6,4))
        sns.barplot(x='HC_INS', y='Decay', data=df_filtered, ci='sd', palette=palette, capsize=0.1)
        
        sns.despine(top=True, right=True)
        
        # Customize tick and label sizes
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    
        plt.ylabel('Decay Rate (%)', fontsize=20)
        plt.xlabel(None)  # Remove xlabel
        #plt.title(f'Group Bar Plot for Channel {chan} and SSM Type {ssmt}', fontsize=16)
        
        # Save the figure with ssmt in the filename
        plt.savefig(os.path.join(savefolder, f'bar_plot_channel_{chan}_ssm_type_{ssmt}.png'))
        
        # Show the plot
        plt.show()

# %%
# Combine EEG2 and EEG1 combined plots into one figure with two panels
# Filter data for EEG1 and EEG2 channels
df_eeg1_eeg2_combined = df_SSM_select[df_SSM_select['CH'].isin(['EEG1', 'EEG2'])].copy()

# Rename SSM_TYPE values
df_eeg1_eeg2_combined['SSM_TYPE'] = df_eeg1_eeg2_combined['SSM_TYPE'].map({
    'overest': 'OverEst',
    'normest': 'OptEst',
    'underest': 'UnderEst'
})

# Change order: HC first, then INS
df_eeg1_eeg2_combined['HC_INS'] = pd.Categorical(df_eeg1_eeg2_combined['HC_INS'], categories=['HC', 'INS'], ordered=True)
df_eeg1_eeg2_combined['SSM_TYPE'] = pd.Categorical(df_eeg1_eeg2_combined['SSM_TYPE'], categories=['OverEst', 'OptEst', 'UnderEst'], ordered=True)
df_eeg1_eeg2_combined = df_eeg1_eeg2_combined.sort_values(['CH', 'SSM_TYPE', 'HC_INS'])
df_eeg1_eeg2_combined['Decay'] = df_eeg1_eeg2_combined['Decay'] * 100

# Create figure with two subplots (180mm = 7.087 inches)
fig, axes = plt.subplots(1, 2, figsize=(7.087, 3), dpi=600)

# Plot EEG2 (C3) on left panel
df_eeg2_combined = df_eeg1_eeg2_combined[df_eeg1_eeg2_combined['CH'] == 'EEG2']
sns.barplot(x='SSM_TYPE', y='Decay', hue='HC_INS', data=df_eeg2_combined, ci='sd',
            palette={'INS': '#ef8a62', 'HC': '#67a9cf'}, ax=axes[0], capsize=0.1)
axes[0].set_ylabel('Decay Rate (%)', fontsize=10)
axes[0].set_xlabel(None)
axes[0].set_title('C3', fontsize=10)
axes[0].tick_params(labelsize=8)
if axes[0].get_legend() is not None:
    axes[0].get_legend().remove()
sns.despine(ax=axes[0], top=True, right=True)

# Plot EEG1 (C4) on right panel
df_eeg1_combined = df_eeg1_eeg2_combined[df_eeg1_eeg2_combined['CH'] == 'EEG1']
sns.barplot(x='SSM_TYPE', y='Decay', hue='HC_INS', data=df_eeg1_combined, ci='sd',
            palette={'INS': '#ef8a62', 'HC': '#67a9cf'}, ax=axes[1], capsize=0.1)
axes[1].set_ylabel('Decay Rate (%)', fontsize=10)
axes[1].set_xlabel(None)
axes[1].set_title('C4', fontsize=10)
axes[1].tick_params(labelsize=8)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='#67a9cf', label='Good sleepers'),
                  Patch(facecolor='#ef8a62', label='Insomnia symptoms')]
if axes[1].get_legend() is not None:
    axes[1].get_legend().remove()
axes[1].legend(handles=legend_handles, fontsize=8,
               loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
sns.despine(ax=axes[1], top=True, right=True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save as PNG with 600 DPI
plt.savefig(os.path.join(savefolder, 'bar_plot_EEG1_EEG2_combined_SSM.png'),
            dpi=600, bbox_inches='tight')

# Show the plot
plt.show()

# %% ANCOVA analysis
# Prepare data for ANCOVA
df_ancova = df_SSM_select.copy()

# Convert categorical variables to appropriate format
df_ancova['HC_INS'] = df_ancova['HC_INS'].astype('category')
df_ancova['SSM_TYPE'] = df_ancova['SSM_TYPE'].astype('category')
df_ancova['GENDER'] = df_ancova['GENDER'].astype('category')

# Perform ANCOVA for each channel separately
ancova_results = []

for chan in df_ancova['CH'].unique():
    print(f"\n{'='*80}")
    print(f"ANCOVA Results for Channel: {chan}")
    print(f"{'='*80}")
    
    # Filter data for current channel
    df_chan = df_ancova[df_ancova['CH'] == chan].copy()
    
    # Remove rows with missing values in covariates
    df_chan = df_chan.dropna(subset=['Decay', 'AGE_S1', 'BMI_S1', 'GENDER', 'HC_INS', 'SSM_TYPE'])
    
    print(f"\nSample size: N = {len(df_chan)}")
    print(f"\nDescriptive Statistics:")
    grouped = df_chan.groupby(['HC_INS', 'SSM_TYPE'])['Decay'].agg(['mean', 'std', 'count'])
    print(grouped.round(4))
    
    # Build ANCOVA model with interaction between HC_INS and SSM_TYPE
    formula = 'Decay ~ C(HC_INS) + C(SSM_TYPE) + C(HC_INS):C(SSM_TYPE) + AGE_S1 + C(GENDER) + BMI_S1'
    
    try:
        model = smf.ols(formula=formula, data=df_chan).fit()
        
        # Store results
        ancova_results.append({
            'Channel': chan,
            'N': len(df_chan),
            'Model': model,
            'Summary': model.summary()
        })
        
        # Print model summary
        print(f"\nRegression Summary:")
        print(model.summary())
        
        # Print ANOVA table
        print(f"\nType II ANOVA Table:")
        print(sm.stats.anova_lm(model, typ=2))
        
    except Exception as e:
        print(f"Error analyzing channel {chan}: {str(e)}")

# %% Post-hoc analysis for SSM_TYPE
# Perform Tukey HSD post-hoc test for SSM_TYPE comparisons
print("\n" + "="*80)
print("POST-HOC ANALYSIS: Tukey HSD for SSM_TYPE")
print("="*80)

for chan in df_ancova['CH'].unique():
    print(f"\n{'='*80}")
    print(f"Post-hoc Analysis for Channel: {chan}")
    print(f"{'='*80}")
    
    # Filter data for current channel
    df_chan = df_ancova[df_ancova['CH'] == chan].copy()
    
    # Remove rows with missing values
    df_chan = df_chan.dropna(subset=['Decay', 'SSM_TYPE'])
    
    # Perform Tukey HSD test for SSM_TYPE
    print(f"\nTukey HSD Test for SSM_TYPE:")
    print("-" * 80)
    tukey_ssm = pairwise_tukeyhsd(endog=df_chan['Decay'],
                                   groups=df_chan['SSM_TYPE'],
                                   alpha=0.05)
    print(tukey_ssm)
    
    # Calculate effect sizes (Cohen's d) for each pairwise comparison
    print(f"\nEffect Sizes (Cohen's d) for SSM_TYPE comparisons:")
    print("-" * 80)
    
    ssm_types = df_chan['SSM_TYPE'].unique()
    for i, ssm1 in enumerate(ssm_types):
        for ssm2 in ssm_types[i+1:]:
            data1 = df_chan[df_chan['SSM_TYPE'] == ssm1]['Decay']
            data2 = df_chan[df_chan['SSM_TYPE'] == ssm2]['Decay']
            
            mean_diff = data1.mean() - data2.mean()
            pooled_std = np.sqrt(((len(data1) - 1) * data1.std()**2 +
                                  (len(data2) - 1) * data2.std()**2) /
                                 (len(data1) + len(data2) - 2))
            cohens_d = mean_diff / pooled_std
            
            print(f"{ssm1} vs {ssm2}: Cohen's d = {cohens_d:.4f}")
            print(f"  {ssm1}: mean={data1.mean():.4f}, std={data1.std():.4f}, n={len(data1)}")
            print(f"  {ssm2}: mean={data2.mean():.4f}, std={data2.std():.4f}, n={len(data2)}")

# %% Merge EEG2 (C3) and EEG1 (C4) as 'central area' and perform ANCOVA
# Create a new dataframe with EEG2 (C3) and EEG1 (C4) merged as 'central area'
print("\n" + "="*80)
print("MERGING EEG2 (C3) AND EEG1 (C4) AS 'CENTRAL AREA'")
print("="*80)

# Filter for EEG2 (C3) and EEG1 (C4) channels
df_central = df_SSM_select[df_SSM_select['CH'].isin(['EEG1', 'EEG2'])].copy()

# Calculate mean Decay across EEG1 and EEG2 for each subject
df_central_merged = df_central.groupby(['NSRRID', 'HC_INS', 'SSM_TYPE', 'AGE_S1',
                                         'BMI_S1', 'GENDER', 'SSM', 'PSD_TYPE']).agg({
    'Decay': 'mean'  # Average decay across C3 and C4
}).reset_index()

# Add a channel label for the merged data
df_central_merged['CH'] = 'Central'

print(f"\nOriginal data shape (EEG1 and EEG2): {df_central.shape}")
print(f"Merged data shape (Central): {df_central_merged.shape}")
print(f"\nSample size by group:")
print(df_central_merged.groupby(['HC_INS', 'SSM_TYPE']).size())

# %% ANCOVA analysis for Central area
print("\n" + "="*80)
print("ANCOVA ANALYSIS FOR CENTRAL AREA (EEG2/C3 + EEG1/C4)")
print("="*80)

# Prepare data for ANCOVA
df_central_ancova = df_central_merged.copy()

# Convert categorical variables to appropriate format
df_central_ancova['HC_INS'] = df_central_ancova['HC_INS'].astype('category')
df_central_ancova['SSM_TYPE'] = df_central_ancova['SSM_TYPE'].astype('category')
df_central_ancova['GENDER'] = df_central_ancova['GENDER'].astype('category')

# Remove rows with missing values in covariates
df_central_ancova = df_central_ancova.dropna(subset=['Decay', 'AGE_S1', 'BMI_S1',
                                                       'GENDER', 'HC_INS', 'SSM_TYPE'])

print(f"\nSample size: N = {len(df_central_ancova)}")
print(f"\nDescriptive Statistics:")
grouped = df_central_ancova.groupby(['HC_INS', 'SSM_TYPE'])['Decay'].agg(['mean', 'std', 'count'])
print(grouped.round(4))

# Build ANCOVA model with interaction between HC_INS and SSM_TYPE
formula = 'Decay ~ C(HC_INS) + C(SSM_TYPE) + C(HC_INS):C(SSM_TYPE) + AGE_S1 + C(GENDER) + BMI_S1'

try:
    model_central = smf.ols(formula=formula, data=df_central_ancova).fit()
    
    # Print model summary
    print(f"\nRegression Summary:")
    print(model_central.summary())
    
    # Print ANOVA table
    print(f"\nType II ANOVA Table:")
    anova_table = sm.stats.anova_lm(model_central, typ=2)
    print(anova_table)
    
    # Calculate partial eta-squared for each effect
    print(f"\nEffect Sizes (Partial η²):")
    print("-" * 80)
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    for index, row in anova_table.iterrows():
        if index != 'Residual':
            partial_eta_squared = row['sum_sq'] / (row['sum_sq'] + ss_residual)
            print(f"{index}: ηp² = {partial_eta_squared:.4f} ({partial_eta_squared*100:.2f}%)")
    
except Exception as e:
    print(f"Error analyzing central area: {str(e)}")

# %% Post-hoc analysis for Central area
print("\n" + "="*80)
print("POST-HOC ANALYSIS FOR CENTRAL AREA: Tukey HSD")
print("="*80)

# Perform Tukey HSD test for SSM_TYPE
print(f"\nTukey HSD Test for SSM_TYPE (Central Area):")
print("-" * 80)
tukey_ssm_central = pairwise_tukeyhsd(endog=df_central_ancova['Decay'],
                                       groups=df_central_ancova['SSM_TYPE'],
                                       alpha=0.05)
print(tukey_ssm_central)

# Calculate effect sizes (Cohen's d) for each pairwise comparison
print(f"\nEffect Sizes (Cohen's d) for SSM_TYPE comparisons:")
print("-" * 80)

ssm_types = df_central_ancova['SSM_TYPE'].unique()
for i, ssm1 in enumerate(ssm_types):
    for ssm2 in ssm_types[i+1:]:
        data1 = df_central_ancova[df_central_ancova['SSM_TYPE'] == ssm1]['Decay']
        data2 = df_central_ancova[df_central_ancova['SSM_TYPE'] == ssm2]['Decay']
        
        mean_diff = data1.mean() - data2.mean()
        pooled_std = np.sqrt(((len(data1) - 1) * data1.std()**2 +
                              (len(data2) - 1) * data2.std()**2) /
                             (len(data1) + len(data2) - 2))
        cohens_d = mean_diff / pooled_std
        
        print(f"{ssm1} vs {ssm2}: Cohen's d = {cohens_d:.4f}")
        print(f"  {ssm1}: mean={data1.mean():.4f}, std={data1.std():.4f}, n={len(data1)}")
        print(f"  {ssm2}: mean={data2.mean():.4f}, std={data2.std():.4f}, n={len(data2)}")

# Perform Tukey HSD test for HC_INS
print(f"\n\nTukey HSD Test for HC_INS (Central Area):")
print("-" * 80)
tukey_hc_central = pairwise_tukeyhsd(endog=df_central_ancova['Decay'],
                                      groups=df_central_ancova['HC_INS'],
                                      alpha=0.05)
print(tukey_hc_central)

# Calculate effect size for HC vs INS
hc_data = df_central_ancova[df_central_ancova['HC_INS'] == 'HC']['Decay']
ins_data = df_central_ancova[df_central_ancova['HC_INS'] == 'INS']['Decay']

mean_diff = hc_data.mean() - ins_data.mean()
pooled_std = np.sqrt(((len(hc_data) - 1) * hc_data.std()**2 +
                      (len(ins_data) - 1) * ins_data.std()**2) /
                     (len(hc_data) + len(ins_data) - 2))
cohens_d = mean_diff / pooled_std

print(f"\nEffect Size (Cohen's d) for HC vs INS:")
print("-" * 80)
print(f"HC vs INS: Cohen's d = {cohens_d:.4f}")
print(f"  HC: mean={hc_data.mean():.4f}, std={hc_data.std():.4f}, n={len(hc_data)}")
print(f"  INS: mean={ins_data.mean():.4f}, std={ins_data.std():.4f}, n={len(ins_data)}")

# Perform Tukey HSD test for HC_INS x SSM_TYPE interaction
print(f"\n\nTukey HSD Test for HC_INS x SSM_TYPE Interaction (Central Area):")
print("-" * 80)

# Create a combined group variable for interaction
df_central_ancova['Group_Interaction'] = df_central_ancova['HC_INS'].astype(str) + '_' + df_central_ancova['SSM_TYPE'].astype(str)

tukey_interaction = pairwise_tukeyhsd(endog=df_central_ancova['Decay'],
                                       groups=df_central_ancova['Group_Interaction'],
                                       alpha=0.05)
print(tukey_interaction)

# %% Visualization for Central area
print("\n" + "="*80)
print("CREATING VISUALIZATION FOR CENTRAL AREA")
print("="*80)

# Prepare data for visualization
df_central_plot = df_central_merged.copy()

# Rename SSM_TYPE values
df_central_plot['SSM_TYPE'] = df_central_plot['SSM_TYPE'].map({
    'overest': 'OverEst',
    'normest': 'OptEst',
    'underest': 'UnderEst'
})

# Change order: HC first, then INS
df_central_plot['HC_INS'] = pd.Categorical(df_central_plot['HC_INS'],
                                            categories=['HC', 'INS'], ordered=True)
df_central_plot['SSM_TYPE'] = pd.Categorical(df_central_plot['SSM_TYPE'],
                                              categories=['OverEst', 'OptEst', 'UnderEst'],
                                              ordered=True)
df_central_plot = df_central_plot.sort_values(['SSM_TYPE', 'HC_INS'])
df_central_plot['Decay'] = df_central_plot['Decay'] * 100

# Convert mm to inches (180mm = 7.087 inches)
fig_width = 180 / 25.4  # 180mm in inches
fig_height = 3  # maximum height in inches

# Create the bar plot
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

sns.barplot(x='SSM_TYPE', y='Decay', hue='HC_INS', data=df_central_plot, ci='sd',
            palette={'INS': '#ef8a62', 'HC': '#67a9cf'}, ax=ax, capsize=0.1)

ax.set_ylabel('Decay Rate (%)', fontsize=14)
ax.set_xlabel('', fontsize=14)
ax.set_title('Central Area', fontsize=14)
ax.set_ylim(0, 80)
ax.tick_params(labelsize=12)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='#67a9cf', label='Good sleepers'),
                  Patch(facecolor='#ef8a62', label='Insomnia symptoms')]
if ax.get_legend() is not None:
    ax.get_legend().remove()
ax.legend(handles=legend_handles, fontsize=8, loc='upper right', frameon=False)
sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()

# Save as PNG with 600 DPI
plt.savefig(os.path.join(savefolder, 'bar_plot_Central_area_combined.png'),
            dpi=600, bbox_inches='tight')

print(f"\nPlot saved to: {os.path.join(savefolder, 'bar_plot_Central_area_combined.png')}")

# Show the plot
plt.show()

# %%
