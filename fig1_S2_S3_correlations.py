# plot correlation between decay rates and SSM

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import matplotlib.gridspec as gridspec
import os

# Define output directory
from config import DATA_DIR, OUT_DIR
output_dir = OUT_DIR
os.makedirs(output_dir, exist_ok=True)

# Load the data from Excel file
file_path = os.path.join(DATA_DIR, 'correlation decay rate copy2R.xlsx')

# Load both sheets
variables_df = pd.read_excel(file_path, sheet_name='variables')
abspow_df = pd.read_excel(file_path, sheet_name='abspow')
# Print basic info about the loaded data
print("Variables Sheet Data:")
print(variables_df.head())
print("\nVariables Sheet Columns:", variables_df.columns.tolist())
print("\nAbspow Sheet Data:")
print(abspow_df.head())
print("\nAbspow Sheet Columns:", abspow_df.columns.tolist())

abspow_df['frontal_decay_rate'] = abspow_df[['F3_DECAY_RATE', 'F4_DECAY_RATE']].mean(axis=1, skipna=True)

abspow_df['central_decay_rate'] = abspow_df[['C3_DECAY_RATE', 'C4_DECAY_RATE']].mean(axis=1, skipna=True)

id_column = 'Subject'  # Replace with your actual identifier column
if id_column in variables_df.columns and id_column in abspow_df.columns:
    merged_df = pd.merge(variables_df, abspow_df, on=id_column, how='inner')
    print(f"\nMerged data contains {len(merged_df)} rows")



# %%

def safe_pearsonr(x, y):
    # Drop rows where either x or y is NaN
    # NOTE: uses Spearman correlation (matches manuscript Methods)
    mask = ~(np.isnan(x) | np.isnan(y))
    if sum(mask) < 3:  # Need at least 3 valid pairs for meaningful correlation
        return np.nan, np.nan
    return spearmanr(x[mask], y[mask])

# Individual channel correlations with SSM
print("\n=== Individual Channel Correlations with SSM ===")
individual_channels = ['F3_DECAY_RATE', 'F4_DECAY_RATE', 'C3_DECAY_RATE', 'C4_DECAY_RATE']
individual_corr_results = {}

for channel in individual_channels:
    corr, pval = safe_pearsonr(merged_df[channel], merged_df['SSM'])
    individual_corr_results[channel] = {'r': corr, 'p': pval}
    print(f"{channel} vs SSM: r = {corr:.3f}, p = {pval:.4f}")

# For frontal decay rate vs SSM
frontal_ssm_corr, frontal_ssm_pval = safe_pearsonr(merged_df['frontal_decay_rate'], merged_df['SSM'])
print(f"\nFrontal (F3+F4 averaged) vs SSM: r = {frontal_ssm_corr:.3f}, p = {frontal_ssm_pval:.4f}")

# For central decay rate vs SSM
central_ssm_corr, central_ssm_pval = safe_pearsonr(merged_df['central_decay_rate'], merged_df['SSM'])
print(f"Central (C3+C4 averaged) vs SSM: r = {central_ssm_corr:.3f}, p = {central_ssm_pval:.4f}")

# For frontal decay rate vs subjective sleep time
frontal_subj_corr, frontal_subj_pval = safe_pearsonr(merged_df['frontal_decay_rate'], merged_df['Subjective_sleep_time'])
print(f"\nFrontal Decay Rate vs Subjective Sleep Time: r = {frontal_subj_corr:.3f}, p = {frontal_subj_pval:.4f}")

# For central decay rate vs subjective sleep time
central_subj_corr, central_subj_pval = safe_pearsonr(merged_df['central_decay_rate'], merged_df['Subjective_sleep_time'])
print(f"Central Decay Rate vs Subjective Sleep Time: r = {central_subj_corr:.3f}, p = {central_subj_pval:.4f}")

# For frontal decay rate vs objective sleep time
frontal_obj_corr, frontal_obj_pval = safe_pearsonr(merged_df['frontal_decay_rate'], merged_df['Objective_sleep_time'])
print(f"Frontal Decay Rate vs Objective Sleep Time: r = {frontal_obj_corr:.3f}, p = {frontal_obj_pval:.4f}")

# For central decay rate vs objective sleep time
central_obj_corr, central_obj_pval = safe_pearsonr(merged_df['central_decay_rate'], merged_df['Objective_sleep_time'])
print(f"Central Decay Rate vs Objective Sleep Time: r = {central_obj_corr:.3f}, p = {central_obj_pval:.4f}")

# Export individual channel statistics to CSV
export_data = []
for channel in individual_channels:
    export_data.append({
        'Channel': channel,
        'Variable': 'SSM',
        'Correlation_r': individual_corr_results[channel]['r'],
        'P_value': individual_corr_results[channel]['p']
    })

# Add averaged regional results
export_data.append({
    'Channel': 'Frontal_Average',
    'Variable': 'SSM',
    'Correlation_r': frontal_ssm_corr,
    'P_value': frontal_ssm_pval
})
export_data.append({
    'Channel': 'Central_Average',
    'Variable': 'SSM',
    'Correlation_r': central_ssm_corr,
    'P_value': central_ssm_pval
})

export_df = pd.DataFrame(export_data)
csv_path = os.path.join(output_dir, 'individual_channel_SSM_correlations.csv')
export_df.to_csv(csv_path, index=False)
print(f"\nIndividual channel statistics exported to '{csv_path}'")

# %%
# Set Seaborn style for all plots
sns.set_style("ticks")

# Convert mm to inches (180mm = 7.087 inches) - matching decay_groups_test.py
fig_width = 180 / 25.4  # 180mm in inches
fig_height = 3  # maximum height in inches

# Create 2x2 grid figure for individual channels vs SSM
fig_individual = plt.figure(figsize=(fig_width * 2, fig_height * 2))
channel_names = ['F3', 'F4', 'C3', 'C4']
channel_columns = ['F3_DECAY_RATE', 'F4_DECAY_RATE', 'C3_DECAY_RATE', 'C4_DECAY_RATE']

for idx, (channel_name, channel_col) in enumerate(zip(channel_names, channel_columns), 1):
    ax = fig_individual.add_subplot(2, 2, idx)
    valid_data = merged_df[['SSM', channel_col]].dropna()
    sns.regplot(x='SSM', y=channel_col, data=valid_data, ax=ax,
                scatter_kws={'alpha':0.6, 'color':'gray'},
                line_kws={'color':'black'},
                scatter=True, ci=95)
    
    corr_r = individual_corr_results[channel_col]['r']
    corr_p = individual_corr_results[channel_col]['p']
    
    ax.set_title(f'{channel_name}', fontsize=14)
    ax.set_ylim(-100, 100)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel('SWD', fontsize=14)
    ax.set_ylabel('Decay Rate (%)', fontsize=14)
    
    # Add Overest. and Underest. labels only to the first subplot (F3 - top left)
    if idx == 1:
        ax.text(-0.5, -90, '← Overest.', ha='left', va='bottom', fontsize=12)
        ax.text(0.5, -90, 'Underest. →', ha='right', va='bottom', fontsize=12)

#fig_individual.suptitle('Individual Channel Decay Rates vs SSM', fontsize=16, y=0.995)
fig_individual.tight_layout()
sns.despine(fig=fig_individual)
fig_path = os.path.join(output_dir, 'individual_channels_SSM_correlations.png')
fig_individual.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')

# Create combined figure for frontal and central areas vs SSM
fig_combined = plt.figure(figsize=(fig_width, fig_height))

# Left panel: Frontal area
ax1 = fig_combined.add_subplot(1, 2, 1)
valid_data = merged_df[['SSM', 'frontal_decay_rate']].dropna()
sns.regplot(x='SSM', y='frontal_decay_rate', data=valid_data, ax=ax1,
            scatter_kws={'alpha':0.6, 'color':'gray'},
            line_kws={'color':'black'},
            scatter=True, ci=95)
ax1.set_title(f'Frontal Area', fontsize=14)
ax1.set_ylim(-100, 100)
ax1.set_xlim(-0.6, 0.6)
ax1.set_ylabel('Decay Rate (%)', fontsize=14)
ax1.set_xlabel('SWD', fontsize=14)

# Add custom x-axis labels with arrows inside the plot
ax1.text(-0.5, -90, '← Overest.', ha='left', va='bottom', fontsize=10)
ax1.text(0.5, -90, 'Underest. →', ha='right', va='bottom', fontsize=10)

# Right panel: Central area
ax2 = fig_combined.add_subplot(1, 2, 2)
valid_data = merged_df[['SSM', 'central_decay_rate']].dropna()
sns.regplot(x='SSM', y='central_decay_rate', data=valid_data, ax=ax2,
            scatter_kws={'alpha':0.6, 'color':'gray'},
            line_kws={'color':'black'},
            scatter=True, ci=95)
ax2.set_title(f'Central Area', fontsize=14)
ax2.set_ylim(-100, 100)
ax2.set_xlim(-0.6, 0.6)
ax2.set_ylabel('')
ax2.set_xlabel('SWD', fontsize=14)

# Add custom x-axis labels with arrows inside the plot
ax2.text(-0.5, -90, '← Overest.', ha='left', va='bottom', fontsize=10)
ax2.text(0.5, -90, 'Underest. →', ha='right', va='bottom', fontsize=10)

fig_combined.tight_layout()
sns.despine(fig=fig_combined)
fig_path = os.path.join(output_dir, 'frontal_central_SSM_correlation.png')
fig_combined.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')

# Create combined figure for subjective (top) and objective (bottom) sleep time correlations
fig_sleep = plt.figure(figsize=(fig_width, fig_height * 2))

# A) Subjective sleep time - Top row
# Frontal decay rate vs subjective sleep time
ax1 = fig_sleep.add_subplot(2, 2, 1)
valid_data = merged_df[['Subjective_sleep_time', 'frontal_decay_rate']].dropna()
sns.regplot(x='Subjective_sleep_time', y='frontal_decay_rate', data=valid_data, ax=ax1,
            scatter_kws={'alpha':0.6, 'color':'gray'},
            line_kws={'color':'black'},
            scatter=True, ci=95)
ax1.set_title('Frontal Area', fontsize=14)
ax1.set_ylim(-100, 100)
ax1.set_xlabel('Subjective Sleep Time (min)', fontsize=14)
ax1.set_ylabel('Decay Rate (%)', fontsize=14)
ax1.text(-0.05, 1.05, 'A)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='bottom')

# Central decay rate vs subjective sleep time
ax2 = fig_sleep.add_subplot(2, 2, 2)
valid_data = merged_df[['Subjective_sleep_time', 'central_decay_rate']].dropna()
sns.regplot(x='Subjective_sleep_time', y='central_decay_rate', data=valid_data, ax=ax2,
            scatter_kws={'alpha':0.6, 'color':'gray'},
            line_kws={'color':'black'},
            scatter=True, ci=95)
ax2.set_ylim(-100, 100)
ax2.set_title('Central Area', fontsize=14)
ax2.set_xlabel('Subjective Sleep Time (min)', fontsize=14)
ax2.set_ylabel('')

# B) Objective sleep time - Bottom row
# Frontal decay rate vs objective sleep time
ax3 = fig_sleep.add_subplot(2, 2, 3)
valid_data = merged_df[['Objective_sleep_time', 'frontal_decay_rate']].dropna()
sns.regplot(x='Objective_sleep_time', y='frontal_decay_rate', data=valid_data, ax=ax3,
            scatter_kws={'alpha':0.6, 'color':'gray'},
            line_kws={'color':'black'},
            scatter=True, ci=95)
ax3.set_ylim(-100, 100)
ax3.set_title('Frontal Area', fontsize=14)
ax3.set_xlabel('Objective Sleep Time (min)', fontsize=14)
ax3.set_ylabel('Decay Rate (%)', fontsize=14)
ax3.text(-0.05, 1.05, 'B)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='bottom')

# Central decay rate vs objective sleep time
ax4 = fig_sleep.add_subplot(2, 2, 4)
valid_data = merged_df[['Objective_sleep_time', 'central_decay_rate']].dropna()
sns.regplot(x='Objective_sleep_time', y='central_decay_rate', data=valid_data, ax=ax4,
            scatter_kws={'alpha':0.6, 'color':'gray'},
            line_kws={'color':'black'},
            scatter=True, ci=95)
ax4.set_ylim(-100, 100)
ax4.set_title('Central Area', fontsize=14)
ax4.set_xlabel('Objective Sleep Time (min)', fontsize=14)
ax4.set_ylabel('')

fig_sleep.tight_layout()
sns.despine(fig=fig_sleep)
fig_path = os.path.join(output_dir, 'sleep_time_correlations.png')
fig_sleep.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')

# Show all plots
plt.show()


# Print a summary of all correlations
print("\n=== Summary of All Correlations ===")
print("-------------------------")
print("Individual Channels vs SSM:")
for channel in individual_channels:
    print(f"  {channel} vs SSM: r = {individual_corr_results[channel]['r']:.3f}, p = {individual_corr_results[channel]['p']:.4f}")
print("\nAveraged Regions vs SSM:")
print(f"  Frontal (F3+F4) vs SSM: r = {frontal_ssm_corr:.3f}, p = {frontal_ssm_pval:.4f}")
print(f"  Central (C3+C4) vs SSM: r = {central_ssm_corr:.3f}, p = {central_ssm_pval:.4f}")
print("\nOther Correlations:")
print(f"  Frontal Decay Rate vs Subjective Sleep Time: r = {frontal_subj_corr:.3f}, p = {frontal_subj_pval:.4f}")
print(f"  Central Decay Rate vs Subjective Sleep Time: r = {central_subj_corr:.3f}, p = {central_subj_pval:.4f}")
print(f"  Frontal Decay Rate vs Objective Sleep Time: r = {frontal_obj_corr:.3f}, p = {frontal_obj_pval:.4f}")
print(f"  Central Decay Rate vs Objective Sleep Time: r = {central_obj_corr:.3f}, p = {central_obj_pval:.4f}")

print("\n=== Files Generated ===")
print(f"All files saved to: {output_dir}")
print("1. individual_channel_SSM_correlations.csv - Statistical results")
print("2. individual_channels_SSM_correlations.png - 2x2 grid of individual channels")
print("3. frontal_central_SSM_correlation.png - Frontal and central areas vs SSM")
print("4. sleep_time_correlations.png - Combined subjective (A) and objective (B) sleep time correlations")
# %%
