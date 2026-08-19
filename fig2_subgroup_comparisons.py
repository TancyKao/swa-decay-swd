# decay_groups_test.py
"""
Decay Groups Analysis Script
This script analyzes the decay rates of EEG signals across different participant groups. 
It performs statistical comparisons and visualizations to identify group differences.
The script analyzes three main metrics:
1. Decay rates (percentage) for peak and mean values
2. Rising rates (slope in 1/min)
3. Rising time (time to peak in minutes)
For each metric, the script:
- Loads data from Excel files
- Performs one-way ANOVA analysis with Tukey post-hoc tests
- Creates visualizations (boxplots with individual data points)
- Tests statistical assumptions (normality and homoscedasticity)
- Generates summary statistics
Key functions:
- perform_anova_analysis_pg(): Performs ANOVA and post-hoc tests using pingouin
Outputs:
- Statistical test results (ANOVA tables, post-hoc comparisons)
- Summary tables of results
- Visualization plots saved as PNG files:
    - peak_comparison_by_group.png
    - mean_comparison_by_group.png
    - RisingTime_comparison_by_group.png
Dependencies:
- pandas, seaborn, matplotlib, scipy, pingouin, numpy
"""

# %%
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
import numpy as np
import os
from statannotations.Annotator import Annotator


from config import DATA_DIR, OUT_DIR
file_path = os.path.join(DATA_DIR, 'DeltaCycle_DecayRate2R.xlsx')
output_path = OUT_DIR

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load the decay rate data
data = pd.read_excel(file_path,sheet_name='decay rate')

# Print the column names to verify data structure
print("Columns in the dataset:", data.columns.tolist())
print("Shape of data:", data.shape)
print("Group column unique values:", data['Group'].unique())

data['Frontal_3peaks_decay'] = data[['F3_3peaks', 'F4_3peaks']].mean(axis=1)
data['Central_3peaks_decay'] = data[['C3_3peaks', 'C4_3peaks']].mean(axis=1)

data['Frontal_mean_decay'] = data[['F3_mean', 'F4_mean']].mean(axis=1)
data['Central_mean_decay'] = data[['C3_mean', 'C4_mean']].mean(axis=1)


# %%
# Function to perform ANOVA and post-hoc tests using pingouin
def perform_anova_analysis_pg(data, variable_name, group_column='Group'):
    """
    Perform one-way ANOVA and post-hoc tests using pingouin
    """
    print(f"\n{'='*50}")
    print(f"ANOVA Analysis for {variable_name}")
    print(f"{'='*50}")
    
    # Remove any missing values
    clean_data = data.dropna(subset=[variable_name, group_column]).copy()
    
    # Perform one-way ANOVA using pingouin
    anova_result = pg.anova(dv=variable_name, between=group_column, data=clean_data, detailed=True)
    
    print(f"\nANOVA Results for {variable_name}:")
    print(anova_result)
    
    # Extract p-value
    p_value = anova_result['p-unc'].iloc[0]
    
    # Post-hoc pairwise comparisons using Tukey HSD
    if p_value < 0.05:
        print(f"\nSignificant ANOVA result (p < 0.05). Performing post-hoc pairwise comparisons:")
        posthoc = pg.pairwise_tukey(data=clean_data, dv=variable_name, between=group_column)
        print(posthoc)
    else:
        print(f"\nNon-significant ANOVA result (p >= 0.05). No post-hoc test needed.")
        posthoc = None
    
    # Summary statistics by group
    summary_stats = clean_data.groupby(group_column)[variable_name].describe()
    print(f"\nSummary Statistics for {variable_name} by {group_column}:")
    print(summary_stats)
    
    # Effect size (eta-squared)
    eta_squared = anova_result['np2'].iloc[0]  # partial eta-squared
    print(f"\nEffect size (partial eta-squared): {eta_squared:.4f}")

    return anova_result, posthoc, clean_data

# %% Perform ANOVA for all decay rate variables
variables = ['Frontal_3peaks_decay', 'Central_3peaks_decay', 'Frontal_mean_decay', 'Central_mean_decay']
anova_results = {}
posthoc_results = {}

for var in variables:
    anova_table, posthoc_table, clean_data = perform_anova_analysis_pg(data, var)
    anova_results[var] = anova_table
    posthoc_results[var] = posthoc_table

# %% Plot decay rate
# Plot 1: Frontal Peak and Central Peak subplots
# Convert mm to inches (180mm = 7.087 inches)
fig_width = 180 / 25.4  # 180mm in inches
fig_height = 3  # maximum height in inches
group_order = ['OverEst', 'OptEst', 'UnderEst']
group_colors = {'OverEst': '#8da0cb', 'OptEst': '#66c2a5', 'UnderEst': '#fc8d62'}
palette = [group_colors[g] for g in group_order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# Subplot 1: Frontal Peak
sns.boxplot(x='Group', y='Frontal_3peaks_decay', data=data, ax=ax1, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Frontal_3peaks_decay', data=data, color='black', alpha=0.5, ax=ax1, order=group_order)
ax1.set_ylim(-40, 100)
ax1.set_title('Frontal Area', fontsize=14)
ax1.set_ylabel('Decay Rate (%)', fontsize=14)
ax1.set_xlabel('', fontsize=14)

# Subplot 2: Central Peak
sns.boxplot(x='Group', y='Central_3peaks_decay', data=data, ax=ax2, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Central_3peaks_decay', data=data, color='black', alpha=0.5, ax=ax2, order=group_order)
ax2.set_ylim(-40, 100)
ax2.set_title('Central Area', fontsize=14)
ax2.set_ylabel('Decay Rate (%)', fontsize=14)
ax2.set_xlabel('', fontsize=14)

plt.tight_layout()
sns.despine(fig=fig)
plt.savefig(os.path.join(output_path, 'peak_comparison_by_group.png'), dpi=600, bbox_inches='tight')
plt.show()


# Plot 2: Frontal Mean and Central Mean subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# Subplot 1: Frontal Mean
sns.boxplot(x='Group', y='Frontal_mean_decay', data=data, ax=ax1, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Frontal_mean_decay', data=data, color='black', alpha=0.5, ax=ax1, order=group_order)
ax1.set_ylim(-50, 120)
ax1.set_title('Frontal Area', fontsize=14)
ax1.set_ylabel('Decay Rate (%)', fontsize=14)
ax1.set_xlabel('', fontsize=14)
# Add significance annotations
pairs_frontal = [("OverEst", "UnderEst"), ("OptEst", "UnderEst")]
annotator1 = Annotator(ax1, pairs_frontal, data=data, x='Group', y='Frontal_mean_decay', order=group_order)
annotator1.configure(text_format='star', loc='inside', fontsize=12)
annotator1.set_pvalues([0.013, 0.017])  # Replace with actual p-values from posthoc_results
annotator1.annotate()


# Subplot 2: Central Mean
sns.boxplot(x='Group', y='Central_mean_decay', data=data, ax=ax2, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Central_mean_decay', data=data, color='black', alpha=0.5, ax=ax2, order=group_order)
ax2.set_ylim(-50, 120)
ax2.set_title('Central Area', fontsize=14)
ax2.set_ylabel('Decay Rate (%)', fontsize=14)
ax2.set_xlabel('', fontsize=14)
# Add significance annotations
pairs_frontal = [("OverEst", "UnderEst"), ("OptEst", "UnderEst")]
annotator2 = Annotator(ax2, pairs_frontal, data=data, x='Group', y='Frontal_mean_decay', order=group_order)
annotator2.configure(text_format='star', loc='inside', fontsize=12)
annotator2.set_pvalues([0.012, 0.037])  # Replace with actual p-values from posthoc_results
annotator2.annotate()


plt.tight_layout()
sns.despine(fig=fig)
plt.savefig(os.path.join(output_path, 'mean_comparison_by_group.png'), dpi=600, bbox_inches='tight')
plt.show()

# Create a summary table of all ANOVA results
print(f"\n{'='*80}")
print("SUMMARY OF ALL ANOVA RESULTS")
print(f"{'='*80}")

summary_df = pd.DataFrame({
    'Variable': variables,
    'F-statistic': [anova_results[var]['F'].iloc[0] for var in variables],
    'p-value': [anova_results[var]['p-unc'].iloc[0] for var in variables],
    'Partial η²': [anova_results[var]['np2'].iloc[0] for var in variables],
    'Significant': ['Yes' if anova_results[var]['p-unc'].iloc[0] < 0.05 else 'No' for var in variables]
})


print(summary_df.to_string(index=False))

# Print post-hoc results summary for significant ANOVAs
print(f"\n{'='*80}")
print("POST-HOC RESULTS SUMMARY")
print(f"{'='*80}")

for var in variables:
    if anova_results[var]['p-unc'].iloc[0] < 0.05 and posthoc_results[var] is not None:
        print(f"\nPost-hoc results for {var}:")
        significant_pairs = posthoc_results[var][posthoc_results[var]['p-tukey'] < 0.05]
        if len(significant_pairs) > 0:
            print("Significant pairwise differences:")
            for idx, row in significant_pairs.iterrows():
                print(f"  {row['A']} vs {row['B']}: p = {row['p-tukey']:.4f}")
        else:
            print("  No significant pairwise differences found")

# Optional: Normality and homoscedasticity tests
print(f"\n{'='*80}")
print("ASSUMPTION TESTING")
print(f"{'='*80}")

for var in variables:
    print(f"\nAssumption tests for {var}:")
    clean_data = data.dropna(subset=[var, 'Group'])
    
    # Test normality for each group using Shapiro-Wilk test
    print("Normality tests (Shapiro-Wilk) by group:")
    for group in clean_data['Group'].unique():
        group_data = clean_data[clean_data['Group'] == group][var]
        if len(group_data) >= 3:  # Shapiro-Wilk requires at least 3 observations
            normality = pg.normality(group_data)
            print(f"  {group}: W = {normality['W'].iloc[0]:.4f}, p = {normality['pval'].iloc[0]:.4f}")
    
    # Test homogeneity of variances using Levene's test
    homoscedasticity = pg.homoscedasticity(data=clean_data, dv=var, group='Group')
    print(f"Homoscedasticity (Levene's test): W = {homoscedasticity['W'].iloc[0]:.4f}, p = {homoscedasticity['pval'].iloc[0]:.4f}")

# %% compute rising rate
data2 = pd.read_excel(file_path,sheet_name='RisingRate')

# Print the column names to verify data structure
print("Columns in the dataset:", data2.columns.tolist())
print("Shape of data:", data2.shape)
print("Group column unique values:", data2['Group'].unique())

data2['Frontal_mean_rising'] = data2[['F3_rising_onset', 'F4_rising_onset']].mean(axis=1)
data2['Central_mean_rising'] = data2[['C3_rising_onset', 'C4_rising_onset']].mean(axis=1)

# Perform ANOVA for rising rate variables
variables = ['Frontal_mean_rising', 'Central_mean_rising']
anova_results = {}
posthoc_results = {}


for var in variables:
    anova_table, posthoc_table, clean_data = perform_anova_analysis_pg(data2, var)
    anova_results[var] = anova_table
    posthoc_results[var] = posthoc_table

# %% Plot rising rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# Subplot 1: Frontal rising rate
sns.boxplot(x='Group', y='Frontal_mean_rising', data=data2, ax=ax1, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Frontal_mean_rising', data=data2, color='black', alpha=0.5, ax=ax1, order=group_order)
ax1.set_ylim(0, 0.02)
ax1.set_title('Frontal Area', fontsize=14)
ax1.set_ylabel('Slope (1/min)', fontsize=14)
ax1.set_xlabel('', fontsize=14)


# Subplot 2: Central rising rate
sns.boxplot(x='Group', y='Central_mean_rising', data=data2, ax=ax2, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Central_mean_rising', data=data2, color='black', alpha=0.5, ax=ax2, order=group_order)
ax2.set_ylim(0, 0.02)
ax2.set_title('Central Area', fontsize=14)
ax2.set_ylabel('Slope (1/min)', fontsize=14)
ax2.set_xlabel('', fontsize=14)

plt.tight_layout()
sns.despine(fig=fig)
plt.savefig(os.path.join(output_path, 'RisingRate_comparison_by_group.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% compute rising time
data3 = pd.read_excel(file_path,sheet_name='RisingTime')


# Print the column names to verify data structure
print("Columns in the dataset:", data3.columns.tolist())
print("Shape of data:", data3.shape)
print("Group column unique values:", data3['Group'].unique())

data3['Frontal_mean_risingtime'] = data3[['F3', 'F4']].mean(axis=1)
data3['Central_mean_risingtime'] = data3[['C3', 'C4']].mean(axis=1)

# Perform ANOVA for rising time variables
variables = ['Frontal_mean_risingtime', 'Central_mean_risingtime']
anova_results = {}
posthoc_results = {}


for var in variables:
    anova_table, posthoc_table, clean_data = perform_anova_analysis_pg(data3, var)
    anova_results[var] = anova_table
    posthoc_results[var] = posthoc_table

# %% Plot rising time
# Plot 1: Frontal rising time and Central rising time subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# Subplot 1: Frontal rising time
sns.boxplot(x='Group', y='Frontal_mean_risingtime', data=data3, ax=ax1, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Frontal_mean_risingtime', data=data3, color='black', alpha=0.5, ax=ax1, order=group_order)
ax1.set_ylim(0, 150)
ax1.set_title('Frontal Area', fontsize=14)
ax1.set_ylabel('Time to Peak (min)', fontsize=14)
ax1.set_xlabel('', fontsize=14)


# Subplot 2: Central rising time
sns.boxplot(x='Group', y='Central_mean_risingtime', data=data3, ax=ax2, order=group_order, palette=palette)
sns.stripplot(x='Group', y='Central_mean_risingtime', data=data3, color='black', alpha=0.5, ax=ax2, order=group_order)
ax2.set_ylim(0, 150)
ax2.set_title('Central Area', fontsize=14)
ax2.set_ylabel('Time to Peak (min)', fontsize=14)
ax2.set_xlabel('', fontsize=14)

plt.tight_layout()
sns.despine(fig=fig)
plt.savefig(os.path.join(output_path, 'RisingTime_comparison_by_group.png'), dpi=600, bbox_inches='tight')
plt.show()

# %%
