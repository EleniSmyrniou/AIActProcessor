import pandas as pd
import matplotlib.pyplot as plt

csv_file = "ai_act_compliance_results.csv"
groups = [
    "Risk Management_score",
    "Logging & Traceability_score",
    "Human Oversight & Governance_score",
    "Monitoring & Maintenance_score",
    "Data Governance & Bias_score",
    "Technical Documentation & Transparency_score",
    "Accuracy, Robustness & Cybersecurity_score"]

# Read the CSV file
df = pd.read_csv(csv_file)
# rename the columns without the _score suffix
df.rename(columns={col: col.replace('_score', '') for col in df.columns if col.endswith('_score')}, inplace=True)
# also do it in the groups list
groups = [col.replace('_score', '') for col in groups]
# per group calculate the mean value
group_means = df[groups].mean()
# Create a bar plot but with a horizontal orientation
plt.figure(figsize=(10, 6))
group_means.plot(kind='barh', color='dodgerblue')
plt.xlabel('Compliance Score')
plt.ylabel('Compliance Groups')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.axvline(x=3.16, color='red', linestyle='--', label='Overall Average Score')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('compliance_groups_bar.png')

# per group histogram of the scores
for group in groups:
    plt.figure(figsize=(10, 6))
    df[group].plot(kind='hist', bins=50, color='dodgerblue', alpha=0.7)
    plt.xlabel('Compliance Score')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {group} Compliance Scores')
    plt.axvline(x=3.16, color='red', linestyle='--', label='Overall Average Score')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # set the x-axis limits to 0-5
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig(f'{group}_hist.png')
    plt.close()
# also for the overall score
plt.figure(figsize=(10, 6))
df['overall'].plot(kind='hist', bins=50, color='dodgerblue', alpha=0.7)
plt.xlabel('Overall Compliance Score')
plt.ylabel('Frequency')
plt.title('Histogram of Overall Compliance Scores')
plt.axvline(x=3.16, color='red', linestyle='--', label='Overall Average Score')
 # set the x-axis limits to 0-5
plt.xlim(0, 5)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('overall_score_hist.png')
plt.close()