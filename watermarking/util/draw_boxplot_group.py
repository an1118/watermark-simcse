import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

import pdb

result = r"/mnt/data2/lian/projects/watermark/adaptive-text-watermark-yepeng/outputs/eval_ppl/watermark-8b-all.csv"
df = pd.read_csv(result)
df = df[['ppl_2step-unwatermarked','ppl_2step','ppl_2step-2sent-nomintoken-unwatermark','ppl_2step-2sent-nomintoken', 'ppl_2step-2models-unwatermark','ppl_2step-2models']]

# mapping = {'ppl_2step': ('ppl_2step-unwatermarked','ppl_2step'), \
#            'ppl_2step-2sent-nomintoken': ('ppl_2step-2sent-nomintoken-unwatermark','ppl_2step-2sent-nomintoken'), \
#             'ppl_2step-2models': ('ppl_2step-2models-unwatermark','ppl_2step-2models')}

# Identify pairs and prepare the data for plotting
pairs = {}
for col in df.columns:
    if 'unwatermark' in col:
        base_name = col.split('-')[:-1]
        base_name = '-'.join(base_name)
        if base_name in df.columns:
            pairs[base_name] = (df[col], df[base_name])

# Prepare data for boxplot
plot_data = []
plot_labels = []
plot_colors = []

for base_name, (unwatermarked, original) in pairs.items():
    plot_data.append(unwatermarked)
    plot_labels.append(base_name)
    plot_colors.append('unwatermarked')
    
    plot_data.append(original)
    plot_labels.append(base_name)
    plot_colors.append('watermarked')

# Flatten the data and create a combined DataFrame for plotting
flat_data = pd.DataFrame({
    'Value': [val for series in plot_data for val in series],
    'Metric': [label for label, series in zip(plot_labels, plot_data) for _ in series],
    'Color': [color for color, series in zip(plot_colors, plot_data) for _ in series],
})

# Create boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=flat_data, x='Metric', y='Value', hue='Color', dodge=True)

# Customize plot
plt.xticks(rotation=45)
plt.title('Boxplot for different settings')
plt.xlabel('Settings')
plt.ylabel('ppl')

plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"outputs/boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
print('Complete!')

