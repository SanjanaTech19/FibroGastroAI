import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Model Metric Results Data
data = {
    'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM (RBF Kernel)', 'XGBoost'],
    'Accuracy': [76.19, 70.24, 70.24, 77.38, 73.81],
    'Precision': [73.00, 68.69, 67.45, 73.26, 71.05],
    'Recall': [76.19, 70.24, 70.24, 77.38, 73.81],
    'F1-Score': [74.19, 69.41, 68.62, 74.84, 72.09]
}

df_metrics = pd.DataFrame(data)

# 2. Configure Bar Plot Layout
plt.figure(figsize=(10, 6), dpi=120)
x = np.arange(len(df_metrics['Algorithm']))
width = 0.18

# 3. Plot Metric Bars
plt.bar(x - 1.5*width, df_metrics['Accuracy'], width, label='Accuracy', color='#06B6D4', edgecolor='black')
plt.bar(x - 0.5*width, df_metrics['Precision'], width, label='Precision', color='#10B981', edgecolor='black')
plt.bar(x + 0.5*width, df_metrics['Recall'], width, label='Recall', color='#F59E0B', edgecolor='black')
plt.bar(x + 1.5*width, df_metrics['F1-Score'], width, label='F1-Score', color='#8B5CF6', edgecolor='black')

# 4. Custom Formatting & Labels
plt.xlabel('Classification Algorithms', fontsize=11, fontweight='bold')
plt.ylabel('Score (%)', fontsize=11, fontweight='bold')
plt.title('Performance Comparison of 5 Classification Algorithms', fontsize=13, fontweight='bold')
plt.xticks(x, df_metrics['Algorithm'], rotation=15, fontweight='bold')
plt.ylim(50, 85)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper right', frameon=True, facecolor='white')
plt.tight_layout()

# 5. Display Plot & Save Image
plt.savefig('classification_chart_standalone.png')
print("Successfully generated and saved classification_chart_standalone.png!")
