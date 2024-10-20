import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import plotly.express as px
# Load the CSV data into a pandas DataFrame
df = pd.read_csv(r"C:\Users\User\Desktop\thesis\vae\experiments\results.csv")


# Normalize the data (except for the 'delta' and 'factor_vae_eval_accuracy' columns)
normalized_df = df.copy()
for column in normalized_df.columns:
    if column not in ['delta', 'factor_vae_eval_accuracy']:
        normalized_df[column] = (normalized_df[column] - normalized_df[column].min()) / (normalized_df[column].max() - normalized_df[column].min())

# Create the parallel coordinates plot
fig = px.parallel_coordinates(
    normalized_df,
    color='factor_vae_eval_accuracy',
    dimensions=['lr', 'delta', 'beta', 'lr_diff', 'factor_vae_eval_accuracy'],
    color_continuous_scale=px.colors.sequential.Viridis,
    labels={
        'lr': 'Learning Rate',
        'delta': 'Delta',
        'beta': 'Beta',
        'lr_diff': 'Learning Rate Difference',
        'factor_vae_eval_accuracy': 'Accuracy'
    },
    title='Interactive Parallel Coordinates Plot'
)

# Show the plot
fig.show()