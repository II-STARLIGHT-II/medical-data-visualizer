import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the dataset from 'medical_examination.csv' file
df = pd.read_csv('medical_examination.csv')

# Step 2: Calculate the BMI and create the 'overweight' column
# Formula for BMI = weight / (height/100)^2. If BMI > 25, overweight = 1, otherwise = 0
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)

# Drop the BMI column after creating 'overweight'
df.drop(columns=['BMI'], inplace=True)

# Step 3: Normalize cholesterol and gluc data. 
# Set 0 for good (<= 1) and 1 for bad (> 1)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

# Step 4-9: Draw the Categorical Plot
def draw_cat_plot():
    # Step 5: Melt the dataframe. 'id_vars' is 'cardio', 'value_vars' are the categorical columns
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Step 6: Group and reformat the data to show counts of features split by 'cardio'
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat.rename(columns={'size': 'total'}, inplace=True)

    # Step 7: No modification needed here

    # Step 8: Draw the categorical plot using sns.catplot
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')

    # The correct figure for catplot is accessed through g.fig
    fig = g.fig

    # Step 9: Save and return the plot
    fig.savefig('catplot.png')
    return fig

# Step 10-16: Draw the Heat Map
def draw_heat_map():
    # Step 11: Clean the data by filtering out incorrect entries
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Step 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 13: Generate a mask for the upper triangle of the heatmap
    mask = np.triu(corr)

    # Step 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 15: Draw the heatmap using seaborn
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=0.5, ax=ax)

    # Step 16: Save and return the figure
    fig.savefig('heatmap.png')
    return fig
