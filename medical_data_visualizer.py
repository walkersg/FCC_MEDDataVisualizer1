import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
df.head(n=10)

# Add 'overweight' column

df['overweight'] = np.where(df['weight']/(np.square(df['height']/100))>25,1,0)
df['overweight']

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = np.where(df['cholesterol']>1,1,0)
df['gluc'] = np.where(df['gluc']>1,1,0)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    cats = ['cholesterol','gluc', 'smoke', 'alco', 'active','overweight']
    df_cat = df[cats]
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars= cats)
    df_cat
    

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=(False)).size().rename(columns={'size':'total'})

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x="variable", y = 'total', col="cardio", hue = 'value' , data= df_cat , kind= 'bar').fig
    


    # Get the figure for the output
    #fig = None


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975))]
    df_heat

    # Calculate the correlation matrix
    corr = df_heat.corr()
    #np.round(corr,3)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype = bool))
    mask



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (16,16))


    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, mask = mask, center=0, fmt='.1f',
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
