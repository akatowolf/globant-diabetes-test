import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def count_distribution(df, column_name, figsize=(10,4)):
    """
    Plots the distribution of counts for a given column in the DataFrame.
    """
    distribution = df[column_name].value_counts()
    plt.figure(figsize=figsize)
    distribution.value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {column_name} Counts')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Number of Records')
    plt.show()

def histplot_with_violin(df, column_name, target=None, figsize=(10, 6)):
    """
    Plots the distribution and violin of numerical columns in the DataFrame.
    """
    plt.figure(figsize=figsize)
    grid = GridSpec(1, 2, width_ratios=[3, 1])
    ax_kde = plt.subplot(grid[0])
    sns.kdeplot(data=df, x=column_name, hue=target, ax=ax_kde, common_norm=False)
    ax_kde.set_title(f'KDE of {column_name}', fontsize=14)
    ax_kde.set_xlabel(column_name)
    ax_kde.set_ylabel('Density')
    ax_violin = plt.subplot(grid[1])
    sns.violinplot(data=df, y=column_name, hue=target, ax=ax_violin, orient='h', linewidth=1)
    ax_violin.set_title('Violin Plot', fontsize=14)
    ax_violin.set_xlabel('')
    ax_violin.set_ylabel(column_name)
    #ax_violin.legend_.remove()
    plt.tight_layout()
    plt.show()

def barplot(df, column_name, target=None, figsize=(10,6)):
    """
    Plots a barplot for the column in the DataFrame.
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=column_name, hue=target)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

def barplot_target(df, column_name, target, figsize=(10,6)):
    """
    Plots a barplot for the column in the DataFrame with percentages.
    """
    plt.figure(figsize=figsize)
    counts = df.groupby([column_name, target]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages.plot(kind='bar', stacked=True, colormap='viridis', figsize=figsize)
    plt.title(f'Distribution of {column_name} by {target}')
    plt.xlabel(column_name)
    plt.ylabel('Percentage')
    plt.xticks(rotation=90)
    plt.legend(title=target)
    plt.show()

def correlation(df, figsize=(10,8), cmap='coolwarm', annot=True):
    """
    Calculates the correlation matrix for the DataFrame.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def kdeplot_by_target(df, feature, target, figsize=(10, 6)):
    """
    Plots a KDE plot for the feature, grouped by the target.
    """
    plt.figure(figsize=figsize)
    sns.kdeplot(data=df, x=feature, hue=target, fill=True, common_norm=False, palette='Set2')
    plt.title(f'Distribution of {feature} by {target}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()

def plot_histogram_with_kde(data, target, feature, figsize=(10, 4)):
    """
    Plots a histogram plot for the feature, grouped by the target.
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=data, x=target, hue=feature,  element='step', stat='density')
    plt.title(f'Histograma con KDE de {target} y {feature}')
    plt.xlabel(target)
    plt.ylabel('Densidad')
    plt.show()

def scatterplot(data, target, feature1, feature2,figsize=(10, 4)):
    """
    Plots a scatter plot of two numerical features, with hue based on target.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=data, x=feature1, y=feature2, hue=target)
    plt.title(f'Scatter plot of {feature1} vs {feature2} by {target}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

def scatterplot_by(data, target, feature1, figsize=(10, 4)):
    """
    Plots a scatter plot of two numerical features, with hue based on target.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=data, x=feature1, by=target, hue=target)
    plt.title(f'Scatter plot of {feature1} by {target}')
    plt.xlabel(feature1)
    plt.show()

