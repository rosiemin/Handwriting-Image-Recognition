from src.read_data import IAMLoadData
from src.image_processing import IAM_imageprocess
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec


def hist_dat(df,col,colr,state, show = True, path = None):
    textstr = '\n'.join((
    r'Max: {}'.format(df[col].max()),
    r'Minimum: {}'.format(df[col].min()),
    r'Mean: {}'.format(round(df[col].mean(),2)),
    r'Total N: {}'.format(len(df[col]))))


    f, axes = plt.subplots(2, 1, figsize=(20,10))
    sns.distplot(df[col], ax=axes[1], color = colr)
    sns.boxplot(df[col], ax=axes[0], color = colr, boxprops=dict(alpha=.5)).set_xlabel('')
    axes[0].set_xticklabels(labels='')
    # f.subplots_adjust(top=0.8)

    plt.suptitle(f"Distribution of {col} {state} cleaning data", fontsize = 20)
    plt.xlabel(f"{col} of image (pixels)", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5,prop=dict(fontsize=16))
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)
    f.subplots_adjust(wspace=0, hspace=0)

    if show:
        plt.show()
    if path:
        plt.savefig(path)

def n_dim_confusion_mat(y_pred, y_true, threshold):
    y_pred_bin = np.where(y_pred >= threshold, 1, 0)




if __name__ == '__main__':
    words = IAMLoadData('data/words.txt')
    df = words.load_data()

    clean = IAM_imageprocess(df)
    full_df, alpha_df = clean.subset_by_size(size_df_path = 'data/words_image_size_df.csv')

    hist_dat(full_df, 'width', 'teal', 'before', False, 'images/full_df_width_hist.png')
    hist_dat(full_df, 'height', 'blue', 'before', False, 'images/full_df_height_hist.png')
    hist_dat(alpha_df, 'width', 'teal', 'after', False, 'images/alpha_df_width_hist.png')
    hist_dat(alpha_df, 'height', 'blue', 'after', False, 'images/alpha_df_height_hist.png')

    
