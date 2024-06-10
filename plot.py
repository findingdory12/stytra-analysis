#%%
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 


#%%
def distribution_binned_average(df, by_col, bin_col, bin):
    '''takes a dataframe, bin by by_col value & calculate averaged bin_col value
    Args:
        df (DataFrmae):
        by_col (string): column to be binned by
        bin_col (string): column to be averaged
        bin (array): an array of bins (bin = np.arange(min,max,bin_width))
    Returns:
        Dataframe
    '''
    df = df.sort_values(by=by_col)
    bins = pd.cut(df[by_col], list(bin))
    grp = df.groupby(bins)
    df_out = grp[[by_col,bin_col,'condition']].mean()
    return df_out

#%%
def plot_response_by_id(df):
    grouped = df.groupby('ID')
    plots = [] # initialize list to store plots
    count = 0 
    for i, (k, v) in enumerate(grouped):
        plt.figure(figsize=(8,5))
        plt.plot(v["aligned_t"], v["absvtheta"])
        plt.xlabel("Time [s]")
        plt.ylabel('Angular Velocity [rad/s]')
        plt.title(k) #ID as title 
        plt.ylim(0, 1.5)
        plots.append(plt.gcf()) # append current figure to list
        count = count + 1
    print("Number of responses:", count)
    return plots # return list of figures
# %%
