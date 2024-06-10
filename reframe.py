#reframe function (csv -> workable dataframe)
#%%
import pandas as pd
import numpy as np 

def reframe(csvfile, exp_id, csv_id): 
    '''
    Function: reframe
    Arguments:
    - csvfile: str - path to the csv file
    - exp_id: int - experiment ID
    - csv_id: str - ID of the CSV file
    Returns:
    - df_plt: pandas.DataFrame - a DataFrame with the reformatted data, 
    including additional columns such as fish_id, vel, absvtheta, abstheta, ms, csv_id, and exp_id.
    '''
    df = pd.read_csv(csvfile)
    tmp = list(df.columns)[-3]
    df = df.iloc[:,1:]

    #warp function starts here
    max_fish = int(tmp[1:tmp.find('_')])
    df["id"] = df.index

    # rename columns for wide_to_long convertion
    stubnames = ['x','vx','y','vy','theta','vtheta','theta-00','theta-01']
    new_colnames = []
    for i in range(max_fish+1):
        new_colnames.extend([f'{name}_f{i:02d}' for name in stubnames])
    new_colnames.extend(['biggest_area', 't', 'id'])
    df.columns = new_colnames
    dfl = pd.wide_to_long(df, stubnames=stubnames, i="id", j="fish",sep='_',suffix='\w+')
    dfl.reset_index(inplace=True)

    df_plt = dfl.set_index('id')
    df_plt = df_plt.dropna(subset=['x'])
    df_plt = dfl.set_index('id')
    df_plt = df_plt.assign(vel = np.linalg.norm(df_plt[['vx','vy']],axis=1),
                        absvtheta=np.abs(df_plt['vtheta']),
                        abstheta=np.abs(df_plt['theta']),
                        fish_id = df_plt.groupby('fish').ngroup(),
                        ms=np.multiply(df_plt['t'], 1000),
                        csv_id = csv_id, 
                        exp_id = exp_id)
    df_plt = df_plt.dropna(subset=['x'])
    df_plt = df_plt.dropna()
    df_plt.reset_index(inplace=True)

    return df_plt

# %%

#User input for baseline, duration, and number of reps of stimulus
def bin_and_cut(df_plt, bl, dur, num):

    arr = list(np.arange(0,num) * dur + bl)
    xpos = np.insert(arr, 0, 0)
    xpos = list(xpos)
    print(xpos)

    df_bin = df_plt.assign(
        # interval = df_plt.groupby('fish_id')[['t']].transform(lambda x: pd.cut(x, bins=xpos).astype('str')),
        bin = pd.cut(df_plt['t'],xpos, labels = np.arange(0,num)))
    df_bin = df_bin.dropna()
    df_bin['bin'] = df_bin["bin"].astype(np.int8)
    df_bin['res_id'] = (np.multiply(df_bin['fish_id'], 100) + df_bin['bin'])

    df_bin = df_bin.assign(ID=df_bin[['res_id','exp_id']].apply(
        lambda row: '_'.join([str(each) for each in row]),axis=1))
    return df_bin,arr


# %%
def align(df_bin, arr, dur):
    arr = list(arr)
    print(arr)
    align = [((arr[i] + arr[i+1]) / 2) for i in range(len(arr) - 1)]
    print(align)

    df_aligned = df_bin

    interval_half_width = np.divide(dur, 2) # half of full interval length in seconds
    for stim in align:
        mask = (df_aligned['t'] >= (stim - interval_half_width)) & (df_aligned['t'] < (stim + interval_half_width))
        df_aligned.loc[mask, 'aligned_t'] = df_aligned.loc[mask, 't'] - stim + 5
        #Check the min and max of the aligned times. should be -0.5 to 15 for 20s 
        print(stim, df_aligned.loc[mask, 'aligned_t'].min(), df_aligned.loc[mask, 'aligned_t'].max())
    df_aligned = df_aligned.dropna()
    df_aligned = df_aligned.assign(aligned_ms=np.multiply(df_aligned['aligned_t'], 1000))

    return df_aligned

