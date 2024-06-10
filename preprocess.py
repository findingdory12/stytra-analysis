import numpy as np
import pandas as pd
import glob
from plot import *
from reframe import *
from utilities import Experiment
import shutil
import os
import argparse


def organize(directory):
    # Make folders
    for filename in os.listdir(directory):
        # print(filename)
        sub_dir = filename.split('_', 1)
        # print("sub_dir", sub_dir)
        parent_dir = directory
        path = os.path.join(parent_dir, sub_dir[0])
        try:
            os.mkdir(path)
        except OSError as error:
            # print(error)
            pass

    # Moves files into the folders made
    for filename in os.listdir(directory):
        # print("filename", filename)
        filename1 = os.path.join(directory, filename)
        # print("filename1", filename1)
        if os.path.isfile(filename1):
            # print("filename1", filename1)
            sub_dir = filename.split('_', 1)
            # print("sub_dir", sub_dir)
            parent_dir = directory
            src_path = directory
            src_path1 = os.path.join(src_path, filename)
            # print("src_path1", src_path1)
            dst_path = os.path.join(parent_dir, sub_dir[0])
            dst_path1 = os.path.join(dst_path, filename)
            # print("dst_path", dst_path)
            dst_path2 = os.path.join(dst_path, "/")
            # print("dst_path1", dst_path1)
            if ("hdf5" in filename1):
                shutil.move(src_path1, dst_path1)
            if ("json" in filename1):
                shutil.move(src_path1, dst_path1)
            if ("img" in filename1):
                shutil.move(src_path1, dst_path1)


def convert_experiment_to_csv(exp_path):
    exp = Experiment(exp_path)
    data_file = os.path.basename(exp_path)
    csv_file_path = os.path.join(exp_path, f"{data_file}.csv")
    df = exp.behavior_log
    df.to_csv(csv_file_path)
    # print(f"CSV file saved for {exp_path}")


def to_csv(parent_folder):
    # Check if the parent folder exists
    if not os.path.exists(parent_folder):
        print(f"The folder {parent_folder} does not exist.")
    else:
        # Iterate through each subfolder in the parent folder
        for subfolder_name in os.listdir(parent_folder):
            subfolder_path = os.path.join(parent_folder, subfolder_name)
            # Ensure the subfolder is indeed a directory
            if os.path.isdir(subfolder_path):
                convert_experiment_to_csv(subfolder_path)


def organize_folders(parent_dir, base_dir, num_experiments_per_trial):
    # Get list of all folders in the base directory
    folders = [f for f in os.listdir(
        base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort()  # Sort folders to ensure consistent ordering

    # Create trial directories and move folders into them
    for i in range(0, len(folders), num_experiments_per_trial):
        trial_number = i // num_experiments_per_trial + 1
        trial_folder_name = f"{parent_dir}trial{trial_number}"
        trial_folder_path = os.path.join(base_dir, trial_folder_name)
        os.makedirs(trial_folder_path, exist_ok=True)

        # Determine folders for the current trial
        for folder in folders[i:i + num_experiments_per_trial]:
            src_folder_path = os.path.join(base_dir, folder)
            dst_folder_path = os.path.join(trial_folder_path, folder)
            shutil.move(src_folder_path, dst_folder_path)


def process_trial(trial_path):
    bl_og = 10
    tap_delay = 0.060 # Can be fixed with the microphone feedback to determine time of taps
    bl = bl_og + tap_delay
    dur = 20
    num = 10
    neg_t_shift = 0.027598

    exp_id = 1
    glob.glob(trial_path)
    exp_df = []
    pre_df = []

    for i, csvfile in enumerate(glob.glob(trial_path)):
        # print(csvfile)
        x = csvfile.split(".")[0]
        csv_id = x.split("/")[-1]
        # print(csv_id)
        df_plt = reframe(csvfile, exp_id, csv_id)
        df_plt = df_plt.assign(t=df_plt['t'] + neg_t_shift)
        df_bin, arr = bin_and_cut(df_plt, bl, dur, num)
        p_pre_df = df_bin.groupby('ID').filter(lambda g: g['bin'].max() == 0)
        df_aligned = align(df_bin, arr, dur)
        exp_df.append(df_aligned)
        pre_df.append(p_pre_df)
        exp_id += 1
        # print(exp_id)

    baseline = pd.concat(pre_df, axis=0, ignore_index=True)
    trial_df = pd.concat(exp_df, axis=0, ignore_index=True)

    # Filtering
    response_attributes = pd.DataFrame()
    df_combined_by_ID = trial_df.groupby('ID')
    response_attributes = response_attributes.assign(
        head_t=df_combined_by_ID.head(1)['aligned_t'].values,
        tail_t=df_combined_by_ID.tail(1)['aligned_t'].values,
        ID=df_combined_by_ID.groups.keys()
    )

    sel_time_begin = -0.5
    sel_time_end = 0.5

    truncate_begin = sel_time_begin + 0.1
    truncate_end = sel_time_end - 0.1

    response_sel = response_attributes.query(
        "head_t <= @truncate_begin & tail_t >= @truncate_end")
    df_combined_sel = trial_df.loc[trial_df['ID'].isin(response_sel['ID'])]

    df_combined_selrows = df_combined_sel.query(
        "aligned_t > @sel_time_begin & aligned_t < @sel_time_end"
    )

    # assign delta t
    deltaT_df = df_combined_selrows.groupby('ID').apply(
        lambda g: g['aligned_t'].diff()
    ).reset_index().set_index('level_1')
    deltaT_df = deltaT_df.sort_index()
    df_combined_selrows = df_combined_selrows.assign(
        delta_t=deltaT_df['aligned_t']
    )
    # down sampling
    time_adj_thres = 10  # ms
    time_adj_bins = np.arange(sel_time_begin*1000, 500, time_adj_thres)
    time_adj_labels = (time_adj_bins[1:] + time_adj_bins[:-1])/2

    # soft code delta_t
    df_filtered = df_combined_selrows.groupby('ID').filter(
        lambda x: (x['aligned_t'].max() > truncate_end) & (x['aligned_t'].min() < truncate_begin) & (
            x['delta_t'].max() <= 0.01))

    # df_filtered = df_filtered.assign(
    #     time_adj=pd.cut(df_filtered['aligned_ms'], bins=time_adj_bins, labels=time_adj_labels)
    # )

    return df_filtered, baseline


def habituation(trial_path):
    bl_og = 194.6
    tap_delay = 0.060
    bl = bl_og + tap_delay
    dur = 1
    num = 15
    neg_t_shift = 0.027598

    exp_id = 1
    glob.glob(trial_path)
    exp_df = []

    for i, csvfile in enumerate(glob.glob(trial_path)):
        # print(csvfile)
        x = csvfile.split(".")[0]
        csv_id = x.split("/")[-1]
        # print(csv_id)
        df_plt = reframe(csvfile, exp_id, csv_id)
        df_plt = df_plt.assign(t=df_plt['t'] + neg_t_shift)
        df_bin = bin_and_cut(df_plt, bl, dur, num)[0]
        df_hab_bin = df_bin.groupby('ID').filter(lambda g: g['bin'].max() > 0)
        exp_df.append(df_hab_bin)  # Append DataFrame to the list

        exp_id += 1
        # print(exp_id)

    hab_df = pd.concat(exp_df, axis=0, ignore_index=True)

    return hab_df


def process_trials(parent_folder):
    for trial_folder_path in glob.glob(os.path.join(parent_folder, '*')):
        trial_path = os.path.join(trial_folder_path, '*/*.csv')
        df_filtered, baseline = process_trial(trial_path)
        hab_df = habituation(trial_path)

        # Save the df_filtered DataFrame in respective trial folder
        output_file_path = os.path.join(trial_folder_path, 'df_filtered.csv')
        df_filtered.to_csv(output_file_path, mode='w')

        # Save the baseline DataFrame in the respective trial folder
        parent_folder = os.path.dirname(os.path.dirname(trial_path))
        baseline_file_path = os.path.join(parent_folder, 'baseline.csv')
        baseline.to_csv(baseline_file_path, index=False)

        # Save the habituation DataFrame in the respective trial folder
        hab_output_file_path = os.path.join(trial_folder_path, 'hab_df.csv')
        hab_df.to_csv(hab_output_file_path, mode='w')


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process and organize directories.")
    parser.add_argument("directory", type=str,
                        help="The path to the directory to be processed.")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of experiments per trial for organizing folders (default is 3).")

    args = parser.parse_args()
    parent = str(args.directory).split('/')[-1]

    for subdir in os.listdir(args.directory):
        subdir_path = os.path.join(args.directory, subdir)
        if os.path.isdir(subdir_path):  # Ensure it's a directory
            organize(subdir_path)
            to_csv(subdir_path)
            organize_folders(parent, subdir_path, args.trials)
            process_trials(subdir_path)


if __name__ == "__main__":
    main()
