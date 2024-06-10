# %%
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#%%
def get_data_dir():
    data_dic = {
        # Format should be
        # "Condition": "path/to/folder,
        # For example
        # "sibs": "users/rhoshinir/Downloads/240304_f0/sibs"
        # Add more conditions here as needed
        "sibs": ,
        "tau": ,
    }
    return data_dic

# Define the columns to identify condition and trial_id
trial_id_col = "trial_id"

# Initialize an empty DataFrame to store concatenated data
concatenated_df = pd.DataFrame()

# Get the data dictionary
data_dic = get_data_dir()

# Iterate over conditions
for condition, parent_folder in data_dic.items():
    for trial_folder in glob.glob(os.path.join(parent_folder, "*")):
        print(f"Processing folder: {parent_folder}, Condition: {condition}")

        trial_id = os.path.basename(trial_folder)

        # look for df_filtered.csv in the trial folder
        df_filtered_path = os.path.join(trial_folder, "df_filtered.csv")

        # check if file exists
        if os.path.isfile(df_filtered_path):
            df_filtered = pd.read_csv(df_filtered_path)
            df_filtered[trial_id_col] = trial_id
            df_filtered["condition"] = condition
            print(f"Reading file: {df_filtered_path}")

            df_filtered = df_filtered.assign(
                unique_res_id=(
                    df_filtered["condition"]
                    + "_"
                    + df_filtered["csv_id"].astype(str)
                    + "_"
                    + df_filtered["trial_id"]
                    + "_"
                    + df_filtered["ID"].astype(str)
                )
            )

            concatenated_df = pd.concat(
                [concatenated_df, df_filtered], ignore_index=True
            )


# %%
group_counts = concatenated_df.groupby("condition")["unique_res_id"].nunique()
group_counts

# %%
sel_time_begin = -0.5
time_adj_thres = 10  # ms
time_adj_bins = np.arange(sel_time_begin * 1000, 500 + time_adj_thres, time_adj_thres)
time_adj_labels = (time_adj_bins[1:] + time_adj_bins[:-1]) / 2

concatenated_df = concatenated_df.assign(
    time_adj=pd.cut(
        concatenated_df["aligned_ms"], bins=time_adj_bins, labels=time_adj_labels
    )
)
numeric_columns = concatenated_df.select_dtypes(include="number").columns
downsampled = (
    concatenated_df.groupby(["condition", "unique_res_id", "time_adj"])
    .agg(
        {
            "trial_id": "first",  # Include 'trial_id' in df without agg function
            **{
                col: "mean" for col in numeric_columns
            },  # Apply mean aggregation to all numeric columns
        }
    )
    .reset_index()
)
# %%
plot_absvtheta = sns.lineplot(data=downsampled, x="time_adj", y="absvtheta", hue="condition")
plot_absvtheta.set_xlim(-50, 200)
plt.savefig("absvtheta_avg.svg")

#%%
plot_vtheta = sns.lineplot(data=downsampled, x="time_adj", y="vtheta", hue="condition")
plot_vtheta.set_xlim(-50, 200)
plt.savefig("vtheta_avg.svg")

# %%
downsampled["time_adj"] = downsampled["time_adj"].astype(int)

res_window_downsampled = downsampled[
    (downsampled["time_adj"] <= 200) * (downsampled["time_adj"] > 0)
].reset_index()

# %%
# FIND JUST RESPONSES IN DOWNSAMPLED DATA
par = "absvtheta"  # Parameter to analyze
par_max_combined = pd.DataFrame()
for cond in res_window_downsampled["condition"].unique():
    that_df = res_window_downsampled.query("condition == @cond")
    that_df.dropna(subset=[par], inplace=True)  # Drop rows with NaN in the 'par' column
    grouped = that_df.groupby(
        "unique_res_id"
    )  # Filter the grouped object by the condition
    max_par_idx = grouped[
        par
    ].idxmax()  # Get the index of the maximum value for each group (in a loc way)
    max_par = that_df.loc[max_par_idx, :]
    max_par = max_par.assign(condition=cond)
    par_max_combined = pd.concat([par_max_combined, max_par], ignore_index=True)

# Plots distribution of times of maximum parameters
bin_width = 3
num_bins = 25
par_plot = sns.histplot(x="aligned_ms", data=par_max_combined, hue="condition", element='poly', stat='density',
                        binwidth=bin_width, bins=num_bins)
par_plot.set_xlim(0, 200)
par_plot.set(xlabel="Time [ms]", title="distribution of max " + par)
plt.savefig('maxangveldist.svg')


# %%
# new dataframes for responses and no responses
no_response_max = par_max_combined.loc[par_max_combined["absvtheta"] < 0.04]

responses_max = par_max_combined.loc[par_max_combined["absvtheta"] >= 0.04]

# %%
# Response rate per condition per trial - box plot
# box plot code below...:
response_rate_per_trial = (
    responses_max.groupby(["condition", "trial_id"]).size()
    / par_max_combined.groupby(["condition", "trial_id"]).size()
)
response_rate_per_trial = response_rate_per_trial.reset_index(name="response_rate")

sp = sns.swarmplot(data=response_rate_per_trial, x="condition", y="response_rate")
bp = sns.boxplot(data=response_rate_per_trial, x="condition", y="response_rate")
sp.set_ylim(0.5,1.1)
bp.set_ylim(0.5,1.1)
plt.title("React Rate per Trial")
plt.xlabel("Condition")
plt.ylabel("Response Rate")
plt.savefig('responseratepertrial.svg')



# %%
plt.figure(figsize=(14, 8))
sns.barplot(
    data=response_rate_per_trial, x="trial_id", y="response_rate", hue="condition"
)
plt.title("Response Rate Comparison Between conditions for each trial")
plt.xlabel("Trial ID")
plt.ylabel("Response Rate")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("reactratepertrial.svg")
# %%

# maybe ugly facet bar chart:

response_rate_per_trial = (
    responses_max.groupby(["condition", "trial_id"]).size()
    / par_max_combined.groupby(["condition", "trial_id"]).size()
)
response_rate_per_trial = response_rate_per_trial.reset_index(name="response_rate")

# Calculate no-response rate per trial for each condition
no_response_rate_per_trial = 1 - response_rate_per_trial["response_rate"]

# Combine response rate and no-response rate into a single DataFrame
response_df = pd.concat(
    [response_rate_per_trial, no_response_rate_per_trial.rename("no_response_rate")],
    axis=1,
)

# Melt the DataFrame to have separate columns for 'response' and 'no_response'
response_df_melted = response_df.melt(
    id_vars=["condition", "trial_id"], var_name="response_type", value_name="rate"
)

# Create FacetGrid with bar plots for response and no-response rates
g = sns.FacetGrid(
    response_df_melted, col="condition", hue="response_type", col_wrap=3, sharey=False
)
g.map_dataframe(sns.barplot, x="trial_id", y="rate")
g.set_axis_labels("Trial ID", "Rate")
g.add_legend()
plt.suptitle("Response Rate vs No-Response Rate per Trial by Condition", y=1.05)
plt.show()
# %%
response_downsampled = downsampled.loc[
    downsampled["unique_res_id"].isin(responses_max["unique_res_id"])
]

# %%
noresponse_downsampled = downsampled.loc[
    downsampled["unique_res_id"].isin(no_response_max["unique_res_id"])
]
nores = sns.lineplot(data=noresponse_downsampled, x="time_adj", y="absvtheta", hue="condition")
nores.set_ylim(0, 0.1)
plt.savefig("noresponse.svg")
# %%
min_idx_df = responses_max.loc[responses_max["vtheta"] <= -0.04]
max_idx_df = responses_max.loc[responses_max["vtheta"] >= 0.04]

negative_peak_ids = min_idx_df["unique_res_id"].unique()
positive_peak_ids = max_idx_df["unique_res_id"].unique()

# DOWNSAMPLED df
response_downsampled["vtheta_adj"] = response_downsampled["vtheta"]
response_downsampled.loc[
    response_downsampled["unique_res_id"].isin(negative_peak_ids), "vtheta_adj"
] *= -1

# %%
# Original df
response_og = concatenated_df.loc[
    concatenated_df["unique_res_id"].isin(responses_max["unique_res_id"])
]

response_og["vtheta_adj"] = response_og["vtheta"]
response_og.loc[
    response_og["unique_res_id"].isin(negative_peak_ids), "vtheta_adj"
] *= -1

# %%
vtheta_adj_plot = sns.lineplot(data=response_downsampled, x="time_adj", y="vtheta_adj", hue="condition")
vtheta_adj_plot.set_xlim(-50,200)
vtheta_adj_plot.set_ylim(-0.03,0.1)

#%%
averaged_trial = response_downsampled.groupby(['condition', 'trial_id', 'time_adj'])['vtheta_adj'].mean().reset_index()
plot = sns.lineplot(data=averaged_trial, x = 'time_adj', y='vtheta_adj', hue = 'condition')
plot.set_xlim(-50,200)

# %%
par_plot = sns.histplot(
    x="aligned_ms", data=responses_max, hue="condition", stat="density", element="poly"
)
par_plot.set_xlim(0, 200)
par_plot.set(xlabel="Time [ms]", title="distribution of max " + par)


# %%
group_counts = response_og.groupby("condition")["unique_res_id"].nunique()
group_counts

# %%
# Add first derivative column
response_downsampled["vtheta_adj_diff"] = response_downsampled.groupby("unique_res_id")[
    "vtheta_adj"
].diff()


# %%
plot = sns.lineplot(
    data=response_downsampled, x="time_adj", y="vtheta_adj_diff", hue="condition"
)
plot.set_xlim(-100, 100)
plot.set_ylim(0, 0.02)
plt.savefig('vtheta_adj_diff.svg')

# %%
# find latencies
threshold = 0.009

lat_window = response_og[
    (response_og["aligned_ms"] >= 0) & (response_og["aligned_ms"] <= 200)
]

lat_window["vtheta_adj_diff"] = lat_window.groupby("unique_res_id")["vtheta_adj"].diff()


def first_occurrence_index(df):
    above_threshold = df["vtheta_adj_diff"] > threshold
    if above_threshold.any():  # Check if any value exceeds the threshold
        return above_threshold.idxmax(
            skipna=True
        )  # Return the index of the first occurrence
    else:
        return np.nan  # Return np.nan if no value is greater than the threshold


result_indices = lat_window.groupby("unique_res_id").apply(first_occurrence_index)

# Drop rows where result is none
result_indices = result_indices.dropna()

latencies = lat_window.loc[result_indices, :]

# %%
latencies

# %%
histogram = sns.histplot(
    data=latencies, x="aligned_ms", hue="condition", element="poly"
)
histogram.set_xlim(0, 100)
histogram.set(xlabel="time (ms)", title="distribution of latencies")

# %%
bin_width = 3
num_bins = 25

histogram = sns.histplot(
    data=latencies,
    x="aligned_ms",
    hue="condition",
    binwidth=bin_width,
    bins=num_bins,
    stat="density",
    element="poly",
)

histogram.set_xlim(0, 200)
histogram.set(xlabel="latency (ms)", title="distribution of latencies")
plt.savefig("histogram.svg")


#%%

# %%
latencies.to_csv("latencies.csv", index=False)

# %%
slc_df = latencies[(latencies["aligned_ms"] >= 0) & (latencies["aligned_ms"] <= 23)]
llc_df = latencies[(latencies["aligned_ms"] >= 23) & (latencies["aligned_ms"] <= 200)]

slc_df.groupby("condition").size()
llc_df.groupby("condition").size()

# %%
sel_res = grouped.size().index
slc_plot_df = response_og.loc[response_og["unique_res_id"].isin(sel_res)]
sns.lineplot(
    data=slc_plot_df,
    x="aligned_ms",
    y="vtheta_adj",
    units="unique_res_id",
    estimator=None,
    alpha=0.2,
)

plt.savefig("vtheta_adj_slc.svg")


# %%
# stacked bar chart of % SLC and LLC per condition

total_counts = latencies.groupby("condition").size()
slc_counts = slc_df.groupby("condition").size()
llc_counts = llc_df.groupby("condition").size()
percent_slc = (slc_counts / total_counts) * 100
percent_llc = (llc_counts / total_counts) * 100
stackedbar = pd.DataFrame(
    {
        "Condition": percent_slc.index,
        "Percentage of SLC": percent_slc.values,
        "Percentage of LLC": percent_llc.values,
    }
)

sns.barplot(
    x="Condition", y="Percentage of SLC", data=stackedbar, color="skyblue", label="SLC"
)
sns.barplot(
    x="Condition",
    y="Percentage of LLC",
    data=stackedbar,
    color="orange",
    label="LLC",
    bottom=stackedbar["Percentage of SLC"],
)

plt.ylabel("Percentage")
plt.legend()

plt.savefig("stackedbarchart.svg")


# %%
latencies.groupby("condition").size()
# %%
# LATENCY PARAMETER
lat_means = (
    latencies.groupby(["condition", "trial_id"])["aligned_ms"].mean().reset_index()
)
strip_plot = sns.stripplot(x="condition", y="aligned_ms", data=lat_means)
box_plot = sns.boxplot(x="condition", y="aligned_ms", data=lat_means)
plt.savefig("latencyboxstrip.svg")

#%%
#Latency between SLC vs. LLC
slc_conditions = (lat_means["aligned_ms"] >= 0) & (lat_means["aligned_ms"] <= 23)
llc_conditions = (lat_means["aligned_ms"] > 23) & (lat_means["aligned_ms"] <= 200)

# Separate latency means into SLC and LLC groups
lat_means_slc = lat_means[slc_conditions]
lat_means_llc = lat_means[llc_conditions]

#%%
latencies 
# %%
# ANGULAR VELOCITY PARAMETER!

slc_max = responses_max.loc[
    responses_max["unique_res_id"].isin(slc_df["unique_res_id"])
]
llc_max = responses_max.loc[
    responses_max["unique_res_id"].isin(llc_df["unique_res_id"])
]

# %%
slc_trial_means = slc_max.groupby(['condition','trial_id'])['absvtheta'].mean().reset_index()
strip_plot = sns.stripplot(x="condition", y="absvtheta", data=slc_trial_means)
box_plot = sns.boxplot(x="condition", y="absvtheta", data=slc_trial_means)
strip_plot.set_ylim(0,0.5)
box_plot.set_ylim(0,0.5)
plt.savefig('slcmax.svg')


#%%
llc_trial_means = llc_max.groupby(['condition','trial_id'])['absvtheta'].mean().reset_index()
strip_plot = sns.stripplot(x="condition", y="absvtheta", data=llc_trial_means)
box_plot = sns.boxplot(x="condition", y="absvtheta", data=llc_trial_means)
strip_plot.set_ylim(0,0.5)
box_plot.set_ylim(0,0.5)
plt.savefig('llcmax.svg')
#%%
llc_sibs = llc_trial_means[llc_trial_means['condition'] == 'sibs']['absvtheta']
llc_tau = llc_trial_means[llc_trial_means['condition'] == 'tau']['absvtheta']



#%%

trial_means = responses_max.groupby(['condition','trial_id'])['absvtheta'].mean().reset_index()
strip_plot = sns.stripplot(x="condition", y="absvtheta", data=trial_means)
box_plot = sns.boxplot(x="condition", y="absvtheta", data=trial_means)
# %%
sns.lineplot(data=trial_means, x="time_adj", y="vtheta_adj", hue="condition")


# %%
max_vtheta_data = (
    trial_means.groupby(["condition", "trial_id"])["vtheta_adj"].max().reset_index()
)


# %%
strip_plot = sns.stripplot(x="condition", y="vtheta_adj", data=max_vtheta_data)
plt.show()

# %%
pointplot = sns.pointplot(x="condition", y="vtheta_adj", data=max_vtheta_data)
plt.show()

# %%
boxplot = sns.boxplot(x="condition", y="vtheta_adj", data=max_vtheta_data)
plt.show()

# %%
# area under the curve
