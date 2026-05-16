import pandas as pd
from config import *

# Script per preparing the data. Given by R.P.
def get_stats(
    name, df, action_col, user_col, item_col, sub_col, time_col, extra_action_cols=None
):
    """Helper to extract dataset stats."""
    if pd.api.types.is_numeric_dtype(df[time_col]):
        ts_series = pd.to_datetime(df[time_col], unit="s")
    else:
        ts_series = pd.to_datetime(df[time_col])

    min_date = ts_series.min().strftime("%b %Y")
    max_date = ts_series.max().strftime("%b %Y")

    def action_counts_for(col):
        series = df[col]
        # If numeric, map to readable labels
        if pd.api.types.is_numeric_dtype(series):
            series = series.map({1: "approve/upvote", -1: "remove/downvote"})
        return series.value_counts().to_dict()

    action_counts = action_counts_for(action_col)
    extra_counts = {}
    if extra_action_cols:
        for col in extra_action_cols:
            extra_counts.update(action_counts_for(col))

    return {
        "Dataset": name,
        "Actions": f"{len(df):,}",
        "Posts": f"{df[item_col].nunique():,}",
        "Users": f"{df[user_col].nunique():,}",
        "Communities": f"{df[sub_col].nunique():,}",
        "Time Period": f"{min_date} -- {max_date}",
        **{
            f"Action: {action}": f"{count:,}" for action, count in action_counts.items()
        },
        **{f"Action: {action}": f"{count:,}" for action, count in extra_counts.items()},
    }


# --- LOAD DATA ---
print("Loading ModLog...")
df_mod = pd.read_csv(
    MODLOG_FILE,
    dtype={"description": "string", "sr_id36": "string", "target_fullname": "string"},
)
# Filter for definitive actions only
df_mod = df_mod[df_mod["general_action"].isin(["approve", "remove"])].copy()

print("Loading Submission Info...")
df_cura = pd.read_csv(CURA_INFO_FILE, sep="\t", dtype={"SUBMISSION_ID": "string"})

print("Loading Votes...")
df_votes = pd.read_csv(
    VOTES_FILE, sep="\t", dtype={"SUBMISSION_ID": "string", "USERNAME": "string"}
)


# --- CALCULATE STATS ---
print("Calculating raw statistics...")
# Stats for ModLog
mod_stats = get_stats(
    "Moderator Log",
    df_mod,
    "general_action",
    "mod",
    "target_fullname",
    "sr_id36",
    "created_utc",
)

# Stats for Vote Log
vote_stats = get_stats(
    "Vote Log",
    df_votes,
    "VOTE",
    "USERNAME",
    "SUBMISSION_ID",
    "SUBREDDIT",
    "CREATED_TIME",
)


# --- INTERSECTION LOGIC ---
print("Computing Intersection...")

# Identify the overlap set: Submissions present in BOTH ModLog and Cura Info
overlap_set = set(df_mod["target_fullname"]).intersection(set(df_cura["SUBMISSION_ID"]))
print(f"Overlap Set Size (Posts): {len(overlap_set):,}")

df_mod_filtered = df_mod[df_mod["target_fullname"].isin(overlap_set)].copy()
df_mod_filtered = (
    df_mod_filtered.sort_values("created_utc").groupby("target_fullname").tail(1)
)
df_votes_filtered = df_votes[df_votes["SUBMISSION_ID"].isin(overlap_set)].copy()


# --- MERGE DATASETS ---
print("Merging Votes and Mod Actions...")
analysis_df = df_votes_filtered.merge(
    df_mod_filtered[["target_fullname", "general_action", "created_utc", "sr_id36"]],
    left_on="SUBMISSION_ID",
    right_on="target_fullname",
    how="inner",
)

# --- CLEANING & MAPPING ---
# Map Labels
# Ground Truth: Approve (+1), Remove (-1)
analysis_df["ground_truth"] = analysis_df["general_action"].map(
    {"approve": 1, "remove": -1}
)
# User Votes: Upvote (+1), Downvote (-1)
analysis_df["user_vote"] = analysis_df["VOTE"].map({"upvote": 1, "downvote": -1})

# Use Vote's CREATED_TIME. If null/nan, fall back to Mod's created_utc
print("Imputing missing timestamps...")
analysis_df["final_time"] = analysis_df.apply(
    lambda row: (
        row["created_utc"] if pd.isnull(row["CREATED_TIME"]) else row["CREATED_TIME"]
    ),
    axis=1,
)

# --- CALCULATE INTERSECTION STATS ---
inter_stats = get_stats(
    "Intersection",
    analysis_df,
    "general_action",  # mod actions: approve / remove
    "USERNAME",
    "SUBMISSION_ID",
    "SUBREDDIT",
    "final_time",
    extra_action_cols=["VOTE"],  # vote actions: upvote / downvote
)

# --- PRINT TABLE 1 ---
print("\n" + "=" * 40)
print("TABLE 1: DATASET STATISTICS")
print("=" * 40)
stats_table = pd.DataFrame([mod_stats, vote_stats, inter_stats])

# Save as Markdown
with open("dataset_stats.md", "w") as f:
    f.write("# Table 1: Dataset Statistics\n\n")
    f.write(stats_table.fillna("—").to_markdown(index=False))
    f.write("\n")

print(stats_table)
print("=" * 40 + "\n")


# --- SAVE FINAL DATASET ---
print(f"Saving final dataset to {OUTPUT_FILE}...")

output_columns = {
    "USERNAME": "username",
    "SUBREDDIT": "community",
    "SUBMISSION_ID": "item_id",
    "final_time": "timestamp",
    "user_vote": "vote",
    "ground_truth": "label",
}

final_df = analysis_df.rename(columns=output_columns)[output_columns.values()]
final_df.to_csv(OUTPUT_FILE, index=False)

print("Processing complete.")
