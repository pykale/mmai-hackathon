# gpt generated
def combine_dataframes(dfs, tags, on="subject_id"):
    """
    Combine multiple dataframes on a common column, prefixing each dataframe's
    columns (except the merge column) with a given tag.
    Keeps only matched IDs across all dataframes (inner join).
    """
    assert len(dfs) == len(tags), "Number of dataframes must match number of tags"

    # Rename columns in each dataframe (except merge key)
    renamed_dfs = []
    for df, tag in zip(dfs, tags):
        df_renamed = df.rename(columns={col: f"{tag}_{col}" for col in df.columns if col != on})
        renamed_dfs.append(df_renamed)

    # Iteratively merge with inner join
    combined = renamed_dfs[0]
    for df in renamed_dfs[1:]:
        combined = combined.merge(df, on=on, how="inner")

    return combined
