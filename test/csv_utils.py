from pathlib import Path
import pandas
import fnmatch


csv_dir_resolve = Path(__file__).resolve().parent / "csv"

# read the csv files
pf_csv_list = [c for c in Path(csv_dir_resolve / "out").glob("*.csv") if fnmatch.fnmatchcase(c.stem, "p2*profiled*")]
pc_csv_list = [c for c in Path(csv_dir_resolve / "out").glob("*.csv") if fnmatch.fnmatchcase(c.stem, "p2*processed*")]

# turn .csv to dataframe
pf_df_list = [pandas.read_csv(c) for c in pf_csv_list]
pc_df_list = [pandas.read_csv(c) for c in pc_csv_list]

# concatenate dataframe
pf_df_concat = pandas.concat(pf_df_list, ignore_index=True)
pc_df_concat = pandas.concat(pc_df_list, ignore_index=True)

# turn concatenated dataframe to .csv
pf_csv = pf_df_concat.to_csv(csv_dir_resolve / "results" / "profiled_concat.csv", mode="a", index=False)
pc_csv = pc_df_concat.to_csv(csv_dir_resolve / "results" / "processed_concat.csv", mode="a", index=False)
