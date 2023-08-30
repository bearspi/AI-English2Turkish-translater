import json
import pandas as pd

def formating_data(row):
    en_text = row["translation"]['en']
    tr_text = row["translation"]['tr']
    return f"<SOS>{en_text}<EOS> \n<SOS>{tr_text}<EOS>"

def format_data(name: str) -> None:
# Read a Parquet file
    df = pd.read_parquet(f'{name}.parquet')

    # Apply the formatting function to each row and create a list of formatted strings
    formatted_data = df.apply(format_data, axis=1)

    # Write the formatted data to a text file
    with open(f'{name}.txt', 'w') as f:
        for line in formatted_data:
            f.write(f"{line}\n")
