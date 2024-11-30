import os
import gzip
import pandas as pd
from carball.analysis.utils.pandas_manager import PandasManager

def load_replay_from_file(filename) -> pd.DataFrame:
    with gzip.open(filename, 'rb') as f:
        dataframe = PandasManager.read_numpy_from_memory(f)
    return dataframe

if __name__ == "__main__":
    in_folder = "ranked-duels-json/"
    out_folder = "processed-dataframes/"
    os.makedirs(out_folder, exist_ok=True)

    files_in_folder = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]
    
    for file in files_in_folder:
        input_path = os.path.join(in_folder, file)
        output_path = os.path.join(out_folder, f"{os.path.splitext(file)[0]}.pkl")
        
        try:
            df = load_replay_from_file(input_path)
            df.to_pickle(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
