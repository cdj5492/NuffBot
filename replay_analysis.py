import carball
import gzip
from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import os

def convert_n_files(in_folder, out_folder, n):
    """
    converts n replay files in the in_folder to the out_folder
    """

    files_in_folder = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]

    for i in range(n):
        print("Starting to process file " + files_in_folder[i])
        in_filename = in_folder + '/' + files_in_folder[i]
        out_filename = out_folder + '/' + files_in_folder[i]
        
        try:
            _json = carball.decompile_replay(in_filename)
        except Exception as e:
            print("Error processing file " + in_filename + ". Skipping...")
            continue

        # _json is a JSON game object (from decompile_replay)
        game = Game()
        game.initialize(loaded_json=_json)

        analysis_manager = AnalysisManager(game)
        analysis_manager.create_analysis()
            
        # return the proto object in python
        proto_object = analysis_manager.get_protobuf_data()

        # return the proto object as a json object
        json_oject = analysis_manager.get_json_data()

        # return the pandas data frame in python
        dataframe = analysis_manager.get_data_frame()

        # write pandas dataframe out as a gzipped numpy array
        with gzip.open(out_filename, 'wb') as fo:
            analysis_manager.write_pandas_out_to_file(fo)

if __name__ == "__main__":
    convert_n_files("ranked-duels-raw", "ranked-duels-json", 50)