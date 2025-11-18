**Usage**

**Create a virtual environment.**

python3.10 -m venv path_to_env
source path_to_env/bin/activate
pip install -r requirements.txt

**Run Predictions**

python predict_brain_age.py --input path_to_csv_with_input_scans --output path_to_output_results_csv --preprocessing_dir directory_for_preprocessed_data --batch_size size_of_batch --num_threads num_threads

The preprocessing steps include SynthSR, SynthSeg and registering to MNI 152 template. The csv with the input scans can have just the original scans in a column named 'filepath', and the preprocessing step will run all three steps mentioned prior. Or, if you have already run SynthSR, the csv should have the filepaths in a column named 'synthsr_path', same with SynthSeg in 'synthseg_path'. If all the preprocessing has been completed, please include the paths in the csv under a column named 'preprocessed_path' and add the --skip_preprocessing True flag.

