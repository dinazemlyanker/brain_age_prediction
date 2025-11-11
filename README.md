**Usage**

python predict_brain_age.py --input path_to_csv_with_input_scans --output path_to_output_results_csv --preprocessing_dir directory_for_preprocessed_data --batch_size size_of_batch --num_threads num_threads

The csv with the input scans should either have a column named 'filepath' with paths to the scans, or 'preprocessed_path' if the scans have already been preprocessed, in which case you should also pass in --skip_preprocessing True as an argument.
