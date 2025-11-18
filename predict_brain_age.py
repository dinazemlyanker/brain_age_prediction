from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn import Dropout, Conv3d, BatchNorm3d, MaxPool3d, AdaptiveAvgPool3d
from typing import Callable, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import nibabel as nib
import argparse



from brain_age_prediction import RegressionSFCN, UncertaintyWrapper, preprocess_data, BaInferenceDataset

def predict_brain_age(input_file, output_file, include_certainty, batch_size=6, num_threads=4):
    # Read CSV file with pandas
    test_data = pd.read_csv(input_file)
    test_dataset = BaInferenceDataset(test_data, degrade=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)
    trainer = Trainer(accelerator='cpu', devices=1, logger=False)
    model = RegressionSFCN.load_from_checkpoint('brain_age_prediction/model_checkpoint/best_checkpoint_deg.ckpt', input_shape=(1, 160, 160, 192), include_top=True, prediction_range=(18, 90), restrict_strat='sigmoid', dropout=0)
    if include_certainty:
        model = UncertaintyWrapper.load_from_checkpoint('brain_age_prediction/model_checkpoint/best_checkpoint_uncert.ckpt', mean_model=model, feature_shape=64)
        # model = model.to('cuda')
        predictions_list = []
        uncert_list = []
        predictions = trainer.predict(model, dataloaders=test_loader)
        pred_df = pd.DataFrame(predictions.numpy())
        for pred, uncert in predictions:
            for prediction in pred.numpy():
                predictions_list.append(prediction[0])
            for uncertainty in uncert.numpy():
                uncert_list.append(uncertainty[0])
        test_data['brain_age_pred'] = predictions_list
        test_data['uncertainty'] = uncert_list
    else:
        # model = model.to('cuda')
        predictions = trainer.predict(model, dataloaders=test_loader)
        print(predictions)
        values = [t.item() for t in predictions[0]]
        print(values)
        test_data['brain_age_pred'] = values

    test_data.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help='Path to input csv with column named (filepath) or (preprocessed_path) if already preprocessed.')
    parser.add_argument("--output", required=True, help='Path to output csv for brain age predictions.')
    # add what filepath it assumes
    parser.add_argument("--preprocessed_dir", help='Directory for the preprocessed files.')
    parser.add_argument("--skip_preprocessing", default=False, help='Skip preprocessing steps')
    parser.add_argument("--include_uncertainty", default=False, help='Include uncertainty estimates in output.')
    parser.add_argument("--batch_size", required=True, help='Batch size for inference.')
    parser.add_argument("--num_threads", required=True, help='Number of threads for processing at inference.')
    args = parser.parse_args()

    if not args.skip_preprocessing and not args.preprocessed_dir:
        parser.error('--preprocessed_dir is required unless --skip_preprocessing is set')
    
    if not args.skip_preprocessing:
        print('Running preprocessing.....................................')
        preprocess_data(args.input, args.preprocessed_dir)

    print('Running inference.................................')
    predict_brain_age(args.input, args.output, args.include_uncertainty, int(args.batch_size), int(args.num_threads))
