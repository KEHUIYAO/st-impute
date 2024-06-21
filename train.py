import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split
from data import GP
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from model import MLP
from utils import STDataset, STDataModule, interpolate_missing_values, convert_st_data_for_dnn_training




   

def main(args):
    if args.dataset == 'GP':
        st_obj_generator = GP

    st_obj = st_obj_generator(args.num_nodes, args.seq_len)
    y_st, y_st_missing, mask_st, space_coords, time_coords = st_obj.generate_st_data_with_missing_values(missing_rate=args.missing_rate, t_l=1, s_l=0.2)
    X_train, y_train, X_test, y_test = convert_st_data_for_dnn_training(y_st, mask_st, space_coords, time_coords, covariate=args.covariate)
    dataset_train_and_val = STDataset(X_train, y_train)
    train_size = int(0.8 * len(dataset_train_and_val))
    val_size = len(dataset_train_and_val) - train_size
    train_dataset, val_dataset = random_split(dataset_train_and_val, [train_size, val_size])
    test_dataset = STDataset(X_test, y_test)
    dm = STDataModule(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)
    
    if args.model == 'MLP':
        model = MLP(input_dim=X_train.shape[1], hidden_dims=args.hidden_dims, output_dim=1)
    
    # Set up the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger)
    trainer.fit(model, dm)

    # test the model
    trainer.test(model, dm)

    # compare with linear interpolation
    y_st_interpolated = interpolate_missing_values(y_st, y_st_missing, mask_st)
    resid = (y_st_interpolated - y_st)[mask_st == 0]
    print("MSE of linear interpolation: ", np.mean(resid ** 2))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='GP')
    parser.add_argument("--model", type=str, default='MLP')
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--covariate", type=str, default='coord')
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dims", type=list, default=[64, 64, 64, 64, 64, 64])
    parser.add_argument("--max_epochs", type=int, default=100)

    args = parser.parse_args()
    main(args)