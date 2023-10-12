import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader ,TensorDataset
from sklearn.preprocessing import StandardScaler

from torch import optim
from models.LSTMAE import LSTMAE
from train_utils import train_model, eval_model
from Unit.utils import get_from_one ,metrics_calculates,anomaly_scoring,evaluate

from early_stopping import EarlyStopping

early_stopping = EarlyStopping('./earlysave')

parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')#0.001
parser.add_argument('--input-size', type=int, default=25, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=5, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=50, help='sequence full size')
parser.add_argument('--datapath',default='./data/PSM',help='datapath')
parser.add_argument('--data',default="PSM",help='data')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args = parser.parse_args(args=[])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# folder settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)


class toy_dataset(torch.utils.data.Dataset):
    def __init__(self, toy_data):
        self.toy_data = toy_data

    def __len__(self):
        return self.toy_data.shape[0]

    def __getitem__(self, index):
        return self.toy_data[index]


def main():

    # Create data loaders
    early = 'lstm'
    states = 0
    train_iter, val_iter = create_dataloaders(args.batch_size)

    # Create model
    model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout, seq_len=args.seq_len)
    model.to(device)

    # Create optimizer & loss functions
    optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.MSELoss(reduction='sum')


    # Grid search run if run-grid-search flag is active
    if args.run_grid_search:
        hyper_params_grid_search(train_iter, val_iter, criterion)
        return

    # Train & Val
    for epoch in range(args.epochs):
        # Train loop
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss, train_acc, train_pred_loss = train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, args.grad_clipping,
                    args.log_interval)

        val_loss, val_acc = eval_model(criterion, model, args.model_type, val_iter)

        early_stopping(val_loss, model,states,early)

        if early_stopping.early_stop :
            print("*******************EMB early stop*********************")
            break
   
    print("END")



def create_toy_data(num_of_sequences=10000, sequence_len=64) -> torch.tensor:
    """
    Generate num_of_sequences random sequences with length of sequence_len each.
    :param num_of_sequences: number of sequences to generate
    :param sequence_len: length of each sequence
    :return: pytorch tensor containing the sequences
    """
    
    path = args.datapath
   
    toy_data =pd.read_csv('./data/PSM/PSM/train.csv')
    
    toy_data = toy_data.values[:, 1:]
    toy_data = np.nan_to_num(toy_data)


    scaler = StandardScaler()

    toy_data = scaler.fit_transform(toy_data)

    toy_data = toy_data.astype(None)

    length = int( len(toy_data))
    toy_data = toy_data[:length]
    print(toy_data.shape)
    toy_data = get_from_one(toy_data,window_size=100,stride=1)
    print(toy_data.shape)
    toy_data = torch.tensor(toy_data).float()


    return toy_data


def create_dataloaders(batch_size, train_ratio=0.75, val_ratio=0.25):
    """
    Build train, validation and tests dataloader using the toy data
    :return: Train, validation and test data loaders
    """
    toy_data = create_toy_data()
    print('************')
    print(toy_data.shape)
    len = toy_data.shape[0]

    train_data = toy_data[:int(len * train_ratio), :]
    val_data = toy_data[int(train_ratio * len):int(len * (train_ratio + val_ratio)), :]
   

    print(f'Datasets shapes: Train={train_data.shape}; Validation={val_data.shape}')
    train_iter = torch.utils.data.DataLoader(toy_dataset(train_data), batch_size=batch_size, drop_last=True,shuffle=True)
    val_iter = torch.utils.data.DataLoader(toy_dataset(val_data), batch_size=batch_size,  drop_last=True,shuffle=True)
  

    return train_iter, val_iter


def plot_toy_data(toy_example, description, color='b'):
    """
    Recieves a toy raw data sequence and plot it
    :param toy_example: toy data example sequence
    :param description: additional description to the plot
    :param color: graph color
    :return:
    """
    time_lst = [t for t in range(toy_example.shape[0])]

    plt.figure()
    plt.plot(time_lst, toy_example.tolist(), color=color)
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    # plt.legend()
    plt.title(f'Single value vs. time for toy example {description}')
    plt.show()


def plot_orig_vs_reconstructed(model, test_iter,modelpath, num_to_plot=2):
    """
    Plot the reconstructed vs. Original MNIST figures
    :param model: model trained to reconstruct MNIST figures
    :param test_iter: test data loader
    :param num_to_plot: number of random plots to present
    :return:
    """



    # Plot original and reconstructed toy data
    plot_test_iter = iter(torch.utils.data.DataLoader(test_iter.dataset, batch_size=1, shuffle=False))

    for i in range(num_to_plot):
        orig = next(plot_test_iter).to(device)
        with torch.no_grad():
            rec = model(orig,'all')
            print(rec.shape)

        time_lst = [t for t in range(orig.shape[1])]

        # Plot original
        plot_toy_data(orig.squeeze(), f'Original sequence #{i + 1}', color='g')

        # Plot reconstruction
        plot_toy_data(rec.squeeze(), f'Reconstructed sequence #{i + 1}', color='r')

        # Plot combined
        plt.figure()
        plt.plot(time_lst, orig.squeeze().tolist(), color='g', label='Original signal')
        plt.plot(time_lst, rec.squeeze().tolist(), color='r', label='Reconstructed signal')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.legend()
        title = f'Original and Reconstruction of Single values vs. time for toy example #{i + 1}'
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.show()


def hyper_params_grid_search(train_iter, val_iter, criterion):
    """
    Function to perform hyper-parameter grid search on a pre-defined range of values.
    :param train_iter: train dataloader
    :param val_iter: validation data loader
    :param criterion: loss criterion to use (MSE for reconstruction)
    :return:
    """
    lr_lst = [1e-2, 1e-3, 1e-4]
    hs_lst = [16, 32, 64, 128, 256]
    clip_lst = [None, 10, 1]

    total_comb = len(lr_lst) * len(hs_lst) * len(clip_lst)
    print(f'Total number of combinations: {total_comb}')

    curr_iter = 1
    best_param = {'lr': None, 'hs': None, 'clip_val': None}
    best_val_loss = np.Inf
    params_loss_dict = {}

    for lr in lr_lst:
        for hs in hs_lst:
            for clip_val in clip_lst:
                print(f'Starting Iteration #{curr_iter}/{total_comb}')
                curr_iter += 1
                model = LSTMAE(input_size=args.input_size, hidden_size=hs, dropout_ratio=args.dropout,
                               seq_len=args.seq_len)
                model = model.to(device)
                optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=lr, weight_decay=args.wd)

                for epoch in range(args.epochs):
                    # Train loop
                    train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, clip_val,
                                args.log_interval)
                avg_val_loss, val_acc = eval_model(criterion, model, args.model_type, val_iter)
                params_loss_dict.update({f'lr={lr}_hs={hs}_clip={clip_val}': avg_val_loss})
                if avg_val_loss < best_val_loss:
                    print(f'Found better validation loss: old={best_val_loss}, new={avg_val_loss}; parameters: lr={lr},hs={hs},clip_val={clip_val}')
                    best_val_loss = avg_val_loss
                    best_param = {'lr': lr, 'hs': hs, 'clip_val': clip_val}

    print(f'Best parameters found: {best_param}')
    print(f'Best Validation Loss: {best_val_loss}')
    print(f'Parameters loss: {params_loss_dict}')


if __name__ == '__main__':
    main()
