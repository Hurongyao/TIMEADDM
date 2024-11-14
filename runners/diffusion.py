import os
import logging
import time
import glob
from Unit.utils import get_from_one,metrics_calculate
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from models.LSTMAE import LSTMAE
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry,noise_estimation_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import pandas as pd
from sklearn.preprocessing import StandardScaler


import torchvision.utils as tvu
import argparse


from early_stopping import EarlyStopping

early_stopping = EarlyStopping('./earlysave')



parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')#0.001
parser.add_argument('--input-size', type=int, default=25, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=5, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=50, help='sequence full size')
parser.add_argument('--datapath',default='./data/PSM/PSM/train.npy',help='datapath')
parser.add_argument('--data',default="PSM",help='data')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args2 = parser.parse_args(args=[])

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
           
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()


    def lstmz(self):
        args, config = self.args, self.config

        #Load data
        if (args.dataset == 'SMAP'):
            print('Load SMAP')
            dataset = np.load('./data/SMAP/SMAP/SMAP_train.npy')

            ckpt1 = torch.load('',
                               map_location=self.device)
            length = int(dataset.shape[0]*0.9)
            traindata = dataset[:length]
            testdata = dataset[length:]


        elif (args.dataset == 'WADI'):
            print('Load WADI')
            dataset = np.load('./data/WADI/wadi_train.npy')
            scaler = StandardScaler()

            dataset = scaler.fit_transform(dataset)

            ckpt1 = torch.load(
                '',
                map_location=self.device)

            length= int(dataset.shape[0]*0.8)
            testdata = dataset[length:]
            traindata = dataset[:length]

        elif (args.dataset == 'SWAT'):
            print('Load SWAT')
            dataset = np.load('./data/SWAT/SWaT_train.npy')
            scaler = StandardScaler()

            dataset = scaler.fit_transform(dataset)

            ckpt1 = torch.load(
                '',
                map_location=self.device)
            length = int(dataset.shape[0]* 0.95)
            testdata = dataset[length:]
            traindata = dataset[:length]


            #
        elif (args.dataset == 'PSM'):
            print('Load PSM')
            dataset = pd.read_csv('./data/PSM/PSM/train.csv')
            dataset = dataset.values[:, 1:]
            dataset = np.nan_to_num(dataset)

            length = int(dataset.shape[0]*0.8)
            traindata = dataset[:length]

            ckpt1 = torch.load(
                './earlysave/best_newPSM_network.pth',
                map_location=self.device)

            label = pd.read_csv('./data/PSM/PSM/test_label.csv')
            label = label.values[:, 1:]
            label = label.astype(None)
            testdata = dataset[length:]

        #Load lstm 
        lstmz = LSTMAE(input_size=args2.input_size, hidden_size=args2.hidden_size, dropout_ratio=args2.dropout,
                       seq_len=args2.seq_len)
        lstmz.to(self.device)
        lstmz.load_state_dict(ckpt1)



        tb_logger = self.config.tb_logger

        #window data
        windowsize = 64
        stride = 1

        traindata = get_from_one(traindata, window_size=windowsize, stride=stride)


        train_loader = data.DataLoader(
            traindata,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        test_loader = data.DataLoader(
            testdata,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        model = Model(config)

        model = model.to(self.device)


        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        # whether to load pre-trained model
        if self.args.resume_training:

            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        #time
        datafirst = time.time()
        real_data = torch.Tensor(testdata)


        for epoch in range(start_epoch, self.config.training.n_epochs):
            earlyloss = 0
            print("This is {} epoch\n".format(epoch))

            data_start = time.time()
            data_time = 0
            for i, x in enumerate(train_loader):

                x =  x.to(self.device)

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                x = x.type(torch.FloatTensor)
                x = x.to(self.device)
                x = lstmz(x,'en')


                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                x, loss = noise_estimation_loss(model, x, t, e, b)
                loss = loss / 1000

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]

                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

            print("Now,eval!")
            re_datas = []
            count = 0
            all_loss = 0
            for tdata in test_loader:
                count += 1
                print("The data is creating:{}".format(count))
                model.eval()
                with torch.no_grad():

                    tdata = torch.reshape(tdata, (2, 64, args2.input_size))
                    tdata = tdata.type(torch.FloatTensor)
                    tdata = tdata.to(self.device)

                    z = lstmz(tdata, 'en')

                    n = z.size(0)

                    e = torch.randn_like(z)
                    b = self.betas
                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)

                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    z_t, loss = noise_estimation_loss(model, z, t, e, b)

                    re_z = self.sample_image(z_t, 1000,model, last=True)

                    re_z = torch.tensor([item.cpu().detach().numpy() for item in re_z])

                    re_z = re_z.to(self.device)
                    tdata = lstmz(re_z, 'de')


                    tdata = torch.reshape(tdata, (128, args2.input_size))
                    re_datas.extend(tdata)

            re_datas = torch.tensor([item.cpu().detach().numpy() for item in re_datas])


            real_data = real_data[:int(len(re_datas))]

            f1=torch.nn.functional.mse_loss(real_data, re_datas)



            earlyloss = f1
            print('earlyloss={}'.format(earlyloss))

            early_stopping(earlyloss, model,states,'ddim')

            if early_stopping.early_stop:
                print("*******************early stop*********************")
                break
        datalast = time.time()
        print((datalast -datafirst )/60)

        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
               #load model
                states = torch.load(
                    './earlysave/best_newtestWADI_DMnetwork.pth',
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )

            model = model.to(self.device)


            model.load_state_dict(states[0])

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            pass
        elif self.args.interpolation:
            pass
        elif self.args.sequence:
            print("sample")
            self.sample_sequence(model,lstmz)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_sequence(self, model,lstmz):

        args, config = self.args, self.config
        with torch.no_grad():
            print("Start smaple!")
           
            print('sequence')
            if (args.dataset == 'SMAP'):
                testdata = np.load('./data/SMAP/SMAP/SMAP_test.npy')

                label = np.load('./data/SMAP/SMAP/SMAP_test_label.npy')


            elif (args.dataset == 'WADI'):
                testdata = np.load('./data/WADI/wadi_test.npy')
                scaler = StandardScaler()

                testdata = scaler.fit_transform(testdata)
                label = np.load('./data/WADI/wadi_labels.npy')


            elif (args.dataset == 'SWAT'):
                testdata = np.load('./data/SWAT/SWaT_test.npy')
                scaler = StandardScaler()
                testdata = scaler.fit_transform(testdata)

                label = np.load('./data/SWAT/SWaT_labels.npy').astype(float)
            elif (args.dataset == 'PSM'):
                testdata = pd.read_csv('./data/PSM/PSM/test.csv')
                testdata = testdata.values[:, 1:]
                testdata = np.nan_to_num(testdata)

                label = pd.read_csv('./data/PSM/PSM/test_label.csv')
                label = label.values[:, 1:]


            label = label.astype(None)
            label = torch.Tensor(label)

            #testdata = testdata[:1280]
            #label = label[:1280]

            dataloader = DataLoader(
                testdata, batch_size=128, shuffle=True, num_workers=0, drop_last=True,
                pin_memory=True)



            real_data = torch.Tensor(testdata)

            re_datas = []
            i = 0
            #Different step
            ts = [50,100,500,1000]
            for tt in range(4):
                for data in dataloader:
                    print("Now step = {},The data is creating:{}".format(ts[tt],i))
                    i += 1
                    data = torch.reshape(data, (2, 64, args2.input_size))
                    data = data.type(torch.FloatTensor)
                    data = data.to(self.device)
                    z = lstmz(data, 'en')

                    n = z.size(0)


                    e = torch.randn_like(z)
                    b = self.betas
                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=ts[tt], size=(n // 2 + 1,)
                    ).to(self.device)


                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]


                    z_t, loss = noise_estimation_loss(model, z, t, e, b)

                    re_z = self.sample_image(z_t,ts[tt],model, last=True)

                    re_z = torch.tensor([item.cpu().detach().numpy() for item in re_z])

                    re_z = re_z.to(self.device)
                    data = lstmz(re_z, 'de')



                    data = torch.reshape(data, (128, args2.input_size))
                    re_datas.extend(data)

            re_datas = torch.tensor([item.cpu().detach().numpy() for item in re_datas])
            print("The length of the data is {}".format(len(re_datas)))
            label = label[:int(len(re_datas)/4)]
            real_data = real_data[:int(len(re_datas)/4)]


            metrics_calculate(real_data, re_datas, label)




    def sample_image(self, x,t_1, model, last=True):

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":

                skip = self.num_timesteps // self.args.timesteps


                seq = range(0, t_1, skip)
                #seq = range(0, 100, skip)


            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs= generalized_steps(x, seq, model, self.betas, eta=self.args.eta)

            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:

            x = x[0][-1]

        return x

    def test(self):
        pass
