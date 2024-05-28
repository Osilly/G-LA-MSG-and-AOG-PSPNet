from unittest import result
from network import *
from dataset import *
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import time
import itertools
import matplotlib
import random
from tool.tool import *

matplotlib.use("Agg")


class Train:
    def __init__(
        self,
        train_path="data/train",
        test_path="data/test",
        result_path="result",
        signal_size=4096,
        num_epochs=5000000,
        batch_size=4,
        z_dim=32,
        num=50,
        input_nc=1,
        output_nc=1,
        ch=64,
        n_blocks=6,
        lr=1e-4,
        weight_decay=1e-4,
        adv_weight=1,
        cycle_weight=10,
        identity_weight=10,
        preserving_identity_weight=10,
        cam_weight=1000,
        decay_flag=True,
        device="cuda:0",
        resume=False,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.result_path = result_path
        self.signal_size = signal_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num = num
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ch = ch
        self.n_blocks = n_blocks
        self.lr = lr
        self.weight_decay = weight_decay
        self.adv_weight = adv_weight
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight
        self.preserving_identity_weight = preserving_identity_weight
        self.cam_weight = cam_weight
        self.decay_flag = decay_flag
        self.device = device
        self.resume = resume

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # A is simulation, B is true
    def dataload(self):
        Traindata = GetData(self.train_path)
        self.trainA, self.trainB = Traindata.get_data()
        self.trainA_loader = DataLoader(
            GetDataset(self.trainA), batch_size=self.batch_size, shuffle=True
        )
        self.trainB_loader = DataLoader(
            GetDataset(self.trainB), batch_size=self.batch_size, shuffle=True
        )

        Testdata = GetData(self.test_path)
        self.testA, self.testB = Testdata.get_data()
        self.testA_loader = DataLoader(
            GetDataset(self.testA), batch_size=self.batch_size, shuffle=False
        )

    def build_model(self):
        self.gen = ResnetGenerator(
            input_nc=self.input_nc,
            output_nc=self.output_nc,
            ngf=self.ch,
            n_blocks=self.n_blocks,
            signal_size=self.signal_size,
            z_dim=self.z_dim,
        ).to(self.device)
        self.disGA = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=7).to(
            self.device
        )
        self.disGB = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=7).to(
            self.device
        )
        self.disLA = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=5).to(
            self.device
        )
        self.disLB = Discriminator(input_nc=self.input_nc, ndf=self.ch, n_layers=5).to(
            self.device
        )

    def define_loss(self):
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

    def define_optim(self):
        self.G_optim = torch.optim.Adam(
            itertools.chain(self.gen.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999),
        )
        self.D_optim = torch.optim.Adam(
            itertools.chain(
                self.disGA.parameters(),
                self.disGB.parameters(),
                self.disLA.parameters(),
                self.disLB.parameters(),
            ),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999),
        )

    def define_rho(self):
        self.Rho_clipper = RhoClipper(0, 1)

    def save_model(self, path, step):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        params = {}
        params["mlp"] = self.gen.mlp.state_dict()
        params["gen"] = self.gen.state_dict()
        params["disGA"] = self.disGA.state_dict()
        params["disGB"] = self.disGB.state_dict()
        params["disLA"] = self.disLA.state_dict()
        params["disLB"] = self.disLB.state_dict()
        params["G_optim"] = self.G_optim.state_dict()
        params["D_optim"] = self.D_optim.state_dict()
        torch.save(params, os.path.join(path, "model_params_%07d.pt" % step))

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, "model_params_%07d.pt" % step))
        self.gen.load_state_dict(params["gen"])
        self.disGA.load_state_dict(params["disGA"])
        self.disGB.load_state_dict(params["disGB"])
        self.disLA.load_state_dict(params["disLA"])
        self.disLB.load_state_dict(params["disLB"])
        self.G_optim.load_state_dict(params["G_optim"])
        self.D_optim.load_state_dict(params["D_optim"])

    def train(self):
        self.gen.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_step = 1
        # 续训
        if self.resume:
            model_list = glob(os.path.join(self.result_path, "model", "*.pt"))
            if len(model_list) != 0:
                model_list.sort()
                start_step = int(model_list[-1].split("_")[-1].split(".")[0])
                self.load_model(os.path.join(self.result_path, "model"), start_step)
                print("load success!")
                # 学习率衰减
                if self.decay_flag and start_step > (self.num_epochs // 2):
                    self.G_optim.param_groups[0]["lr"] -= (
                        self.lr / (self.num_epochs // 2)
                    ) * (start_step - self.num_epochs // 2)
                    self.D_optim.param_groups[0]["lr"] -= (
                        self.lr / (self.num_epochs // 2)
                    ) * (start_step - self.num_epochs // 2)

        print("training start!")
        start_time = time.time()
        # 学习率衰减
        for step in range(start_step, self.num_epochs + 1):
            if self.decay_flag and step > (self.num_epochs // 2):
                self.G_optim.param_groups[0]["lr"] -= self.lr / (self.num_epochs // 2)
                self.D_optim.param_groups[0]["lr"] -= self.lr / (self.num_epochs // 2)

            try:
                real_A = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A = trainA_iter.next()

            try:
                real_B = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # generate z
            z_A2A = torch.randn(self.batch_size, self.z_dim * 2) * 0.2
            z_A2A[:, self.z_dim :] += 1
            z_A2A = z_A2A.to(self.device)

            z_B2B = torch.randn(self.batch_size, self.z_dim * 2) * 0.2
            z_B2B[:, : self.z_dim] += 1
            z_B2B = z_B2B.to(self.device)

            z_A2B = torch.randn(self.batch_size, self.z_dim * 2) * 0.2
            z_A2B[:, : self.z_dim] += 1
            z_A2B = z_A2B.to(self.device)

            z_B2A = torch.randn(self.batch_size, self.z_dim * 2) * 0.2
            z_B2A[:, self.z_dim :] += 1
            z_B2A = z_B2A.to(self.device)

            z_A2B2A = torch.randn(self.batch_size, self.z_dim * 2) * 0.2
            z_A2B2A[:, self.z_dim :] += 1
            z_A2B2A = z_A2B2A.to(self.device)

            z_B2A2B = torch.randn(self.batch_size, self.z_dim * 2) * 0.2
            z_B2A2B[:, : self.z_dim] += 1
            z_B2A2B = z_B2A2B.to(self.device)

            # D_loss
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.gen(real_A, z_A2B)
            fake_B2A, _, _ = self.gen(real_B, z_B2A)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(
                real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)
            ) + self.MSE_loss(
                fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device)
            )
            D_ad_cam_loss_GA = self.MSE_loss(
                real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)
            ) + self.MSE_loss(
                fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device)
            )
            D_ad_loss_LA = self.MSE_loss(
                real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)
            ) + self.MSE_loss(
                fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device)
            )
            D_ad_cam_loss_LA = self.MSE_loss(
                real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)
            ) + self.MSE_loss(
                fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device)
            )
            D_ad_loss_GB = self.MSE_loss(
                real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)
            ) + self.MSE_loss(
                fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device)
            )
            D_ad_cam_loss_GB = self.MSE_loss(
                real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)
            ) + self.MSE_loss(
                fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device)
            )
            D_ad_loss_LB = self.MSE_loss(
                real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)
            ) + self.MSE_loss(
                fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device)
            )
            D_ad_cam_loss_LB = self.MSE_loss(
                real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)
            ) + self.MSE_loss(
                fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device)
            )

            D_loss_A = self.adv_weight * (
                D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA
            )
            D_loss_B = self.adv_weight * (
                D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB
            )
            Discriminator_loss = D_loss_A + D_loss_B

            Discriminator_loss.backward()
            self.D_optim.step()

            # G_loss
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, real_e_A = self.gen(real_A, z_A2B)
            fake_B2A, fake_B2A_cam_logit, real_e_B = self.gen(real_B, z_B2A)

            fake_A2B2A, _, fake_e_A2B = self.gen(fake_A2B, z_A2B2A)
            fake_B2A2B, _, fake_e_B2A = self.gen(fake_B2A, z_B2A2B)

            fake_A2A, fake_A2A_cam_logit, _ = self.gen(real_A, z_A2A)
            fake_B2B, fake_B2B_cam_logit, _ = self.gen(real_B, z_B2B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(
                fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device)
            )
            G_ad_cam_loss_GA = self.MSE_loss(
                fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device)
            )
            G_ad_loss_LA = self.MSE_loss(
                fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device)
            )
            G_ad_cam_loss_LA = self.MSE_loss(
                fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device)
            )
            G_ad_loss_GB = self.MSE_loss(
                fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device)
            )
            G_ad_cam_loss_GB = self.MSE_loss(
                fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device)
            )
            G_ad_loss_LB = self.MSE_loss(
                fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device)
            )
            G_ad_cam_loss_LB = self.MSE_loss(
                fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device)
            )

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            # 让gen的encoder强制学习两个域之间的共通之处
            G_preserving_identity_loss_A = self.L1_loss(fake_e_A2B, real_e_A)
            G_preserving_identity_loss_B = self.L1_loss(fake_e_B2A, real_e_B)

            G_cam_loss_A = self.BCE_loss(
                fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)
            ) + self.BCE_loss(
                fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device)
            )
            G_cam_loss_B = self.BCE_loss(
                fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)
            ) + self.BCE_loss(
                fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device)
            )

            G_loss_A = (
                self.adv_weight
                * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA)
                + self.cycle_weight * G_recon_loss_A
                + self.identity_weight * G_identity_loss_A
                + self.preserving_identity_weight * G_preserving_identity_loss_A
                + self.cam_weight * G_cam_loss_A
            )
            G_loss_B = (
                self.adv_weight
                * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB)
                + self.cycle_weight * G_recon_loss_B
                + self.identity_weight * G_identity_loss_B
                + self.preserving_identity_weight * G_preserving_identity_loss_B
                + self.cam_weight * G_cam_loss_B
            )

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss = Generator_loss * 0.5
            Generator_loss.backward()
            self.G_optim.step()

            # 更新adaILN的rho参数
            self.gen.apply(self.Rho_clipper)
            print(
                "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f"
                % (
                    step,
                    self.num_epochs,
                    time.time() - start_time,
                    Discriminator_loss,
                    Generator_loss,
                )
            )

            # 测试
            if step % 2000 == 0:
                test_sample_num = 5
                self.gen.eval()
                for i in range(test_sample_num):
                    try:
                        real_A = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A = testA_iter.next()
                    real_A = real_A.to(self.device)
                    get_interpolation(
                        real_A,
                        os.path.join(self.result_path, "train", str(step), str(i)),
                        self.num,
                        self.z_dim,
                        self.gen,
                        self.device,
                    )
                self.gen.train()

            if step % 10000 == 0:
                self.save_model(os.path.join(self.result_path, "model"), step)

            if step % 1000 == 0:
                params = {}
                params["mlp"] = self.gen.mlp.state_dict()
                params["gen"] = self.gen.state_dict()
                params["disGA"] = self.disGA.state_dict()
                params["disGB"] = self.disGB.state_dict()
                params["disLA"] = self.disLA.state_dict()
                params["disLB"] = self.disLB.state_dict()
                params["G_optim"] = self.G_optim.state_dict()
                params["D_optim"] = self.D_optim.state_dict()
                torch.save(params, os.path.join("model_params_latest.pt"))


if __name__ == "__main__":
    gan = Train(
        train_path="data/train",
        test_path="data/test",
        result_path="result",
        signal_size=4096,
        num_epochs=1000000,
        batch_size=6,
        z_dim=32,
        num=50,
        input_nc=1,
        output_nc=1,
        ch=64,
        n_blocks=6,
        lr=1e-4,
        weight_decay=1e-4,
        adv_weight=1,
        cycle_weight=10,
        identity_weight=10,
        preserving_identity_weight=10,
        cam_weight=0,
        decay_flag=True,
        device="cuda:2",
        resume=True,
    )
    gan.setup_seed(2022)
    gan.dataload()
    gan.build_model()
    gan.define_loss()
    gan.define_optim()
    gan.define_rho()
    gan.train()
    print("training finished!")
