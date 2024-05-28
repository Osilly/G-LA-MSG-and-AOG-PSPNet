import os
from torch import nn
from torch import optim
from torch._C import device
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from glob import glob
from Network import *
from AogBlock import *
from Dataset import *
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


class Train:
    def __init__(
        self,
        train_path="data/train",
        test_path="data/test",
        result_path="result",
        signal_size=4096,
        num_epochs=200,
        batch_size=50,
        input_nc=1,
        output_nc=1,
        resblock_size=[3, 4, 23, 3],
        Ttype=T_Normal_Block,
        sub_nums=4,
        alpha=1.0,
        milestones=[10, 20, 30],
        lr=1e-4,
        weight_decay=1e-4,
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
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.resblock_size = resblock_size
        self.Ttype = Ttype
        self.sub_nums = sub_nums
        self.alpha = alpha
        self.milestones = milestones
        self.lr = lr
        self.weight_decay = weight_decay
        self.decay_flag = decay_flag
        self.device = device
        self.resume = resume

        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.start_epoch = 1

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def dataload(self):
        getdata = GetData(self.train_path)
        data, label, label_cls = getdata.get_data()

        train_size = int(len(data) * 0.85)
        valid_size = len(data) - train_size
        train_iter, valid_iter = torch.utils.data.random_split(
            GetDataset(data, label, label_cls), [train_size, valid_size]
        )
        self.train_iter = DataLoader(
            train_iter, batch_size=self.batch_size, shuffle=True
        )
        self.valid_iter = DataLoader(
            valid_iter, batch_size=self.batch_size, shuffle=False
        )

        getdata = GetData(self.test_path)
        test_data, test_label, test_label_cls = getdata.get_data()
        self.test_iter = DataLoader(
            GetDataset(test_data, test_label, test_label_cls),
            batch_size=1,
            shuffle=False,
        )

    def build_model(self):
        self.net = PSPNet(
            sizes=(1, 2, 3, 6),
            psp_size=2048,
            deep_features_size=1024,
            resblock_size=self.resblock_size,
        ).to(device=self.device)

    def define_loss(self):

        self.seg_criterion = nn.CrossEntropyLoss().to(device=self.device)
        self.cls_criterion = nn.BCEWithLogitsLoss().to(device=self.device)

    def define_optim(self):
        self.optim = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = MultiStepLR(
            self.optim, milestones=[int(x) for x in self.milestones]
        )

    def save_model(self, path, step):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        params = {}
        params["net"] = self.net.state_dict()
        params["optim"] = self.optim.state_dict()
        params["scheduler"] = self.scheduler.state_dict
        params["train_loss"] = self.train_loss
        params["valid_loss"] = self.valid_loss
        params["train_acc"] = self.train_acc
        params["valid_acc"] = self.valid_acc
        params["start_epoch"] = self.start_epoch
        torch.save(params, os.path.join(path, "model_params_%07d.pt" % step))

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, "model_params_%07d.pt" % step))
        self.net.load_state_dict(params["net"])
        self.optim.load_state_dict(params["optim"])
        # self.scheduler.load_state_dict(params['scheduler'])
        self.train_loss = params["train_loss"]
        self.valid_loss = params["valid_loss"]
        self.train_acc = params["train_acc"]
        self.valid_acc = params["valid_acc"]
        self.start_epoch = params["start_epoch"]

    def plot_result(self, result, step, num):
        path = os.path.join(os.path.join(self.result_path, str(step)), "result")
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        np.savetxt(os.path.join(path, "result{}.txt".format(num)), result)
        plt.plot(result, label="result")
        plt.legend()
        plt.savefig(os.path.join(path, "result{}.png".format(num)), dpi=600)
        plt.cla()

    @staticmethod
    def get_acc(out, label):
        num_correct = 0
        total = out.shape[0]
        _, pred_label = out.max(1)
        for i in range(len(label)):
            if (
                np.abs(
                    pred_label[i].cpu().detach().numpy()
                    - label[i].cpu().detach().numpy()
                )
                <= 6
            ):
                num_correct += 1
        # num_correct = (pred_label == label).sum().item()
        return num_correct / total

    def train(self):
        if self.resume:
            model_list = glob(os.path.join(self.result_path, "model", "*.pt"))
            if len(model_list) != 0:
                model_list.sort()
                start_step = int(model_list[-1].split("_")[-1].split(".")[0])
                self.load_model(os.path.join(self.result_path, "model"), start_step)
                print("load success!")

        for epoch in range(self.start_epoch, 1 + self.num_epochs):
            self.net.train()
            train_loss = 0
            train_acc = 0
            valid_loss = 0
            valid_acc = 0
            # train
            for x, y, y_cls in self.train_iter:
                self.optim.zero_grad()
                x, y, y_cls = (
                    x.to(dtype=torch.float, device=self.device),
                    y.to(dtype=torch.long, device=self.device),
                    y_cls.to(dtype=torch.float, device=self.device),
                )
                out, out_cls = self.net(x)
                out = out.view(out.shape[0], -1)
                out = out.to(dtype=torch.float, device=self.device)
                seg_loss = self.seg_criterion(out, y)
                cls_loss = self.cls_criterion(out_cls, y_cls)
                loss = seg_loss + self.alpha * cls_loss
                loss.backward()
                self.optim.step()
                train_loss += loss.item()
                train_acc += self.get_acc(out, y)
            self.scheduler.step()
            train_loss = train_loss / len(self.train_iter)
            train_acc = train_acc / len(self.train_iter)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            # valid
            self.net.eval()
            for x, y, y_cls in self.valid_iter:
                x, y, y_cls = (
                    x.to(dtype=torch.float, device=self.device),
                    y.to(dtype=torch.long, device=self.device),
                    y_cls.to(dtype=torch.float, device=self.device),
                )
                out, out_cls = self.net(x)
                out = out.view(out.shape[0], -1)
                out = out.to(dtype=torch.float, device=self.device)
                seg_loss = self.seg_criterion(out, y)
                cls_loss = self.cls_criterion(out_cls, y_cls)
                loss = seg_loss + self.alpha * cls_loss
                valid_loss += loss.item()
                valid_acc += self.get_acc(out, y)
            valid_acc = valid_acc / len(self.valid_iter)
            valid_loss = valid_loss / len(self.valid_iter)
            self.valid_loss.append(valid_loss)
            self.valid_acc.append(valid_acc)

            print(
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss, train_acc, valid_loss, valid_acc)
            )

            if epoch % 40 == 0:
                num = 0
                for x, y, y_cls in self.test_iter:
                    x, y, y_cls = (
                        x.to(dtype=torch.float, device=self.device),
                        y.to(dtype=torch.long, device=self.device),
                        y_cls.to(dtype=torch.float, device=self.device),
                    )
                    out, out_cls = self.net(x)
                    out = torch.exp(out)
                    out = out.view(out.shape[0], -1)
                    out = out.to(dtype=torch.float, device="cpu").detach().numpy()
                    for i in range(out.shape[0]):
                        num += 1
                        result = out[i]
                        self.plot_result(result, epoch, num)

            if epoch % 50 == 0:
                self.start_epoch = epoch
                self.save_model(os.path.join(self.result_path, "model"), epoch)


if __name__ == "__main__":
    train = Train(
        train_path="data/train",
        result_path="result",
        signal_size=4096,
        num_epochs=800,
        batch_size=6,
        input_nc=1,
        output_nc=1,
        resblock_size=[3, 4, 23, 3],
        Ttype=T_Normal_Block,
        sub_nums=2,
        alpha=0.5,
        milestones=[10, 20, 30],
        lr=1e-3,
        weight_decay=0,
        decay_flag=True,
        device="cuda:1",
        resume=False,
    )
    train.setup_seed(2021)
    train.dataload()
    train.build_model()
    train.define_loss()
    train.define_optim()
    train.train()
    print("training finished!")
