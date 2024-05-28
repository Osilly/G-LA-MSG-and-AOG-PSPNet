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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Test:
    def __init__(
        self,
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
        model_step=0,
    ):
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
        self.model_step = model_step

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
        getdata = GetData(self.test_path)
        data, label, label_cls = getdata.get_data()

        test_iter = GetDataset(data, label, label_cls)
        self.test_iter = DataLoader(
            test_iter, batch_size=self.batch_size, shuffle=False
        )

    def build_model(self):
        self.net = PSPNet(
            sizes=(1, 2, 3, 6),
            psp_size=2048,
            deep_features_size=1024,
            resblock_size=self.resblock_size,
        ).to(device=self.device)

    def load_model(self, path, step):
        params = torch.load(
            os.path.join(path, "model_params_%07d.pt" % step),
            map_location=torch.device(self.device),
        )
        self.net.load_state_dict(params["net"])
        self.train_loss = params["train_loss"]
        self.valid_loss = params["valid_loss"]
        self.train_acc = params["train_acc"]
        self.valid_acc = params["valid_acc"]
        self.start_epoch = params["start_epoch"]

    # paint loss and acc
    def plot_loss_and_acc(self, step):
        path = os.path.join(os.path.join(self.result_path, str(step)), "loss&acc")
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        np.savetxt(os.path.join(path, "train_loss.txt"), self.train_loss)
        np.savetxt(os.path.join(path, "valid_loss.txt"), self.valid_loss)
        plt.plot(self.train_loss, label="train_loss")
        plt.plot(self.valid_loss, label="valid_loss")
        plt.legend()
        plt.savefig(os.path.join(path, "loss.png"), dpi=600)
        plt.cla()

        np.savetxt(os.path.join(path, "train_acc.txt"), self.train_acc)
        np.savetxt(os.path.join(path, "valid_acc.txt"), self.valid_acc)
        plt.plot(self.train_acc, label="train_acc")
        plt.plot(self.valid_acc, label="train_acc")
        plt.legend()
        plt.savefig(os.path.join(path, "acc.png"), dpi=600)
        plt.cla()

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

    def test(self):
        model_list = glob(os.path.join(self.result_path, "model", "*.pt"))
        model_step = 0
        if not len(model_list) == 0:
            if self.model_step and os.path.exists(
                os.path.join(
                    os.path.join(self.result_path, "model"),
                    "model_params_%07d.pt" % self.model_step,
                )
            ):
                model_step = self.model_step
            else:
                model_list.sort()
                model_step = int(model_list[-1].split("_")[-1].split(".")[0])
            self.load_model(os.path.join(self.result_path, "model"), model_step)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.net.eval()
        self.plot_loss_and_acc(model_step)
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
                self.plot_result(result, model_step, num)


if __name__ == "__main__":
    test = Test(
        test_path="data/test",
        result_path="result",
        signal_size=4096,
        num_epochs=200,
        batch_size=3,
        input_nc=1,
        output_nc=1,
        resblock_size=[3, 4, 23, 3],
        Ttype=T_Normal_Block,
        sub_nums=4,
        alpha=0.1,
        milestones=[10, 20, 30],
        lr=1e-3,
        weight_decay=0,
        decay_flag=True,
        device="cuda:1",
        model_step=0,
    )
    test.setup_seed(2021)
    test.dataload()
    test.build_model()
    test.test()
