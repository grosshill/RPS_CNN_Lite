import os
import torch
import argparse
from torch import nn
from tqdm import tqdm
from dataset import rps_dataset
from torchvision import transforms
from RPS_CNN_Lite import RPS_CNN_Lite, func_timer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RPS_train:
    def __init__(self, args: argparse.Namespace):
        self.data_dir = args.data_dir
        self.device = args.device
        if self.device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.epochs = int(args.epochs)
        self.batch_size = int(args.batch_size)
        self.ds_rate = float(args.downsample_rate)
        self.learning_rate = float(args.learning_rate)
        self.tb = args.tensorboard
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        l_len = len(os.listdir(self.output_dir))
        self.exp_dir = os.path.join(self.output_dir, f'exp_{str(l_len + 1).zfill(3)}')
        os.makedirs(self.exp_dir, exist_ok=True)
        self.best_pth = None

        if self.tb:
            self.writer = SummaryWriter(self.exp_dir)

        transform = transforms.Compose([
            transforms.ToTensor(),  # to tensor
            transforms.Resize((128, 128)),                                      # downsample and resize the image
            transforms.RandomAffine(degrees=3, translate=(0.2, 0.2)),
        ])

        self.dataset = rps_dataset(self.data_dir, 'train', transform)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = RPS_CNN_Lite()
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    @func_timer
    def train(self):
        self.model.train()
        best_precision = -1
        for epoch in range(self.epochs):
            log_loss = 0
            epoch_precision = 0
            total_samples = 0

            print(f'{epoch + 1} / {self.epochs}: ')
            for images, labels, _ in tqdm(self.dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                loss.backward()
                self.optimizer.step()

                log_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                precision = torch.argmax(outputs, dim=1)
                epoch_precision += (precision == labels).sum().item()

            avg_loss = log_loss / total_samples
            accuracy = epoch_precision / total_samples * 100

            if self.tb:
                self.writer.add_scalar("training loss", avg_loss, epoch)
                self.writer.add_scalar("training accuracy", accuracy, epoch)

            print(f'Loss: {avg_loss:.4f}')
            print(f'Precision: {accuracy:.2f}%\n')

            if best_precision < accuracy:
                best_precision = accuracy
                self.best_pth = self.model.state_dict()

        if self.tb:
            images, _, _ = next(iter(self.dataloader))
            self.writer.add_graph(self.model, images.to(self.device))
            self.writer.close()

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'final.pth'))
        torch.save(self.best_pth, os.path.join(self.exp_dir, 'best.pth'))
        print(f'Model saved in {self.exp_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=40)
    parser.add_argument('--downsample_rate', '-s', type=int, default=0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=8e-5)
    parser.add_argument('--tensorboard', '-tb', action='store_true', default=False)
    parser.add_argument('--output_dir', '-o', type=str, default='./output')
    cfg = parser.parse_args()

    RPS = RPS_train(cfg)
    RPS.train()
    print('Finished Training')
    RPS.save_model()
