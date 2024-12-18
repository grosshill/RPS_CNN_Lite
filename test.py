import json
import shutil
import os
import torch
import argparse
from tqdm import tqdm
from dataset import rps_dataset
from torchvision import transforms
from RPS_CNN_Lite import RPS_CNN_Lite, func_timer


class test_result:
    def __init__(self, model_path: str, data_dir: str = './data', device: str = 'cuda'):
        os.makedirs('test_result', exist_ok=True)
        log_len = len(os.listdir('test_result'))
        self.log_dir = os.path.join('test_result', f'result_{str(log_len).zfill(3)}')
        os.makedirs(self.log_dir, exist_ok=True)
        shutil.copyfile(model_path, os.path.join(self.log_dir, 'model.pth'))
        self.log_dict = dict()
        self.model_path = model_path
        self.data_dir = data_dir

        self.device = device
        if self.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = RPS_CNN_Lite()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        transform = transforms.Compose([
            transforms.ToTensor(),  # to tensor
            transforms.Resize((128, 128)),  # downsample and resize the image
        ])

        self.dataset = rps_dataset(self.data_dir, 'test', transform=transform)

    @func_timer
    def test(self):
        self.model.eval()
        total_correct = 0
        total_length = len(self.dataset)
        classes = ['rock', 'paper', 'scissors']
        for img, label, name in tqdm(self.dataset):
            img = img.to(self.device).unsqueeze(0)
            output = self.model(img)
            pred = output.argmax(dim=1, keepdim=True)

            if pred == label:
                total_correct += 1

            tep_dict = {
                'actual': classes[label],
                'predicted': classes[pred],
                'state': bool(pred == label)
            }
            self.log_dict[name.split('\\')[-1].strip()] = tep_dict

        acc = total_correct / total_length
        print('Accuracy of the network on test images: {:.2f}%'.format(acc * 100))
        with open(os.path.join(self.log_dir, 'test_result.json'), 'w') as f:
            json.dump(
                {
                    'Accuracy of the network on test images': '{:.2f}%'.format(acc * 100),
                    'Details': self.log_dict
                }, f, indent=4
            )

        print(f'Check test result: {self.log_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, help='path to trained model')
    parser.add_argument('--data_dir', '-p', default='./data', type=str, help='path to dataset')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='device')
    args = parser.parse_args()

    result = test_result(args.model_path, args.data_dir, args.device)
    result.test()
