# RPS_CNN_Lite
## Get started
### Create environment
```shell
git clone https://github.com/grosshill/RPS_CNN_Lite.git
cd RPS_CNN_Lite
conda env create -f environment.yml
conda activate rps
```

## Data structure
Please copy datasets as following structure:
```shell
./root
    - data
    |
    | -rps
    | | -paper
    | | -rock
    | | -scissors
    |
    | -rps-test-set
    | | -paper
    | | -rock
    | | -scissors
```
Do not change the structure or the directory name, or you have to modify code in `dataset.py`.

## Train
```shell
python train.py --device cuda --batch_size 64 --epochs 100 -tb
```
Results can be found in `./output/exp__`

### Check the training process
The training process is recorded by `TensorBoard`:
```shell
tensorboard --logdir ./output/exp__ --reload_interval 5
```
Click the link `http://localhost:6006/` shown in shell to check the graphs.
```shell
TensorBoard 2.12.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

## Test
```shell
python test.py -m ./output/exp__/best.pth
```
### Check the test result
Results can be found in `./test_result/result__`, where `model.pth` is the weight file you used to test, and `test_result.json` records the details of test.