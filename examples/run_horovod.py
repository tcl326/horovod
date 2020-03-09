import os
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--machines', nargs='+', help='machine ips', required=True)

args = parser.parse_args()

for cnn_model in ['resnet101', 'densenet121', 'inceptionv3', 'vgg16']:
    cmd = f"horovodrun -np {4 * len(args.machines)} -H {','.join(args.machines)} -p 12345 python main_horovod.py --data_dir /data --train_epochs 10 --cnn_model={cnn_model}"
    print(cmd)
    os.system(cmd)