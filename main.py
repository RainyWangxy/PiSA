import os
import traceback
from config import get_configs
from PiSA import run_PiSA

# from method.train_resnet import run_train_resnet
def main():
    args = get_configs()
    print(args, '\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == 'train_resnet':
        print('## Train ResNet Start ##')
        run_PiSA(args)
if __name__ == "__main__":
    print("## Main Start ##")
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())