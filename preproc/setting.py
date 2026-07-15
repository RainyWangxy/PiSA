import os
import traceback
from config import get_configs
import datasets as datasets
import numpy as np
import torch
from lib.amp import autocast
from PiSA import set_random_seed
from metrics import compute_metrics_all
from model import ImageClassifier
from tqdm import tqdm
import losses
import argparse
def get_salient_value(args,dataloader):
    P = args
    format_path = f"format_label"
    train_loader = dataloader['train']
    train_loader.shuffle = False
    set_random_seed(0)
    select_model = ImageClassifier(args)
    select_model.cuda()
    epoch = 10
    model_state, _ = torch.load(os.path.join("../results/setting",args.dataset,f"{epoch}.pt"))
    select_model.load_state_dict(model_state)
    select_model.eval()
    salient_value = np.zeros((len(train_loader.dataset), args.num_classes))
    batch_stack = 0
    with torch.no_grad():
        for batch in tqdm(train_loader):
            image = batch['image'].cuda()
            logits = select_model(image)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            salient_value[batch_stack: batch_stack + this_batch_size] = preds_np
            batch_stack += this_batch_size
    save_path = f"{format_path}/{args.dataset}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(f"{save_path}/salient_value.npy", salient_value)
    return


def select_model(args,dataloader):
    save_path = os.path.join("../results/setting",args.dataset)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    model = ImageClassifier(args)
    model.cuda()
    bestmap_val = 0
    best_epoch = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-5)
    for epoch in range(1, args.num_epochs + 1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                y_pred = np.zeros((len(dataloader[phase].dataset), args.num_classes))
                y_true = np.zeros((len(dataloader[phase].dataset), args.num_classes))
                batch_stack = 0
            with torch.set_grad_enabled(phase == 'train'):
                for batchIndex, batch in enumerate(dataloader[phase]):
                    label_vec_obs = batch['label_vec'].cuda()
                    # Forward pass
                    optimizer.zero_grad()
                    image = batch['image'].cuda()
                    with autocast():
                        logits = model(image)
                        if logits.dim() == 1:
                            logits = torch.unsqueeze(logits, 0)
                        preds = torch.sigmoid(logits)
                    if phase == 'train':
                        with autocast():
                            loss_ = losses.loss_an(preds,label_vec_obs).mean()
                        loss_.backward()
                        optimizer.step()
                    else:
                        label_vec_true = batch['label_vec'].clone().numpy()
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack:batch_stack + this_batch_size] = preds_np
                        y_true[batch_stack:batch_stack + this_batch_size] = label_vec_true
                        batch_stack += this_batch_size                        
            if phase == 'val':
                metrics = compute_metrics_all(y_pred, y_true)
                print(f"epoch = {epoch}, val_metrics = {metrics}")
                del y_pred
                del y_true
                map_val = metrics['map']                     
        print(" Epoch {} : val mAP {:.3f} \n".format(epoch, map_val))
        torch.cuda.empty_cache()
        if bestmap_val < map_val:
            bestmap_val = map_val
            best_epoch = epoch
        path = os.path.join(save_path, "{}.pt".format(epoch))
        torch.save((model.state_dict(), args), path)
    print("Best epoch: {}, Best mAP: {:.3f}".format(best_epoch, bestmap_val))  
def generate_observe(args):
    def get_single_salient(salient_val_path, label_true_path):
        with torch.no_grad():
            salient_val_np = np.load(salient_val_path)
            label_true_np = np.load(label_true_path)
            observe_label = np.zeros_like(label_true_np)
            for i in range(label_true_np.shape[0]):
                salient_label = -1
                salient_val = 0
                for j in range(label_true_np.shape[1]):
                    if label_true_np[i,j] == 1:
                        if salient_val_np[i,j] > salient_val:
                            salient_val = salient_val_np[i,j]
                            salient_label = j
                if salient_label != -1:
                    observe_label[i,salient_label] = 1
        return observe_label   
    format_path = f"format_label/{args.dataset}"
    if args.setting == "single":
        salient_val_path = f"{format_path}/salient_value.npy"
        observe_label = get_single_salient(salient_val_path,f"{format_path}/formatted_train_labels.npy")
        np.save(os.path.join(format_path,"single_salient.npy"), observe_label)  
    return
      
def main():
    from munch import Munch as mch
    # args.seg = False
    args = mch(**vars(args))
    set_random_seed(args.pytorch_seed)
    dataset_name = args.dataset
    tx = datasets.get_transforms()
    dataset = {}
    for phase in ['train', 'val']:
        meta = datasets.get_metadata(dataset_name)
        image_ids = np.load(os.path.join(meta['path_to_dataset'],f"formatted_{phase}_images.npy"))
        label_matrix = np.load(os.path.join(meta['path_to_dataset'],f"formatted_{phase}_labels.npy"))
        dataset[phase] = datasets.SimpleSet(dataset_name, image_ids, label_matrix,tx[phase])
    dataloader = {}
    for phase in ['train', 'val']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=args.bsize,
            # shuffle=phase == 'train',
            shuffle=False,
            sampler=None,
            num_workers=args.num_workers,
        )
    print(args, '\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    select_model(args, dataloader)
    get_salient_value(args,dataloader) # select model epoch 10
    generate_observe(args)
if __name__ == "__main__":
    print("## Main Start ##")
    try:
        args = get_configs()
        from munch import Munch as mch
        args.seg = False
        args = mch(**vars(args))
        set_random_seed(args.pytorch_seed)
        
    except Exception as e:
        print(traceback.format_exc())


if __name__ == "__main__":
    print("## Main End ##")
    main()