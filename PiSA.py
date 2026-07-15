import numpy as np
import torch
import losses
from model import build_model
from lib.amp import autocast
from metrics import compute_metrics_all
import datasets as datasets
import os
import random


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

'''
top-level wrapper
'''
def remove_img_no_label(dataset):
    has_label_mask = np.sum(dataset.label_matrix_obs, axis=1) > 0
    original_length = len(dataset)
    filtered_length = np.sum(has_label_mask)
    removed_count = original_length - filtered_length
    print(f"init {original_length}, "
            f"now {filtered_length}, "
            f"remove {removed_count} unlabeled images")
    if removed_count > 0:
        dataset.image_ids = dataset.image_ids[has_label_mask]
        dataset.label_matrix = dataset.label_matrix[has_label_mask]
        dataset.label_matrix_obs = dataset.label_matrix_obs[has_label_mask]
    return dataset

def run_PiSA(args):
    from munch import Munch as mch
    args = mch(**vars(args))
    set_random_seed(args.pytorch_seed)
    dataset = datasets.get_data(args)
    for phase in ['train']:
        dataset[phase] = remove_img_no_label(dataset[phase])
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=args.bsize,
            shuffle=phase == 'train',
            sampler=None,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True
        )
    model = build_model(backbone_name=args.backbone, num_classes=args.num_classes, dim_embed=args.dim_embed,pretrained=args.pretrained)
    model.cuda()
    MethodName = args.MethodName
    ignore_num = args.k
    print(f"MethodName:{MethodName}, ignore_num:{ignore_num}")
    optimizer_name = "Adam"
    print(f"Using optimizer: {optimizer_name}")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-5)
    bestmap_val = 0
    gamma = args.gamma
    alpha = args.alpha
    print("gamma:",gamma)
    print("alpha:",alpha)
    Loss_proto = losses.PrototypeLoss(gamma=gamma)
    for epoch in range(1, args.num_epochs + 1):
        print(f'Epoch {epoch}/{args.num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                score1_preds = np.zeros((len(dataset[phase]), args.num_classes))
                y_true = np.zeros((len(dataset[phase]), args.num_classes))
                batch_stack = 0
            with torch.set_grad_enabled(phase == 'train'):
                for batchIndex, batch in enumerate(dataloader[phase]):
                    label_vec_obs = batch['label_vec_obs'].cuda()
                    optimizer.zero_grad()
                    image = batch['image'].cuda()
                    with autocast():
                        cls_logits, score2, x_norm = model(image)
                        score1 = torch.sigmoid(cls_logits)  # [batch_size, num_classes]
                    if score1.dim() == 1:
                        score1 = torch.unsqueeze(score1, 0)
                    if phase == 'train':
                        with autocast():
                            if score2.dim() == 1:
                                score2 = torch.unsqueeze(score2, 0)
                            label_vec_true = batch['label_vec_true'].cuda()
                            if epoch == 1:
                                loss_ = losses.loss_an(score1, label_vec_obs).mean()
                            else:
                                loss_mtx = losses.loss_an(score1, label_vec_obs)
                                loss_mtx, mask_neg = losses.PGRL(score2, label_vec_obs, loss_mtx,ignore_num)
                                loss_ = loss_mtx.mean() + Loss_proto(x_norm, model.class_wise_prototypes, label_vec_obs) *alpha
                        loss_.backward()
                        optimizer.step()
                        model.update_prototypes(x_norm, label_vec_obs)
                    else:
                        label_vec_true = batch['label_vec_obs'].clone().numpy()
                        this_batch_size = label_vec_true.shape[0]
                        score1_preds[batch_stack:batch_stack + this_batch_size] = score1.cpu().numpy()
                        y_true[batch_stack:batch_stack + this_batch_size] = label_vec_true
                        batch_stack += this_batch_size
            if phase == 'val':
                metrics1 = compute_metrics_all(score1_preds, y_true)
                print(f"val_metrics_classify = {metrics1}")
                del score1_preds
                del y_true
                map_val = metrics1['map']
        torch.cuda.empty_cache()
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            print(f'best val mAP {bestmap_val:.3f}')
        if args.seg:
            path = os.path.join(args.save_path, "{}_{}_{}_seg.pt".format(args.dataset, args.mode,epoch))
        else:
            path = os.path.join(args.save_path, "{}_{}_{}.pt".format(args.dataset, args.mode,epoch))
            torch.save((model.state_dict(), args), path)
    if args.seg:
        path = os.path.join(args.save_path, "{}_{}_{}_seg.pt".format(args.dataset, args.mode,bestmap_epoch))
    else:
        path = os.path.join(args.save_path, "{}_{}_{}.pt".format(args.dataset, args.mode,bestmap_epoch))
    model_state, _ = torch.load(path, weights_only=False)
    model.load_state_dict(model_state)
    model = model.cuda()
    phase = 'test'
    model.eval()
    score1_pred = np.zeros((len(dataset[phase]), args.num_classes))
    y_true = np.zeros((len(dataset[phase]), args.num_classes))
    batch_stack = 0
    with torch.set_grad_enabled(phase == 'train'):
        for batch in dataloader[phase]:
            # Move data to GPU
            image = batch['image'].cuda()
            label_vec_obs = batch['label_vec_obs'].cuda()
            label_vec_true = batch['label_vec_true'].clone().numpy()
            # Forward pass
            optimizer.zero_grad()
            cls_logits,score2,_ = model(image)
            score1 = torch.sigmoid(cls_logits)
            if score1.dim() == 1:
                score1 = torch.unsqueeze(score1, 0)
            preds_np = score1.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            score1_pred[batch_stack:batch_stack + this_batch_size] = score1.cpu().numpy()
            y_true[batch_stack:batch_stack + this_batch_size] = label_vec_true
            batch_stack += this_batch_size
    metrics1 = compute_metrics_all(score1_pred, y_true)
    print('Training procedure completed!')
    print(f"test_metrics_classify = {metrics1}")