import argparse
import numpy as np
pp = argparse.ArgumentParser(description='')
pp.add_argument('--dataset', type=str, default='pascal', choices=['pascal',
                                                        'coco', 'nuswide', 'cub'])
pp.add_argument('--left_rate', type=float, default=0.5, help='The rate of positive samples to be selected')
args = pp.parse_args()

label_path = 'format_label/vg'
salient_value = np.load(f"{label_path}/salient_value.npy")
label_vec_true = np.load(f"{label_path}/formatted_train_labels.npy")
salient_preds_masked =  np.where(label_vec_true==1, salient_value, 0)
pos_sum = np.sum(label_vec_true==1)
# pos_sum
left_rate = args.left_rate
save_num = int(pos_sum * left_rate)
salient_preds_masked
k = save_num  # 例如第5大的值
flat = salient_preds_masked.flatten()
if k <= len(flat):
    kth_largest = np.partition(flat, -k)[-k]
    print(f"第{k}大的值为: {kth_largest}")
else:
    print("k超出范围")
label_vec_obs = np.zeros_like(label_vec_true)
for i in range(salient_preds_masked.shape[0]):
    for j in range(salient_preds_masked.shape[1]):
        if salient_preds_masked[i, j] >= kth_largest:
            label_vec_obs[i,j] = 1
np.save(f"{label_path}/salient_{left_rate}.npy", label_vec_obs)