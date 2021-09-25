from pathlib import Path
import numpy as np

save_dir = Path('data/Celeba/Anno')
attr_pred_path = 'saves/Celeba/33-attr.txt'

img_names = []
pred_attrs = []
with open(attr_pred_path) as fin:
    for line in fin:
        splits = line.strip().split(',')
        img_names.append(splits[0])
        pred_attrs.append([float(i) for i in splits[1:]])

pred_attrs = np.array(pred_attrs)
for i in range(pred_attrs.shape[1]):
    pred_attrs[:, i] = (pred_attrs[:, i] - pred_attrs[:, i].min()) / (pred_attrs[:, i].max() - pred_attrs[:, i].min())

output_lines = []
for img_name, normed_attr in zip(img_names, pred_attrs):
    normed_attr = normed_attr.tolist()
    line_list = [img_name]
    for attr in normed_attr:
        line_list.append('{:.4f}'.format(attr))
    output_lines.append(','.join(line_list) + '\n')

with open(save_dir / 'normed-attr.txt', 'w') as fout:
    for l in output_lines:
        fout.write(l)