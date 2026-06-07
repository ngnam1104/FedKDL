import json
import sys
import copy

with open('d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/Kaggle_FedKDL.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the index of the markdown cell for '4.1'
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '### 4.1' in cell['source'][0]:
        insert_idx = i
        break

if insert_idx != -1:
    md_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### 3.5. Tạo Proxy Dataset (15% URPC)\n',
            'Trích xuất 15% dữ liệu URPC để tạo tập proxy dùng cho việc Warmup (trước FL) và Knowledge Distillation (tại Gateway).'
        ]
    }
    
    code_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import os, random, yaml\n',
            'from pathlib import Path\n',
            '\n',
            'orig_yaml = "datasets/URPC2020.yaml"\n',
            'proxy_yaml = "datasets/URPC2020_proxy.yaml"\n',
            '\n',
            'with open(orig_yaml, "r") as f:\n',
            '    config = yaml.safe_load(f)\n',
            '\n',
            'train_dir = Path("datasets/URPC2020/URPC2020/images/train")\n',
            'if train_dir.exists():\n',
            '    all_imgs = list(train_dir.glob("*.jpg"))\n',
            '    random.seed(1104)\n',
            '    proxy_imgs = random.sample(all_imgs, int(len(all_imgs) * 0.15))\n',
            '    \n',
            '    proxy_txt = Path("datasets/URPC2020/proxy_train.txt")\n',
            '    with open(proxy_txt, "w") as f:\n',
            '        for img in proxy_imgs:\n',
            '            f.write(f"{img.absolute()}\\n")\n',
            '            \n',
            '    config["train"] = str(proxy_txt.absolute())\n',
            '    with open(proxy_yaml, "w") as f:\n',
            '        yaml.dump(config, f)\n',
            '    print(f"Đã tạo {proxy_yaml} với {len(proxy_imgs)} ảnh proxy (15%).")\n',
            'else:\n',
            '    print("Chưa tìm thấy thư mục ảnh. Hãy đảm bảo mount dataset đúng cách.")'
        ]
    }
    
    nb['cells'].insert(insert_idx, md_cell)
    nb['cells'].insert(insert_idx + 1, code_cell)
    
    with open('d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/Kaggle_FedKDL.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('Inserted cells successfully.')
else:
    print('Could not find insertion point.')
