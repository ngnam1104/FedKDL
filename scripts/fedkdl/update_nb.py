import json
import os

path = r'Kaggle_FedKDL.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "import os, random, yaml\n",
    "from pathlib import Path\n",
    "\n",
    "orig_yaml = \"/kaggle/input/datasets/lywang777/urpc2020/URPC2020/data.yaml\"\n",
    "proxy_yaml = \"datasets/URPC2020_proxy.yaml\"\n",
    "\n",
    "if not os.path.exists(orig_yaml):\n",
    "    print(f\"Không tìm thấy {orig_yaml}\")\n",
    "else:\n",
    "    with open(orig_yaml, \"r\") as f:\n",
    "        config = yaml.safe_load(f)\n",
    "\n",
    "    # Theo cấu trúc của dataset URPC2020 trên Kaggle\n",
    "    dataset_root = \"/kaggle/input/datasets/lywang777/urpc2020/URPC2020\"\n",
    "    train_images_dir = Path(f\"{dataset_root}/train/images\")\n",
    "    train_labels_dir = Path(f\"{dataset_root}/train/labels\")\n",
    "\n",
    "    if train_images_dir.exists() and train_labels_dir.exists():\n",
    "        all_imgs = sorted(list(train_images_dir.glob(\"*.jpg\")))\n",
    "        HABITAT_TO_URPC = {0: 2, 1: 1, 2: 3, 3: 0}\n",
    "        URPC_TO_HABITAT = {v: k for k, v in HABITAT_TO_URPC.items()}\n",
    "\n",
    "        imgs_by_habitat = {h: [] for h in range(4)}\n",
    "        imgs_noclass = []\n",
    "\n",
    "        print(\"Đang phân loại ảnh theo Habitat...\")\n",
    "        for img_path in all_imgs:\n",
    "            lbl_path = train_labels_dir / (img_path.stem + \".txt\")\n",
    "            counts = {c: 0 for c in range(4)}\n",
    "            try:\n",
    "                if lbl_path.exists():\n",
    "                    with open(lbl_path, \"r\") as lf:\n",
    "                        for line in lf:\n",
    "                            parts = line.strip().split()\n",
    "                            if parts:\n",
    "                                cls_id = int(parts[0])\n",
    "                                if cls_id in counts:\n",
    "                                    counts[cls_id] += 1\n",
    "                    if sum(counts.values()) == 0:\n",
    "                        imgs_noclass.append(img_path)\n",
    "                    else:\n",
    "                        dominant = max(counts, key=counts.get)\n",
    "                        habitat = URPC_TO_HABITAT.get(dominant, 0)\n",
    "                        imgs_by_habitat[habitat].append(img_path)\n",
    "                else:\n",
    "                    imgs_noclass.append(img_path)\n",
    "            except Exception:\n",
    "                imgs_noclass.append(img_path)\n",
    "\n",
    "        for i, img_path in enumerate(imgs_noclass):\n",
    "            imgs_by_habitat[i % 4].append(img_path)\n",
    "\n",
    "        random.seed(1104)\n",
    "        proxy_imgs = []\n",
    "        print(\"\\n[Data Partitioning] Tách 15% Proxy Data từ các Habitat:\")\n",
    "        for h in range(4):\n",
    "            pool = imgs_by_habitat[h]\n",
    "            old_size = len(pool)\n",
    "            proxy_for_h = int(old_size * 0.15)\n",
    "            sampled = random.sample(pool, proxy_for_h)\n",
    "            proxy_imgs.extend(sampled)\n",
    "            print(f\"    - Habitat {h}: Tổng {old_size} ảnh -> Lấy {proxy_for_h} ảnh cho KD\")\n",
    "\n",
    "        proxy_txt = Path(\"datasets/URPC2020/proxy_train.txt\")\n",
    "        proxy_txt.parent.mkdir(parents=True, exist_ok=True)\n",
    "        with open(proxy_txt, \"w\") as f:\n",
    "            for img in proxy_imgs:\n",
    "                f.write(f\"{img.absolute()}\\n\")\n",
    "\n",
    "        # Fix absolute paths from original data.yaml\n",
    "        config[\"path\"] = dataset_root\n",
    "        config[\"train\"] = str(proxy_txt.absolute())\n",
    "        config[\"val\"] = \"valid/images\"\n",
    "        config[\"test\"] = \"test/images\"\n",
    "\n",
    "        with open(proxy_yaml, \"w\") as f:\n",
    "            yaml.dump(config, f)\n",
    "        print(f\"\\n=> Đã tạo {proxy_yaml} với tổng {len(proxy_imgs)} ảnh proxy (15%).\")\n",
    "    else:\n",
    "        print(\"Chưa tìm thấy thư mục ảnh hoặc nhãn. Hãy kiểm tra lại đường dẫn dataset.\")\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('orig_yaml = "/kaggle/input/datasets/lywang777/urpc2020/URPC2020/data.yaml"' in line for line in cell['source']):
        cell['source'] = new_source
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Kaggle_FedKDL.ipynb updated.")
