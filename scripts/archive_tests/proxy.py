import os, random, yaml
from pathlib import Path

orig_yaml = "datasets/URPC2020/URPC2020/data.yaml"
proxy_yaml = "datasets/URPC2020_proxy.yaml"

if not os.path.exists(orig_yaml):
    print(f"Không tìm thấy {orig_yaml}")
else:
    with open(orig_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Theo cấu trúc của dataset URPC2020 trên Kaggle
    dataset_root = "datasets/URPC2020/URPC2020"
    train_images_dir = Path(f"{dataset_root}/train/images")
    train_labels_dir = Path(f"{dataset_root}/train/labels")

    if train_images_dir.exists() and train_labels_dir.exists():
        all_imgs = sorted(list(train_images_dir.glob("*.jpg")))
        HABITAT_TO_URPC = {0: 2, 1: 1, 2: 3, 3: 0}
        URPC_TO_HABITAT = {v: k for k, v in HABITAT_TO_URPC.items()}

        imgs_by_habitat = {h: [] for h in range(4)}
        imgs_noclass = []

        print("Đang phân loại ảnh theo Habitat...")
        for img_path in all_imgs:
            lbl_path = train_labels_dir / (img_path.stem + ".txt")
            counts = {c: 0 for c in range(4)}
            try:
                if lbl_path.exists():
                    with open(lbl_path, "r") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                if cls_id in counts:
                                    counts[cls_id] += 1
                    if sum(counts.values()) == 0:
                        imgs_noclass.append(img_path)
                    else:
                        dominant = max(counts, key=counts.get)
                        habitat = URPC_TO_HABITAT.get(dominant, 0)
                        imgs_by_habitat[habitat].append(img_path)
                else:
                    imgs_noclass.append(img_path)
            except Exception:
                imgs_noclass.append(img_path)

        for i, img_path in enumerate(imgs_noclass):
            imgs_by_habitat[i % 4].append(img_path)

        random.seed(1104)
        proxy_imgs = []
        print("\n[Data Partitioning] Tách 15% Proxy Data từ các Habitat:")
        for h in range(4):
            pool = imgs_by_habitat[h]
            old_size = len(pool)
            proxy_for_h = int(old_size * 0.15)
            sampled = random.sample(pool, proxy_for_h)
            proxy_imgs.extend(sampled)
            print(f"    - Habitat {h}: Tổng {old_size} ảnh -> Lấy {proxy_for_h} ảnh cho KD")

        proxy_txt = Path("datasets/URPC2020/proxy_train.txt")
        proxy_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(proxy_txt, "w") as f:
            for img in proxy_imgs:
                f.write(f"{img.absolute()}\n")

        # Fix absolute paths from original data.yaml
        config["path"] = dataset_root
        config["train"] = str(proxy_txt.absolute())
        config["val"] = "valid/images"
        config["test"] = "test/images"

        with open(proxy_yaml, "w") as f:
            yaml.dump(config, f)
        print(f"\n=> Đã tạo {proxy_yaml} với tổng {len(proxy_imgs)} ảnh proxy (15%).")
    else:
        print("Chưa tìm thấy thư mục ảnh hoặc nhãn. Hãy kiểm tra lại đường dẫn dataset.")
