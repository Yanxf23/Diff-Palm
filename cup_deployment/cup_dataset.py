import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import blobfile as bf
import json
import multiprocessing
from PCEM_numpy import process_images
import sys
sys.path.append("../DiffModels")
from guided_diffusion import dist_util, logger

def _list_image_files_recursively(folder):
    exts = [".png", ".jpg", ".jpeg"]
    results = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                results.append(os.path.join(root, f))
    return sorted(results)

def generate_labels_with_pcem(spath, dpath, remove_json="remove.json", num_process=16):
    spath = r"C:\Users\mobil\Desktop\25spring\stylePalm\evaluation\datasets\cup_final\val"
    dpath = r"./cup_final_label/val"
    
    os.makedirs(rf"{dpath}", exist_ok=True)

    if os.path.exists(remove_json):
        with open(remove_json, "r") as f:
            removed_items = json.load(f)["images_to_remove"]  
    removed_imgs = []
    for removed in removed_items:
        removed_imgs.append(removed["path"])

    src_imgs = _list_image_files_recursively(spath)

    filtered_imgs = []
    save_files = []
    for img_path in src_imgs:
        fname = os.path.join(*img_path.split(os.sep)[-4:])
        if fname in removed_imgs:
            # print(f"Skipping {img_path} as it is in the remove list.")
            continue
        rel_path = os.path.relpath(img_path, spath)
        save_path = os.path.join(dpath, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        filtered_imgs.append(img_path)
        save_files.append(save_path)

    assert len(filtered_imgs) == len(save_files)
    process_list = []
    for i in range(num_process):
        p = multiprocessing.Process(
            target=process_images,
            args=(filtered_imgs[i::num_process], save_files[i::num_process])
        )
        process_list.append(p)

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()

    print(f"{spath} --> {dpath} : Done (processed {len(filtered_imgs)} files)")

class CupDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        resolution,
        remove_json=None,
        include_key=None,
        save_debug_dir=None,
        random_crop=True,
        random_flip=True,
    ):
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.save_debug_dir = save_debug_dir
        self.include_key = include_key
        self.to_remove = set()
        self.save_counter = 0

        if remove_json and os.path.exists(remove_json):
            with open(remove_json, 'r') as f:
                remove_data = json.load(f)
                for item in remove_data.get("images_to_remove", []):
                    user = os.path.normpath(item["path"]).split("\\")[-2]
                    name = os.path.normpath(item["path"]).split("\\")[-1]
                    # user = os.path.normpath(item["path"]).split(os.sep)[-2]
                    # name = os.path.basename(item["path"])
                    # print(f"Removing {user}/{name} from dataset")
                    self.to_remove.add((user, name))

        self.image_paths, self.label_paths = self._filter_images(image_paths, label_paths)

        if self.save_debug_dir:
            os.makedirs(self.save_debug_dir, exist_ok=True)
            print(f"Debug saving directory: {self.save_debug_dir}")

    def _filter_images(self, image_paths, label_paths):
        filtered_images = []
        filtered_labels = []
        for img_path, lbl_path in zip(image_paths, label_paths):
            user = os.path.normpath(img_path).split(os.sep)[-2]
            name = os.path.basename(img_path)
            if (user, name) in self.to_remove:
                continue
            if self.include_key and self.include_key not in img_path:
                continue
            filtered_images.append(img_path)
            filtered_labels.append(lbl_path)
        return filtered_images, filtered_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        with bf.BlobFile(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        with bf.BlobFile(lbl_path, "rb") as f:
            lbl = Image.open(f).convert("L")

        resize_size = int(self.resolution * 1.1)
        img = img.resize((resize_size, resize_size), Image.BICUBIC)
        lbl = lbl.resize((resize_size, resize_size), Image.BICUBIC)

        if self.random_crop:
            crop_y = random.randint(0, img.size[1] - self.resolution)
            crop_x = random.randint(0, img.size[0] - self.resolution)
            img = TF.crop(img, crop_y, crop_x, self.resolution, self.resolution)
            lbl = TF.crop(lbl, crop_y, crop_x, self.resolution, self.resolution)
        else:
            img = TF.center_crop(img, self.resolution)
            lbl = TF.center_crop(lbl, self.resolution)

        if self.random_flip and random.random() < 0.5:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)

        # Save side-by-side visualization (only for first 10 samples)
        if self.save_debug_dir and self.save_counter < 10:
            concat = Image.new("RGB", (img.width + lbl.width, img.height))
            concat.paste(img, (0, 0))
            concat.paste(lbl.convert("RGB"), (img.width, 0))
            concat.save(os.path.join(self.save_debug_dir, f"pair_{self.save_counter}.png"))
            self.save_counter += 1

        img_tensor = TF.to_tensor(img) * 2.0 - 1.0
        lbl_tensor = TF.to_tensor(lbl) * 2.0 - 1.0

        return img_tensor, {"low_res": lbl_tensor}



def load_palm_cup_data(
    raw_dir,
    label_dir,
    data_type,
    batch_size,
    image_size,
    remove_json=None,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    include_key=None,
    save_debug_dir=None,
):  
    logger.log("creating data loader...")
    palm_paths = _list_image_files_recursively(os.path.join(raw_dir, data_type))
    label_paths = _list_image_files_recursively(os.path.join(label_dir, data_type))
    dataset = CupDataset(
        palm_paths,
        label_paths,
        resolution=image_size,
        remove_json=remove_json,
        random_crop=random_crop,
        random_flip=random_flip,
        include_key=include_key,
        save_debug_dir=save_debug_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )
    while True:
        yield from loader

