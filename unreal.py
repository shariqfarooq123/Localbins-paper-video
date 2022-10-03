import os
from PIL import Image
import numpy as np
import matplotlib.cm


UNREAL_EXAMPLES = ["06495", "07620", "08910", "09092", "09541", "10362"]
UNREAL_EXAMPLES_ROOT = "./assets/unreal"

def load_sample(img_path, gt_path):
        img = np.asarray(Image.open(img_path), dtype=np.uint8)
        gt = np.array(Image.open(gt_path), dtype=np.uint8)[..., 0]

        # Normalize GT to 0 to 1
        gt = gt / 255.0
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        # colorize the ground truth image using magma_r colormap and convert to numpy array
        cmap = matplotlib.cm.get_cmap("magma_r")
        gt = cmap(gt)[:, :, :3]
        # Inverse Gamma correct GT
        gt = np.power(gt, 2.2)
        gt = np.array(gt * 255, dtype=np.uint8)

        return img, gt


class UnrealExamples:
    def __init__(self):
        """
        Unreal example images
        .get() returns a sample
        samples are cycled through
        """
        self.image_dir = UNREAL_EXAMPLES_ROOT
        self.label_dir = UNREAL_EXAMPLES_ROOT
        self.image_files = [os.path.join(UNREAL_EXAMPLES_ROOT, f"{i}.jpg") for i in UNREAL_EXAMPLES]
        self.label_files = [os.path.join(UNREAL_EXAMPLES_ROOT, f"{i}.png") for i in UNREAL_EXAMPLES]
        self.idx = 0

    def __len__(self):
        return len(self.image_files)

    def get(self):
        img_path = self.image_files[self.idx]
        gt_path = self.label_files[self.idx]
        self.idx = (self.idx + 1) % len(self.image_files)
        return load_sample(img_path, gt_path)