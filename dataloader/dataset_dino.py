import cv2
from torchvision import transforms
from dataloader.dataset import DataSet


class DataSet_DINO(DataSet):
    """DINOv2 dataset with ImageNet normalization."""

    def __init__(self, data_path, dataset, fn, split):
        super().__init__(data_path, dataset, fn, split)

        # DINOv2 (ViT-B/14) works best with multiples of 14. 
        # 518 is a standard size (37x37 patches).
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        # Load the image (DataSet._load_img returns BGR)
        im = self._load_img(index)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Apply Transforms (Resize -> Tensor -> Normalize)
        im_tensor = self.transform(im)

        return im_tensor
