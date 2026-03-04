import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, batch_size=32, is_train=True, num_workers=2):
    """
    Hàm khởi tạo DataLoader cho bộ dữ liệu khuôn mặt.
    
    Args:
        data_dir (str): Đường dẫn đến thư mục chứa dữ liệu (VD: 'data/casia_webface_clean')
        batch_size (int): Số lượng ảnh trong một batch
        is_train (bool): Nếu True, áp dụng Data Augmentation. Nếu False, chỉ Resize và Normalize.
        num_workers (int): Số lượng luồng xử lý dữ liệu song song.
        
    Returns:
        DataLoader, int: Trả về đối tượng DataLoader và tổng số class (số người)
    """
    
    # 1. Định nghĩa các phép biến đổi ảnh (Transforms)
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((112, 112)), # Kích thước chuẩn cho ArcFace/MobileFaceNet
            transforms.RandomHorizontalFlip(p=0.5), # Lật ngang ngẫu nhiên
            # Thêm thay đổi độ sáng/tương phản nhẹ để quen với camera Kiosk
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), 
            transforms.ToTensor(),
            # Đưa giá trị pixel từ [0, 1] về dải [-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])
    else:
        # Khi test/validate thì không dùng Augmentation
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # 2. Load Dataset bằng ImageFolder
    # Vì thư mục tổ chức theo dạng: data_dir/class_id/image.jpg
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    num_classes = len(dataset.classes)

    