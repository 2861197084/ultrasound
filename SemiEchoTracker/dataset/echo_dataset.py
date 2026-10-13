import os
import torch
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np
from torchvision import transforms as T
import random
import albumentations as A
import hashlib
import json
from pathlib import Path

class EchoDataset(Dataset):
    def __init__(self, root_dir="", transform=None, frame_num=10, splits='train',
                 target_size=(224, 224), data_type="PLAX",
                 rotation_limit=30.0, rotation_prob=0.5):
        target_data_dir = os.path.join(root_dir, data_type)
        if not os.path.exists(target_data_dir):
            raise FileNotFoundError(f"Directory {target_data_dir} does not exist")
        self.root_dir = target_data_dir
        self.transform = transform
        self.normalize = T.Compose([
            T.ToTensor(), 
            T.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225])
        ])
        self.target_size = target_size
        self.frame_num = frame_num
        self.splits = splits
        self.data_type = data_type
        self.rotation_limit = float(rotation_limit)
        self.rotation_prob = float(rotation_prob)
        self.data_list = self.read_data()
        self.rotation = splits == 'train' and self.rotation_limit > 0 and self.rotation_prob > 0
        if len(self.data_list) == 0:
            raise ValueError(f"No valid data found in {self.root_dir}")
        print(f'{len(self.data_list)} patients selected for {self.splits} set in {self.root_dir}')
        
    def read_data(self):
        print(f'Reading data from {self.root_dir}/{self.splits}.txt')
        with open(f'{self.root_dir}/{self.splits}.txt', 'r') as f:
            patient_list = f.read().splitlines()
        data_all = []
        print(len(patient_list))
        for patient in patient_list:
            patient_dir = os.path.join(self.root_dir, "selected_data", patient)
            if not os.path.isdir(patient_dir):
                continue
            anno_file_list = glob.glob(f'{self.root_dir}/selected_data/{patient}/data.npz')
            base_data_anno = np.load(anno_file_list[0], allow_pickle=True)
            landmarks = base_data_anno['landmarks']
            image_list = base_data_anno['video']
            data_all.append({
                'patient': patient,
                'landmarks': landmarks,
                'image_list': image_list
            })
        return data_all

    def __len__(self):
        return len(self.data_list)

    def rotate_landmarks(self, landmarks, angle, center):
        """
        Rotate landmarks around center point
        Args:
            landmarks: array of shape (N, 2) containing x,y coordinates
            angle: rotation angle in degrees
            center: tuple of (x,y) rotation center coordinates
        """
        angle_rad = np.deg2rad(angle)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        landmarks = landmarks - center
        landmarks = landmarks @ rot_matrix
        landmarks = landmarks + center
        return landmarks

    def __getitem__(self, idx):
        meta_data = self.data_list[idx]
        patient_name = meta_data['patient']
        image_list = meta_data['image_list']
        landmarks_raw = np.array(meta_data['landmarks'])
        has_framewise_landmarks = len(landmarks_raw) == len(image_list)
        has_endpoint_landmarks = len(landmarks_raw) == 2

        # Decide the clip-level rotation angle (applied to every frame / landmark)
        if self.rotation:
            self.current_angle = random.uniform(-self.rotation_limit, self.rotation_limit) \
                if random.random() < self.rotation_prob else 0

        # Whether the transform supports replay (ReplayCompose). We apply the
        # same pixel-level augmentation to every frame so the appearance is
        # consistent across time (critical for the tracker).
        replay_params = None
        is_replay_compose = isinstance(self.transform, A.ReplayCompose)

        input_data = []
        scale_x = scale_y = rescale_x = rescale_y = None
        for index, image in enumerate(image_list):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                orig_h, orig_w = image.shape[:2]
                scale_x = self.target_size[1] / orig_w
                scale_y = self.target_size[0] / orig_h
                rescale_x = orig_w / self.target_size[1]
                rescale_y = orig_h / self.target_size[0]
                image = cv2.resize(image, self.target_size)

                if self.rotation:
                    image = A.rotate(image, angle=self.current_angle)

                if is_replay_compose:
                    if replay_params is None:
                        aug = self.transform(image=image)
                        replay_params = aug['replay']
                    else:
                        aug = A.ReplayCompose.replay(replay_params, image=image)
                    image = aug['image']
                else:
                    aug = self.transform(image=image)
                    image = aug['image']
            image = self.normalize(image)
            input_data.append(image)
        input_data = torch.stack(input_data)
        landmarks = landmarks_raw * np.array([scale_x, scale_y])
        
        # Apply rotation to landmarks if needed
        if self.rotation:
            center = np.array([self.target_size[1]/2, self.target_size[0]/2])  # rotation center (x,y)
            # Rotate both ED and ES frames
            landmarks[0] = self.rotate_landmarks(landmarks[0], self.current_angle, center)
            landmarks[-1] = self.rotate_landmarks(landmarks[-1], self.current_angle, center)
            
        # check if need reverse
        if self.data_type == "A4C":
            area_first_frame = cv2.contourArea(landmarks[0].astype(np.int32))
            area_last_frame = cv2.contourArea(landmarks[-1].astype(np.int32))
            if area_first_frame > area_last_frame:
                landmarks = np.flip(landmarks, axis=0).copy()
                input_data = torch.flip(input_data, dims=[0]).contiguous()
        elif self.data_type == "PLAX":
            check_index_1, check_index_2 = 3, 6
            dis_first_frame = np.sqrt((landmarks_raw[0, check_index_1,0] - landmarks_raw[0, check_index_2,0])**2 + (landmarks_raw[0, check_index_1,1] - landmarks_raw[0, check_index_2,1])**2)
            dis_last_frame  = np.sqrt((landmarks_raw[-1, check_index_1,0] - landmarks_raw[-1, check_index_2,0])**2 + (landmarks_raw[-1, check_index_1,1] - landmarks_raw[-1, check_index_2,1])**2)
            if dis_first_frame > dis_last_frame:
                landmarks = np.flip(landmarks, axis=0).copy()
                input_data = torch.flip(input_data, dims=[0]).contiguous()
        # sample self.frame_num
        if len(input_data) > self.frame_num:
            if self.frame_num >= 2:
                middle_indices = np.linspace(1, len(input_data)-2, self.frame_num-2, dtype=int)
                indices = np.concatenate([[0], middle_indices, [len(input_data)-1]])
            else:
                indices = np.linspace(0, len(input_data)-1, self.frame_num, dtype=int)
            input_data = input_data[indices]
            if has_framewise_landmarks:
                landmarks = landmarks[indices]
            elif has_endpoint_landmarks:
                landmarks = landmarks[[0, -1]]
            else:
                raise ValueError(
                    f"Unsupported landmark count {len(landmarks)} for {len(image_list)} frames in {patient_name}"
                )
        if self.splits == 'train':
            landmarks = landmarks[[0, -1]]
        return {
            'video': input_data,
            'landmarks': landmarks,
            'patient_name': patient_name,
            'rescale_x': rescale_x,
            'rescale_y': rescale_y
        }


class EchoXFlowDataset(EchoDataset):
    """固定 10 帧数据；训练缓存只包含端点坐标，不读取原始轮廓。"""

    def read_data(self):
        if self.frame_num != 10:
            raise ValueError('EchoXFlow prepared protocol requires frame_num=10')
        root = Path(self.root_dir)
        self.manifest_path = root / 'manifest.json'
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        manifest = json.loads(self.manifest_path.read_text())
        if not manifest['complete'] or manifest['dataset'] != 'EchoXFlow A4C':
            raise ValueError('Incomplete or incompatible EchoXFlow preparation')
        split_path = root / f'{self.splits}.txt'
        if hashlib.sha256(split_path.read_bytes()).hexdigest() != manifest['split_sha256'][self.splits]:
            raise ValueError('EchoXFlow split checksum mismatch')
        ids = split_path.read_text().splitlines()
        records = {row['sample_id']: row for row in manifest['samples'] if row['split'] == self.splits}
        if len(ids) != len(set(ids)) or set(ids) != set(records):
            raise ValueError('EchoXFlow split does not match manifest')
        data = []
        for sid in ids:
            path = root / records[sid]['data_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != records[sid]['data_sha256']:
                raise ValueError(f'EchoXFlow data checksum mismatch: {sid}')
            with np.load(path, allow_pickle=False) as archive:
                item = {key: archive[key] for key in archive.files}
            expected_frames = np.array([0, 9]) if self.splits == 'train' else np.arange(10)
            if (item['video'].shape != (10, 224, 224, 3)
                    or item['landmarks'].shape != (len(expected_frames), 3, 2)
                    or not np.array_equal(item['label_frame_indices'], expected_frames)
                    or not item['valid_mask'].all()):
                raise ValueError(f'EchoXFlow label scope or shape mismatch: {sid}')
            item['patient'] = sid
            data.append(item)
        return data

    def __getitem__(self, idx):
        item = self.data_list[idx]
        native_h, native_w = item['video'].shape[1:3]
        height, width = self.target_size
        sx, sy = width / native_w, height / native_h
        # OpenCV resize 使用像素中心；随后对图像和全部已提供标签使用同一旋转矩阵。
        resize = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]], dtype=np.float64)
        angle = (random.uniform(-self.rotation_limit, self.rotation_limit)
                 if self.rotation and random.random() < self.rotation_prob else 0.)
        rotate = np.eye(3)
        rotate[:2] = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), angle, 1.)
        native_to_input = rotate @ resize
        landmarks = (item['landmarks'] @ native_to_input[:2, :2].T + native_to_input[:2, 2]).astype(np.float32)
        replay = None
        frames = []
        for image in item['video']:
            image = cv2.resize(image, (width, height))
            if angle:
                image = cv2.warpAffine(image, rotate[:2], (width, height),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if self.transform:
                if isinstance(self.transform, A.ReplayCompose):
                    augmented = (self.transform(image=image) if replay is None
                                 else A.ReplayCompose.replay(replay, image=image))
                    replay = augmented['replay']
                else:
                    augmented = self.transform(image=image)
                image = augmented['image']
            frames.append(self.normalize(image))
        return dict(video=torch.stack(frames), landmarks=landmarks, patient_name=item['patient'],
                    label_frame_indices=item['label_frame_indices'].copy(),
                    frame_indices=item['frame_indices'].copy(),
                    timestamps_seconds=item['timestamps_seconds'].copy(),
                    input_to_mm=item['native_to_mm'] @ np.linalg.inv(native_to_input),
                    rescale_x=1/sx, rescale_y=1/sy)


def make_dataset(**kwargs):
    dataset_class = EchoXFlowDataset if kwargs.get('data_type') == 'echoxflow' else EchoDataset
    return dataset_class(**kwargs)
