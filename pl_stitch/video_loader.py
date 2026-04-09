import random
import math
import numpy as np
import os
import lmdb
from pathlib import Path
import torch
import glob
from tqdm import tqdm
from typing import Any, Callable, Optional, List, Tuple, Dict
from PIL import Image
import utils
import cv2
import random, collections
import decord
import json
import collections
from collections import defaultdict
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as T



class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base, times):
        self.dataset  = base
        self.times = times
        self._len  = len(base)

    def __len__(self):
        return self._len * self.times

    def __getitem__(self, idx):
        return self.dataset[idx % self._len]


class Dataset_puzzle(VisionDataset):
    def __init__(
        self,
        lmdb_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.lmdb = lmdb_path
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.data = []
        self.targets = []
        self.num_samples = 0
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            print(f"total number is: {txn.stat()['entries']}")
            cursor = txn.cursor()
            for key in tqdm(cursor.iternext(keys=True, values=False), total=txn.stat()['entries']):
                self.data.append(key)
                self.num_samples += 1


    def load_image_from_lmdb(self, img_key):
        """
        Load an image from LMDB using the provided key.
        """
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(img_key)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        return img

    def get_time_sequence_frame(self, frame_name: str):
        video_name, t_str = frame_name.rsplit('_', 1)
        t = int(t_str)
        vid_b = video_name.encode('utf-8')

        past_offsets = np.array([-3, -2, -1])
        future_offsets = np.array([1, 2, 3])
    
        past_frame = None
        future_frame = None
    
        with self.env.begin(write=False, buffers=True) as txn:
            # --- Step 1: Search for a "past" frame ---
            for off in np.random.permutation(past_offsets):
                tt = t + int(off)
                if tt < 0: continue
                
                nei_key = vid_b + b'_' + str(tt).encode('ascii')
                buf = txn.get(nei_key)
                
                if buf is not None:
                    arr = np.frombuffer(memoryview(buf), dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    past_frame = Image.fromarray(img_bgr)
                    break 
            
            # --- Step 2: Search for a "future" frame ---
            for off in np.random.permutation(future_offsets):
                tt = t + int(off)
                nei_key = vid_b + b'_' + str(tt).encode('ascii')
                buf = txn.get(nei_key)
    
                if buf is not None:
                    arr = np.frombuffer(memoryview(buf), dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    future_frame = Image.fromarray(img_bgr)
                    break
    
            # --- Step 3: Load the current frame ONLY if a neighbor was not found ---
            if past_frame is None or future_frame is None:
                cur_key = frame_name.encode('utf-8')
                buf0 = txn.get(cur_key)
                if buf0 is None:
                    raise KeyError(f"LMDB missing the current frame key: {frame_name}")
                
                arr0 = np.frombuffer(memoryview(buf0), dtype=np.uint8)
                img0_bgr = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
                fallback_img = Image.fromarray(img0_bgr)
    
                # Apply fallback only where needed
                if past_frame is None:
                    past_frame = fallback_img
                if future_frame is None:
                    future_frame = fallback_img
                    
        return past_frame, future_frame
    
    def __getitem__(self, index: int):
        img_key = self.data[index]
        frame_name = img_key.decode('utf-8')  # Assuming the keys are encoded as bytes
        original_img = self.load_image_from_lmdb(img_key)

        # Get time sequence frames
        time_sequence_frames = self.get_time_sequence_frame(frame_name)

        if self.transform is not None:
            imgs = self.transform(original_img, time_sequence_frames)

        return imgs

    def __len__(self) -> int:
        return self.num_samples





class ImageFolderMask_puzzle(Dataset_puzzle):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask_puzzle, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output, puzzle_images = super(ImageFolderMask_puzzle, self).__getitem__(index)
                
        masks = []
        for img in output + [puzzle_images[0]]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count 

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output,) + (masks,) + (puzzle_images,)





class Dataset(VisionDataset):
    def __init__(
        self,
        lmdb_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.lmdb = lmdb_path
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.data = []
        self.targets = []
        self.num_samples = 0
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            print(f"total number is: {txn.stat()['entries']}")
            cursor = txn.cursor()
            for key in tqdm(cursor.iternext(keys=True, values=False), total=txn.stat()['entries']):
                self.data.append(key)
                self.num_samples += 1


    def load_image_from_lmdb(self, img_key):
        """
        Load an image from LMDB using the provided key.
        """
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(img_key)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        return img
    
    def __getitem__(self, index: int):
        img_key = self.data[index]
        frame_name = img_key.decode('utf-8')  # Assuming the keys are encoded as bytes
        img = self.load_image_from_lmdb(img_key)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return self.num_samples





class Temporal_RandStep_dataset(Dataset):
    _RULES: List[Tuple[int, int, int]] = [
        (0,   240, 60),     # segments  8–15  → 100 samples/epoch
        (240,  960, 200),     # segments 16–31  → 200 samples/epoch
        (960,  10**9, 300),  # segments ≥32    → 300 samples/epoch
    ]

    def __init__(
        self,
        lmdb_path: str,
        split_txt: Optional[str] = None,   # kept for API compatibility (unused)
        img_size: int = 224,
        seq_len: int = 8,
        transform: Optional[T.Compose] = None,
        min_step: int = 1,
        max_step: int = 20,
    ):
        # Parent must set self.data and provide load_image_from_lmdb
        super().__init__(lmdb_path, transform=None, size=img_size)

        assert seq_len > 0, "seq_len must be positive"
        assert 1 <= min_step <= max_step, "Require 1 <= min_step <= max_step"
        self.seq_len = seq_len
        self.min_step = min_step
        self.max_step = max_step

        # 1) Group inherited keys by vid -> sorted list of integer frame indices
        self.vid2frames: Dict[str, List[int]] = collections.defaultdict(list)

        def _decode(k):
            return k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)

        for k in getattr(self, "data", []):
            s = _decode(k).strip()
            if not s:
                continue
            try:
                vid, idx_str = s.rsplit("_", 1)
                idx = int(idx_str)
            except Exception:
                # Skip malformed keys silently
                continue
            self.vid2frames[vid].append(idx)
        print(len(self.vid2frames))

        # sort indices per video and drop videos with fewer than seq_len indices
        for vid in list(self.vid2frames.keys()):
            arr = sorted(set(self.vid2frames[vid]))
            if len(arr) < self.seq_len:
                del self.vid2frames[vid]
            else:
                self.vid2frames[vid] = arr

        # 2) Aug pipeline (independent across frames; matches your original)
        self.frame_tx = transform or T.Compose([
            T.Resize((img_size, img_size)),
            #T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            utils.GaussianBlur(0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 3) Build samples for epoch 0
        self._build_samples(seed=0)

    def set_epoch(self, epoch: int):
        """Call once per (global) epoch to resample."""
        self._build_samples(seed=epoch)

    # ---- sampling helpers ----

    def _per_video_repeats(self, n_indices: int) -> int:
        # Proxy “segments” = how many seq_len-sized chunks exist
        segments = max(1, n_indices)
        return next(rep for lo, hi, rep in self._RULES if lo <= segments < hi)

    # ---- core sampler ----

    def _build_samples(self, seed: int):
        rng = random.Random(seed)
        samples: List[List[bytes]] = []

        for vid, idx_list in self.vid2frames.items():
            n = len(idx_list)
            if n < self.seq_len:
                continue

            repeats = self._per_video_repeats(n)

            # Minimal span required to place a clip with min_step
            min_total_span = self.min_step * (self.seq_len - 1)
            if n - 1 < min_total_span:
                # Shouldn't happen since n >= seq_len and min_step >= 1, but guard anyway
                continue

            for _ in range(repeats):
                # 1) Random START that fits with min_step
                max_start_for_min = n - 1 - min_total_span  # inclusive
                start_pos = rng.randint(0, max_start_for_min)

                # 2) Max feasible step for this start (respecting bounds and max_step)
                s_max_feasible = (n - 1 - start_pos) // (self.seq_len - 1)
                s_max = min(self.max_step, s_max_feasible)  # s_max_feasible >= min_step by construction

                # 3) FIXED step for this clip (varies across clips)
                step = rng.randint(self.min_step, s_max)

                # 4) Build positions (start_pos + i*step)
                positions = [start_pos + i * step for i in range(self.seq_len)]

                # 5) Map to frame indices / LMDB keys
                chosen_frame_indices = [idx_list[p] for p in positions]
                keys = [f"{vid}_{t}".encode("utf-8") for t in chosen_frame_indices]
                samples.append(keys)

        self.samples: List[List[bytes]] = samples


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        keys = self.samples[idx]  # list of seq_len keys

        frames = [self.frame_tx(self.load_image_from_lmdb(k)) for k in keys]
        return torch.stack(frames, 0)  # [seq_len, 3, H, W]



