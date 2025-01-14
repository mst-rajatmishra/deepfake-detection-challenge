import os
import random
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional


class UnlabeledVideoDataset(Dataset):
    """
    A Dataset class for loading unlabeled video data, where each video is represented as a sequence of frames.
    """
    def __init__(self, root_dir: str, content: Optional[List[str]] = None, transform: Optional[callable] = None):
        """
        Args:
            root_dir (str): Path to the root directory containing video files.
            content (list, optional): List of video file paths. If None, all .mp4 files in the directory are used.
            transform (callable, optional): A function to apply transformations to the video frames.
        """
        self.root_dir = os.path.normpath(root_dir)
        self.transform = transform

        if content is not None:
            self.content = content
        else:
            self.content = []
            for path in glob.iglob(os.path.join(self.root_dir, '**', '*.mp4'), recursive=True):
                rel_path = path[len(self.root_dir) + 1:]
                self.content.append(rel_path)
            self.content = sorted(self.content)

    def __len__(self) -> int:
        """
        Returns the total number of video files in the dataset.
        """
        return len(self.content)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int): Index of the video file to load.

        Returns:
            dict: Dictionary containing the video frames and the index.
        """
        rel_path = self.content[idx]
        path = os.path.join(self.root_dir, rel_path)

        capture = cv2.VideoCapture(path)
        frames = []

        if capture.isOpened():
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                if self.transform is not None:
                    frame = self.transform(frame)

                frames.append(frame)

        capture.release()  # Release the video capture resource

        sample = {
            'frames': frames,
            'index': idx
        }

        return sample


class FaceDataset(Dataset):
    """
    A Dataset class for loading face images.
    """
    def __init__(self, root_dir: str, content: List[str], labels: Optional[List[int]] = None, transform: Optional[callable] = None):
        """
        Args:
            root_dir (str): Path to the root directory containing face images.
            content (list): List of image file paths.
            labels (list, optional): List of labels for the images.
            transform (callable, optional): A function to apply transformations to the face images.
        """
        self.root_dir = os.path.normpath(root_dir)
        self.content = content
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the number of face images in the dataset.
        """
        return len(self.content)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int): Index of the face image to load.

        Returns:
            dict: Dictionary containing the face image and the index (and label if available).
        """
        rel_path = self.content[idx]
        path = os.path.join(self.root_dir, rel_path)

        face = cv2.imread(path, cv2.IMREAD_COLOR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            face = self.transform(image=face)['image']

        sample = {
            'face': face,
            'index': idx
        }

        if self.labels is not None:
            sample['label'] = self.labels[idx]

        return sample


class TrackPairDataset(Dataset):
    FPS = 30

    def __init__(self, tracks_root: str, pairs_path: str, indices: List[int], track_length: int, 
                 track_transform: Optional[callable] = None, image_transform: Optional[callable] = None,
                 sequence_mode: bool = True):
        """
        Args:
            tracks_root (str): Path to the root directory containing video tracks.
            pairs_path (str): Path to the file containing pairs of video tracks (real, fake).
            indices (list): List of indices to sample frames from each track.
            track_length (int): Length of the track to be sampled.
            track_transform (callable, optional): Function to transform the entire video track.
            image_transform (callable, optional): Function to transform individual frames.
            sequence_mode (bool, optional): Whether to apply the same random seed across the sequence.
        """
        self.tracks_root = os.path.normpath(tracks_root)
        self.track_transform = track_transform
        self.image_transform = image_transform
        self.indices = np.asarray(indices, dtype=np.int32)
        self.track_length = track_length
        self.sequence_mode = sequence_mode

        # Load pairs into memory for better performance
        self.pairs = []
        with open(pairs_path, 'r') as f:
            self.pairs = [tuple(line.strip().split(',')) for line in f]

    def __len__(self) -> int:
        """
        Returns the number of track pairs in the dataset.
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int): Index of the track pair to load.

        Returns:
            dict: Dictionary containing the real and fake tracks.
        """
        real_track_path, fake_track_path = self.pairs[idx]

        real_track_path = os.path.join(self.tracks_root, real_track_path)
        fake_track_path = os.path.join(self.tracks_root, fake_track_path)

        # Apply transformation to the tracks if needed
        if self.track_transform is not None:
            img = self.load_img(real_track_path, 0)
            src_height, src_width = img.shape[:2]
            track_transform_params = self.track_transform.get_params(self.FPS, src_height, src_width)
        else:
            track_transform_params = None

        real_track = self.load_track(real_track_path, self.indices, track_transform_params)
        fake_track = self.load_track(fake_track_path, self.indices, track_transform_params)

        # Apply image transformation to each frame if needed
        if self.image_transform is not None:
            prev_state = random.getstate()  # Save random state for deterministic transformations
            real_track = [self.image_transform(image=img)['image'] for img in real_track]
            random.setstate(prev_state)
            fake_track = [self.image_transform(image=img)['image'] for img in fake_track]

        sample = {
            'real': real_track,
            'fake': fake_track
        }

        return sample

    def load_img(self, track_path: str, idx: int) -> np.ndarray:
        """
        Loads a single image from the specified track.

        Args:
            track_path (str): Path to the track directory.
            idx (int): Index of the image to load.

        Returns:
            np.ndarray: The loaded image in RGB format.
        """
        img = cv2.imread(os.path.join(track_path, f"{idx}.png"))
        if img is None:
            raise FileNotFoundError(f"Image {idx}.png not found in {track_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_track(self, track_path: str, indices: np.ndarray, transform_params: Optional[dict]) -> np.ndarray:
        """
        Loads a sequence of frames from the specified track.

        Args:
            track_path (str): Path to the track directory.
            indices (np.ndarray): Indices of the frames to load.
            transform_params (dict, optional): Transformation parameters if needed.

        Returns:
            np.ndarray: The sequence of frames.
        """
        if transform_params is None:
            track = np.stack([self.load_img(track_path, idx) for idx in indices])
        else:
            track = self.track_transform(track_path, self.FPS, *transform_params)
            indices = (indices.astype(np.float32) / self.track_length) * len(track)
            indices = np.round(indices).astype(np.int32).clip(0, len(track) - 1)
            track = track[indices]

        return track
