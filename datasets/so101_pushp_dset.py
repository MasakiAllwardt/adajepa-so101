"""
SO-101 PushP dataset: PushTDataset adapted for the metric SO-101 data emitted by
collect_data.py. Same on-disk layout (states.pth / rel_actions.pth /
velocities.pth / seq_lengths.pkl / shapes.pkl / obses/episode_%03d.mp4), but:

  * units are METRES/RADIANS, not pixels — PushT's hardcoded pixel-space stats
    (pusht_dset.py:14-19) are invalid here
  * stats therefore come from the stats.json that collect_data.py writes next to
    the train/ and val/ splits, loaded by load_stats(); they regenerate together
    with the data, so they can never go stale the way pasted constants would
  * no abs_actions variant: the collector stores commanded relative deltas only
"""

import json
import pickle
from pathlib import Path
from typing import Callable, Optional

import decord
import numpy as np
import torch
from decord import VideoReader
from einops import rearrange

from datasets.traj_dset import TrajDataset, TrajSlicerDataset

decord.bridge.set_bridge("torch")

STATS_KEYS = ("ACTION_MEAN", "ACTION_STD", "STATE_MEAN", "STATE_STD",
              "PROPRIO_MEAN", "PROPRIO_STD")


def load_stats(stats_path):
    """Load normalization stats from collect_data.py's stats.json -> dict of
    1-D float tensors. Raises with a pointed message if missing: training or
    planning with wrong stats fails silently and expensively otherwise."""
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found. collect_data.py writes stats.json at the "
            f"dataset root (next to train/ and val/). Regenerate the dataset or "
            f"point data_path at the right root.")
    raw = json.loads(stats_path.read_text())
    missing = [k for k in STATS_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{stats_path} is missing keys: {missing}")
    return {k: torch.tensor(raw[k], dtype=torch.float32) for k in STATS_KEYS}


class SO101PushPDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/so101_pushp/train",
        normalize_action: bool = True,
        with_velocity: bool = True,
        stats_path: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "rel_actions.pth").float()
        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, "rb") as f:
                self.shapes = pickle.load(f)
        else:
            self.shapes = ["P"] * len(self.states)

        n = self.n_rollout = n_rollout or len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()

        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts from {self.data_path}")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            # stats.json lives at the dataset root, one level above train/ / val/
            stats = load_stats(stats_path or self.data_path.parent / "stats.json")
            self.action_mean = stats["ACTION_MEAN"]
            self.action_std = stats["ACTION_STD"]
            self.state_mean = stats["STATE_MEAN"][: self.state_dim]
            self.state_std = stats["STATE_STD"][: self.state_dim]
            self.proprio_mean = stats["PROPRIO_MEAN"][: self.proprio_dim]
            self.proprio_std = stats["PROPRIO_STD"][: self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {"shape": shape}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_so101_pushp_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/so101_pushp",
    normalize_action=True,
    split_ratio=0.9,        # inert, kept for config parity with pusht
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
):
    train_dset = SO101PushPDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )
    val_dset = SO101PushPDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
