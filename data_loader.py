import pathlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import downsample_image

class twoD_Data(Dataset):
    def __init__(self, patients, mode, data_path, apply_augmentation=None):
        self.patients = patients
        self.mode = mode
        self.data_path = data_path
        self.resolution = 64
        self.low_res = 32
        self.patient_list = [f'Patient{p}' for p in self.patients]
        self.met_list = ['Cr+PCr', 'Gln', 'Glu', 'Gly', 'GPC+PCh', 'Ins', 'NAA']

        # augment only during training
        self.apply_augmentation = (self.mode == 'train') if apply_augmentation is None else bool(apply_augmentation)

        self.patient_folders = list(pathlib.Path(self.data_path).iterdir())
        self.examples = []
        self.slices = []
        self.fnames = []
        self.fname2nslice = {}


        for patient in sorted(self.patient_folders):
            patientname = str(patient.name)
            if patientname not in self.patient_list:
                continue
            met = np.load(str(patient) + '/Met_filtered/Gln.npy')
            num_slices = met.shape[0]
            self.examples += [(patient, slice, met) for slice in range(num_slices) for met in self.met_list]
            self.slices += [(str(patient), slice) for slice in range(num_slices)]
            self.fnames += [patientname]
            self.fname2nslice[patientname] = num_slices

    def __len__(self):
        return len(self.examples)

    # create random horizontal/vertical flips and shifts to augment small training dataset
    def apply_random_flip(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])  # horizontal
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-2])  # vertical
        return image

    def apply_random_shift(self, image: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
        _, h, w = image.shape
        sh = random.randint(-max_shift, max_shift)
        sw = random.randint(-max_shift, max_shift)
        shifted = torch.zeros_like(image)

        src_h0, src_w0 = max(0, sh), max(0, sw)
        src_h1, src_w1 = min(h, h + sh), min(w, w + sw)
        tgt_h0, tgt_w0 = max(0, -sh), max(0, -sw)
        tgt_h1, tgt_w1 = min(h, h - sh), min(w, w - sw)

        hr, wr = src_h1 - src_h0, src_w1 - src_w0
        if hr > 0 and wr > 0:
            shifted[:, tgt_h0:tgt_h0+hr, tgt_w0:tgt_w0+wr] = image[:, src_h0:src_h0+hr, src_w0:src_w0+wr]
        return shifted

    def __getitem__(self, idx):
        patient_path, slice_idx, metname = self.examples[idx]

        # load HR metabolite slice
        met_HR = np.load(str(patient_path / 'Met_filtered' / f'{metname}.npy'))[slice_idx]
        met_HR = torch.from_numpy(met_HR).float()
        met_max = float(met_HR.max())
        if met_max > 0:
            met_HR = met_HR / met_max
        met_HR = met_HR.unsqueeze(0) if met_HR.ndim == 2 else met_HR  # [1, H, W]

        # apply augmentations
        if self.apply_augmentation and random.random() > 0.5:
            aug_type = random.choice(['flip', 'shift', 'both'])
            if aug_type in ('flip', 'both'):
                met_HR = self.apply_random_flip(met_HR)
            if aug_type in ('shift', 'both'):
                met_HR = self.apply_random_shift(met_HR, max_shift=2)

        # downsample metabolic map to low resolution [kspace: 64x64 -> kspace: 32x32]
        lr_image = downsample_image(met_HR, 2)  # expects [1,H,W]

        # metadata (metabolic maps, maximum metabolite value (normalization), patient name, slice index, metabolite name)
        patient_name = str(patient_path.name)
        return lr_image, met_HR, met_max, patient_name, slice_idx, metname


# create dataloaders based off of specified train/valid/test patient splits
def create_data_loaders(data_path, train_patients, valid_patients, test_patients, batch_size=16, num_workers=4):
    train_dataset = twoD_Data(train_patients, mode='train', data_path=data_path)
    valid_dataset = twoD_Data(valid_patients, mode='valid', data_path=data_path, apply_augmentation=False)
    test_dataset = twoD_Data(test_patients, mode='test', data_path=data_path, apply_augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, valid_loader, test_loader
