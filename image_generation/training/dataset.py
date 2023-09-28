import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import pickle

from compute_embed_mean import TEXT_EMBED_MEAN_LAFITE, IMAGE_EMBED_MEAN_LAFITE

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        use_clip    = False,
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        ratio = 1.0,  # how many text-image pairs will be used (0.5 means 0.5 image-text pairs + 0.5 fake pairs. Note if one want to use only 0.5 image-text pairs without using the rest images, set max_size)
        remove_mean=False,
        add_noise=False,
        noise_level=0.75,
        add_lafite_noise=False,
        norm_after_remove_mean=False
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._use_clip = use_clip
        self._raw_labels = None
        self._raw_clip_txt_features = None
        self._raw_clip_img_features = None
        self._label_shape = None
        self._ratio = ratio
        
        self.add_noise = add_noise
        self.remove_mean = remove_mean
        self.noise_level = noise_level
        self.add_lafite_noise = add_lafite_noise
        self.norm_after_remove_mean = norm_after_remove_mean
        
        # Load means gap
        self.text_embed_mean = None
        self.image_embed_mean = None
        
        if remove_mean:
            with open(TEXT_EMBED_MEAN_LAFITE, 'rb') as f:
                self.text_embed_mean = pickle.load(f).numpy()

            with open(IMAGE_EMBED_MEAN_LAFITE, 'rb') as f:
                self.image_embed_mean = pickle.load(f).numpy()

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            print("=> Start loading raw labels")
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            print("=> Finished loading raw labels")
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def _get_clip_img_features(self):
        if self._raw_clip_img_features is None:
            self._raw_clip_img_features = self._load_clip_img_features() if self._use_clip else None
        return self._raw_clip_img_features

    def _get_clip_txt_features(self):
        if self._raw_clip_txt_features is None:
            self._raw_clip_txt_features = self._load_clip_txt_features() if self._use_clip else None
        return self._raw_clip_txt_features


    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def _load_clip_img_features(self):
        raise NotImplementedError

    def _load_clip_txt_features(self):
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        if self._use_clip:
            if idx % self._raw_shape[0] > self._ratio*self._raw_shape[0]:
                noise = np.random.normal(0., 1., (512))
                img_fts = self.get_img_features(idx)
                img_fts = img_fts/np.linalg.norm(img_fts)
                
                if self.remove_mean:
                    img_fts = img_fts - self.image_embed_mean
                    
                    if self.norm_after_remove_mean:
                        img_fts = img_fts/np.linalg.norm(img_fts)
                
                revised_img_fts = img_fts.copy()
                
                if self.add_lafite_noise:
                    revised_img_fts = (1 - self.noise_level) *revised_img_fts \
                    + self.noise_level*noise/np.linalg.norm(noise)
                elif self.add_noise:
                    revised_img_fts = revised_img_fts + self.noise_level*noise
                
                revised_img_fts = revised_img_fts/np.linalg.norm(revised_img_fts)
                
                return image.copy(), self.get_label(idx), img_fts, revised_img_fts
            else:
                img_fts = self.get_img_features(idx)
                img_fts = img_fts/np.linalg.norm(img_fts)
                txt_fts = self.get_txt_features(idx)
                txt_fts = txt_fts/np.linalg.norm(txt_fts)
                
                if self.remove_mean:
                    img_fts = img_fts - self.image_embed_mean
                    img_fts = img_fts/np.linalg.norm(img_fts)
                    
                    txt_fts = txt_fts - self.text_embed_mean
                    txt_fts = txt_fts/np.linalg.norm(txt_fts)
                
                return image.copy(), self.get_label(idx), img_fts, txt_fts
#             return image.copy(), self.get_label(idx), self.get_img_features(idx), self.get_txt_features(idx)
        else:
            return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_img_features(self, idx):
        img_features = self._get_clip_img_features()[self._raw_idx[idx]]
        return img_features.copy()

    def get_txt_features(self, idx):
        try:
            txt_features = self._get_clip_txt_features()[self._raw_idx[idx]]
            index = np.random.randint(0, len(txt_features), ())
            txt_features = txt_features[index] # randomly select one from the features
            txt_features = np.array(txt_features)
            txt_features = txt_features.astype(np.float32)
            return txt_features.copy()
        except:
            return np.random.normal(0., 1., (512))
    
    
    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
            self.json_name = 'dataset.json'

        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
            self.json_name = 'dataset.json'
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_clip_img_features(self):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            clip_features = json.load(f)['clip_img_features']
        if clip_features is None:
            return None
        clip_features = dict(clip_features)
        clip_features = [clip_features[fname.replace('\\', '/')] for fname in self._image_fnames]
        clip_features = np.array(clip_features)
        clip_features = clip_features.astype(np.float32)
        return clip_features

    def _load_clip_txt_features(self):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            clip_features = json.load(f)['clip_txt_features']
        if clip_features is None:
            return None
        clip_features = dict(clip_features)
        clip_features = [clip_features[fname.replace('\\', '/')] for fname in self._image_fnames]
        return clip_features
#----------------------------------------------------------------------------
