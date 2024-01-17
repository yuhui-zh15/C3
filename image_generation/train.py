import datetime
import json
import math
import os
import re
import tempfile
from pathlib import Path

import click
import dnnlib
import torch
import wandb
from dateutil import tz
from metrics import metric_main
from torch_utils import custom_ops, training_stats
from training import training_loop

# ----------------------------------------------------------------------------


class UserError(Exception):
    pass


# ----------------------------------------------------------------------------


def get_experiment_name(args):
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    experiment_name = args.experiment_name_prefix

    if "network-snapshot" in args.resume_pkl:
        iteration_num = Path(args.resume_pkl).parts[-1]
        iteration_num = int(re.findall(r"\d+", iteration_num)[0])
        experiment_name += f"_resume_from_{iteration_num}"

    if args.remove_mean:
        experiment_name += f"_remove_mean"

    if (
        args.remove_mean
        and (args.add_noise or args.add_lafite_noise)
        and args.norm_after_remove_mean
    ):
        experiment_name += f"_norm_after_remove_mean"

    if args.add_noise:
        experiment_name += f"_add_noise"
    elif args.add_lafite_noise:
        experiment_name += "_add_lafite_noise"

    if args.add_noise or args.add_lafite_noise:
        experiment_name += f"_level_{round(args.noise_level, 3)}"

    experiment_name += f"/{timestamp}"

    return experiment_name


def build_wandb_logger(args, experiment_name):
    # wandb logging
    wandb.init(
        name=experiment_name,
        project=args.logger_project,
        entity=args.logger_entity,
        dir=args.logger_save_dir,
    )
    wandb.config.update(args)


def setup_training_loop_kwargs(
    f_dim=None,
    d_use_norm=None,  # normalize the feature extracted by discriminator or not
    d_use_fts=None,  # discriminator extract semantic feature or not
    mixing_prob=None,  # mixing probability of ground-truth and language-free generated pairs, mixing_prob=0 means only use ground-truth, mixing_prob=1. means using only pseudo pairs(language-free)
    lam=None,  # hyper-parameter for contrastive loss
    temp=None,  # hyper-parameter for contrastive loss
    change=None,  # hyper-parameter for architecture
    map_num=None,  # hyper-parameter for architecture
    gather=None,  # hyper-parameter for contrastive loss
    itd=None,  # hyper-parameter for contrastive loss
    itc=None,  # hyper-parameter for contrastive loss
    iid=None,  # hyper-parameter for contrastive loss
    iic=None,  # hyper-parameter for contrastive loss
    metric_only_test=None,  # hyper-parameter for computing metrics
    fmap=None,  # hyper-parameter for architecture, related to channel number
    ratio=None,
    # General options (not included in desc).
    gpus=None,  # Number of GPUs: <int>, default = 1 gpu
    snap=None,  # Snapshot interval: <int>, default = 50 ticks
    metrics=None,  # List of metric names: [], ['fid50k_full'] (default), ...
    seed=None,  # Random seed: <int>, default = 0
    # Dataset.
    data=None,  # Training dataset (required): <path>
    test_data=None,  # Testing dataset for metrics, if not use training dataset
    cond=None,  # Train conditional model based on dataset labels: <bool>, default = False
    subset=None,  # Train with only N images: <int>, default = all
    mirror=None,  # Augment dataset with x-flips: <bool>, default = False
    # Base config.
    cfg=None,  # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma=None,  # Override R1 gamma: <float>
    kimg=None,  # Override training duration: <int>
    batch=None,  # Override batch size: <int>
    # Discriminator augmentation.
    aug=None,  # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p=None,  # Specify p for 'fixed' (required): <float>
    target=None,  # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe=None,  # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'
    # Transfer learning.
    resume=None,  # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed=None,  # Freeze-D: <int>, default = 0 discriminator layers
    # Performance options (not included in desc).
    fp32=None,  # Disable mixed-precision training: <bool>, default = False
    nhwc=None,  # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32=None,  # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench=None,  # Disable cuDNN benchmarking: <bool>, default = False
    workers=None,  # Override number of DataLoader workers: <int>, default = 3
    remove_mean=False,
    add_noise=False,
    add_lafite_noise=False,
    noise_level=None,
    norm_after_remove_mean=False,
    logger_save_dir=None,
    logger_project=None,
    logger_entity=None,
    experiment_name_prefix=None,
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------
    if f_dim is None:
        f_dim = 512
    assert isinstance(f_dim, int)
    args.f_dim = f_dim

    if ratio is None:
        ratio = 1.0
    args.ratio = ratio

    if mixing_prob is None:
        mixing_prob = 0.0
    args.mixing_prob = mixing_prob

    if fmap is None:
        fmap = 1.0

    if metric_only_test is None:
        metric_only_test = False
    args.metric_only_test = metric_only_test

    if map_num is None:
        map_num = 8

    if lam is None:
        lam = 0.0
    args.lam = lam

    if temp is None:
        temp = 0.5
    args.temp = temp

    if itd is None:
        itd = 10.0
    args.itd = itd
    if itc is None:
        itc = 10.0
    args.itc = itc

    if iid is None:
        iid = 0.0
    args.iid = iid
    if iic is None:
        iic = 0.0
    args.iic = iic

    if change is None:
        change = 256

    if d_use_norm is None:
        d_use_norm = False
    assert isinstance(d_use_norm, bool)
    args.d_use_norm = d_use_norm

    if d_use_fts is None:
        d_use_fts = True
    args.d_use_fts = d_use_fts

    if gather is None:
        gather = False
    args.gather = gather

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError("--gpus must be a power of two")
    args.num_gpus = gpus

    if snap is None:
        snap = 150
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError("--snap must be at least 1")
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ["fid50k_full", "is50k"]
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError(
            "\n".join(
                ["--metrics can only contain the following values:"]
                + metric_main.list_valid_metrics()
            )
        )
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Additional: embed transform, logging
    # -----------------------------------
    # Embed transform
    if noise_level is None:
        noise_level = math.sqrt(0.016)
    args.noise_level = noise_level

    args.remove_mean = remove_mean
    args.add_noise = add_noise
    args.add_lafite_noise = add_lafite_noise
    args.norm_after_remove_mean = norm_after_remove_mean

    # Logging
    args.logger_save_dir = logger_save_dir
    args.logger_project = logger_project
    args.logger_entity = logger_entity

    args.experiment_name_prefix = experiment_name_prefix

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    print("using data: ", data, "testing data: ", test_data)
    if test_data is None:
        test_data = data
    args.training_set_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        path=data,
        use_labels=True,
        max_size=None,
        xflip=False,
        use_clip=True,
        ratio=args.ratio,
        remove_mean=remove_mean,
        add_noise=add_noise,
        noise_level=noise_level,
        add_lafite_noise=add_lafite_noise,
        norm_after_remove_mean=norm_after_remove_mean,
    )
    args.testing_set_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        path=test_data,
        use_labels=True,
        max_size=None,
        xflip=False,
        use_clip=True,
        ratio=1.0,
        remove_mean=remove_mean,
    )
    args.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=False, num_workers=0, prefetch_factor=2
    )
    try:
        training_set = dnnlib.util.construct_class_by_name(
            **args.training_set_kwargs
        )  # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = (
            training_set.resolution
        )  # be explicit about resolution
        args.training_set_kwargs.use_labels = (
            training_set.has_labels
        )  # be explicit about labels
        args.training_set_kwargs.max_size = len(
            training_set
        )  # be explicit about dataset size
        desc = training_set.name
        args.testing_set_kwargs.resolution = (
            training_set.resolution
        )  # be explicit about resolution
        args.testing_set_kwargs.use_labels = (
            training_set.has_labels
        )  # be explicit about labels
        del training_set  # conserve memory

    except IOError as err:
        raise UserError(f"--data: {err}")

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError("--cond=True requires labels specified in dataset.json")
        desc += "-cond"
    else:
        args.training_set_kwargs.use_labels = False
        args.testing_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(
                f"--subset must be between 1 and {args.training_set_kwargs.max_size}"
            )
        desc += f"-subset{subset}"
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += "-mirror"
        args.training_set_kwargs.xflip = True
        args.testing_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = "auto"
    assert isinstance(cfg, str)
    desc += f"-{cfg}-lam{lam:g}-temp{temp:g}-map_num{map_num:g}"

    cfg_specs = {
        "auto": dict(
            ref_gpus=-1,
            kimg=25000,
            mb=-1,
            mbstd=-1,
            fmaps=-1,
            lrate=-1,
            gamma=1.0,
            ema=-1,
            ramp=0.05,
            map=map_num,
        ),  # Populated dynamically based on resolution and GPU count.
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == "auto":
        desc += f"-gpus{gpus:d}"
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = (
            16 * gpus
        )  # max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(
            spec.mb // gpus, 4
        )  # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else fmap
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res**2) / spec.mb  # heuristic formula
        spec.ema = spec.mb * 10 / 32

    #     args.M_kwargs = dnnlib.EasyDict(class_name='training.networks.ManiNetwork', z_dim=args.f_dim,  layer_features=args.f_dim, w_dim=512, num_layers=8)
    args.G_kwargs = dnnlib.EasyDict(
        class_name="training.networks.Generator",
        z_dim=512,
        w_dim=512,
        m_layer_features=args.f_dim,
        m_num_layers=8,
        mapping_kwargs=dnnlib.EasyDict(),
        synthesis_kwargs=dnnlib.EasyDict(),
    )
    args.D_kwargs = dnnlib.EasyDict(
        class_name="training.networks.Discriminator",
        use_norm=args.d_use_norm,
        use_fts=args.d_use_fts,
        block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict(),
    )
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(
        spec.fmaps * 32768
    )
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = (
        args.D_kwargs.num_fp16_res
    ) = 4  # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = (
        args.D_kwargs.conv_clamp
    ) = 256  # clamp activations to avoid float16 overflow
    args.G_kwargs.synthesis_kwargs.change = change
    args.G_kwargs.synthesis_kwargs.f_dim = args.f_dim
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    args.D_kwargs.epilogue_kwargs.f_dim = args.f_dim

    args.G_opt_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", lr=spec.lrate, betas=[0, 0.99], eps=1e-8
    )
    args.D_opt_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", lr=spec.lrate, betas=[0, 0.99], eps=1e-8
    )
    args.loss_kwargs = dnnlib.EasyDict(
        class_name="training.loss.StyleGAN2Loss",
        r1_gamma=spec.gamma,
        remove_mean=remove_mean,
        add_noise=add_noise,
        noise_level=noise_level,
        add_lafite_noise=add_lafite_noise,
        norm_after_remove_mean=norm_after_remove_mean,
    )

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == "cifar":
        args.loss_kwargs.pl_weight = 0  # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0  # disable style mixing
        args.D_kwargs.architecture = "orig"  # disable residual skip connections

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError("--gamma must be non-negative")
        desc += f"-gamma{gamma:g}"
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError("--kimg must be at least 1")
        desc += f"-kimg{kimg:d}"
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError("--batch must be at least 1 and divisible by --gpus")
        desc += f"-batch{batch}"
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = "noaug"  # no augmentation is used in our experiments
    else:
        assert isinstance(aug, str)
        desc += f"-{aug}"

    if aug == "ada":
        args.ada_target = 0.6

    elif aug == "noaug":
        pass

    elif aug == "fixed":
        if p is None:
            raise UserError(f"--aug={aug} requires specifying --p")

    else:
        raise UserError(f"--aug={aug} not supported")

    if p is not None:
        assert isinstance(p, float)
        if aug != "fixed":
            raise UserError("--p can only be specified with --aug=fixed")
        if not 0 <= p <= 1:
            raise UserError("--p must be between 0 and 1")
        desc += f"-p{p:g}"
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != "ada":
            raise UserError("--target can only be specified with --aug=ada")
        if not 0 <= target <= 1:
            raise UserError("--target must be between 0 and 1")
        desc += f"-target{target:g}"
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = "bgc"
    else:
        if aug == "noaug":
            raise UserError("--augpipe cannot be specified with --aug=noaug")
        desc += f"-{augpipe}"

    augpipe_specs = {
        "blit": dict(xflip=1, rotate90=1, xint=1),
        "geom": dict(scale=1, rotate=1, aniso=1, xfrac=1),
        "color": dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        "filter": dict(imgfilter=1),
        "noise": dict(noise=1),
        "cutout": dict(cutout=1),
        "bg": dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        "bgc": dict(
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
        ),
        "bgcf": dict(
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
            imgfilter=1,
        ),
        "bgfn": dict(
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            imgfilter=1,
            noise=1,
        ),
        "bgcfn": dict(
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
            imgfilter=1,
            noise=1,
        ),
        "bgcfnc": dict(
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
            imgfilter=1,
            noise=1,
            cutout=1,
        ),
    }

    assert augpipe in augpipe_specs
    if aug != "noaug":
        args.augment_kwargs = dnnlib.EasyDict(
            class_name="training.augment.AugmentPipe", **augpipe_specs[augpipe]
        )

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        "ffhq256": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl",
        "ffhq512": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl",
        "ffhq1024": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl",
        "celebahq256": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl",
        "lsundog256": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl",
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = "noresume"
    elif resume == "noresume":
        desc += "-noresume"
    elif resume in resume_specs:
        desc += f"-resume{resume}"
        args.resume_pkl = resume_specs[resume]  # predefined url
    else:
        desc += "-resumecustom"
        args.resume_pkl = resume  # custom path or url

    if resume != "noresume":
        args.ada_kimg = 100  # make ADA react faster at the beginning
        args.ema_rampup = None  # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError("--freezed must be non-negative")
        desc += f"-freezed{freezed:d}"
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = (
            args.D_kwargs.block_kwargs.fp16_channels_last
        ) = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError("--workers must be at least 1")
        args.data_loader_kwargs.num_workers = workers

    return desc, args


# ----------------------------------------------------------------------------


def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(
        file_name=os.path.join(args.run_dir, "log.txt"),
        file_mode="a",
        should_flush=True,
    )

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus,
            )
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus,
            )

    # Init torch_utils.
    sync_device = torch.device("cuda", rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = "none"

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)


# ----------------------------------------------------------------------------


class CommaSeparatedList(click.ParamType):
    name = "list"

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == "none" or value == "":
            return []
        return value.split(",")


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option("--f_dim", help="dimension of features", type=int, metavar="INT")
@click.option("--change", help="change structure", type=int, metavar="INT")
@click.option(
    "--map_num", help="layer number of mapping network", type=int, metavar="INT"
)
@click.option(
    "--d_use_norm",
    help="Input features into every layer of discriminator",
    type=bool,
    metavar="BOOL",
)
@click.option(
    "--d_use_fts",
    help="Use text feature in discriminator or not",
    type=bool,
    metavar="BOOL",
)
@click.option(
    "--gather",
    help="gather all negative samples across gpus or not",
    type=bool,
    metavar="BOOL",
)
@click.option(
    "--mixing_prob", help="if mixing_prob==1 -> no text data used", type=float
)
@click.option(
    "--lam",
    help="hyper-parameter for contrastive loss (softmax along different dimensions)",
    type=float,
)
@click.option("--temp", help="temperature for contrastive loss", type=float)
@click.option("--itd", help="", type=float)
@click.option("--itc", help="", type=float)
@click.option("--iid", help="", type=float)
@click.option("--iic", help="", type=float)
@click.option(
    "--metric_only_test",
    help="compute metrics using test dataset vs test dataset?",
    type=bool,
    metavar="BOOL",
)
@click.option("--fmap", help="", type=float)
@click.option("--ratio", help="ratio of data with ground-truth text used", type=float)


# General options.
@click.option(
    "--outdir", help="Where to save the results", required=True, metavar="DIR"
)
@click.option(
    "--gpus", help="Number of GPUs to use [default: 1]", type=int, metavar="INT"
)
@click.option(
    "--snap", help="Snapshot interval [default: 50 ticks]", type=int, metavar="INT"
)
@click.option(
    "--metrics",
    help='Comma-separated list or "none" [default: fid50k_full]',
    type=CommaSeparatedList(),
)
@click.option("--seed", help="Random seed [default: 0]", type=int, metavar="INT")
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)

# Dataset.
@click.option(
    "--data", help="Training data (directory or zip)", metavar="PATH", required=True
)
@click.option(
    "--test_data", help="Testing data (directory or zip)", metavar="PATH", required=True
)
@click.option(
    "--cond",
    help="Train conditional model based on dataset labels [default: false]",
    type=bool,
    metavar="BOOL",
)
@click.option(
    "--subset", help="Train with only N images [default: all]", type=int, metavar="INT"
)
@click.option(
    "--mirror",
    help="Enable dataset x-flips [default: false]",
    type=bool,
    metavar="BOOL",
)

# Base config.
@click.option(
    "--cfg",
    help="Base config [default: auto]",
    type=click.Choice(
        ["auto", "stylegan2", "paper256", "paper512", "paper1024", "cifar"]
    ),
)
@click.option("--gamma", help="Override R1 gamma", type=float)
@click.option("--kimg", help="Override training duration", type=int, metavar="INT")
@click.option("--batch", help="Override batch size", type=int, metavar="INT")

# Discriminator augmentation.
@click.option(
    "--aug",
    help="Augmentation mode [default: ada]",
    type=click.Choice(["noaug", "ada", "fixed"]),
)
@click.option("--p", help="Augmentation probability for --aug=fixed", type=float)
@click.option("--target", help="ADA target value for --aug=ada", type=float)
@click.option(
    "--augpipe",
    help="Augmentation pipeline [default: bgc]",
    type=click.Choice(
        [
            "blit",
            "geom",
            "color",
            "filter",
            "noise",
            "cutout",
            "bg",
            "bgc",
            "bgcf",
            "bgcfn",
            "bgcfnc",
        ]
    ),
)

# Transfer learning.
@click.option("--resume", help="Resume training [default: noresume]", metavar="PKL")
@click.option("--freezed", help="Freeze-D [default: 0 layers]", type=int, metavar="INT")

# Performance options.
@click.option(
    "--fp32", help="Disable mixed-precision training", type=bool, metavar="BOOL"
)
@click.option(
    "--nhwc", help="Use NHWC memory format with FP16", type=bool, metavar="BOOL"
)
@click.option("--nobench", help="Disable cuDNN benchmarking", type=bool, metavar="BOOL")
@click.option(
    "--allow-tf32",
    help="Allow PyTorch to use TF32 internally",
    type=bool,
    metavar="BOOL",
)
@click.option(
    "--workers", help="Override number of DataLoader workers", type=int, metavar="INT"
)
@click.option("--remove_mean", is_flag=True)
@click.option("--add_noise", is_flag=True)
@click.option("--add_lafite_noise", is_flag=True)
@click.option("--noise_level", type=float, help="amount of noise to add", default=None)
@click.option("--norm_after_remove_mean", is_flag=True)
@click.option("--logger_save_dir", type=str)
@click.option("--logger_project", type=str)
@click.option("--logger_entity", type=str)
@click.option("--experiment_name_prefix", type=str)
def main(ctx, outdir, dry_run, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".

    Examples:

    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1

    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.

    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.

    experiment_name = get_experiment_name(args)

    args.pop("remove_mean")
    args.pop("add_noise")
    args.pop("add_lafite_noise")
    args.pop("noise_level")
    args.pop("norm_after_remove_mean")

    print("Creating output directory...")
    args.run_dir = os.path.join(outdir, experiment_name)
    assert not os.path.exists(args.run_dir)

    # prev_run_dirs = []
    # if os.path.isdir(outdir):
    #     prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    # prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    # prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    # cur_run_id = max(prev_run_ids, default=-1) + 1
    # args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    # assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print("Training options:")
    print(json.dumps(args, indent=2))
    print()
    print(f"Output directory:   {args.run_dir}")
    print(f"Training data:      {args.training_set_kwargs.path}")
    print(f"Training duration:  {args.total_kimg} kimg")
    print(f"Number of GPUs:     {args.num_gpus}")
    print(f"Number of images:   {args.training_set_kwargs.max_size}")
    print(f"Image resolution:   {args.training_set_kwargs.resolution}")
    print(f"Conditional model:  {args.training_set_kwargs.use_labels}")
    print(f"Dataset x-flips:    {args.training_set_kwargs.xflip}")
    print(f"Discriminator use normalization:  {args.d_use_norm}")
    print(f"Discriminator use fts: {args.d_use_fts}")

    # Dry run?
    if dry_run:
        print("Dry run; exiting.")
        return

    # Create output directory.
    print("Creating output directory...")
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, "training_options.json"), "wt") as f:
        json.dump(args, f, indent=2)

    build_wandb_logger(args, experiment_name)

    args.pop("logger_save_dir")
    args.pop("logger_project")
    args.pop("logger_entity")
    args.pop("experiment_name_prefix")

    # Launch processes.
    print("Launching processes...")
    torch.multiprocessing.set_start_method("spawn")
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(
                fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus
            )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
