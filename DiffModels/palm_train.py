"""
Train a super-resolution model.
"""

import argparse
import inspect

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data, load_palm_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    # sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

import sys
sys.path.append(r"C:\Users\mobil\Desktop\25summer\GenPalm\Diff-Palm\cup_deployment")
from cup_dataset import load_palm_cup_data


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, palm_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # data = load_palm_data(
    #     args.data_dir,
    #     args.batch_size,
    #     large_size=args.large_size,
    #     small_size=args.small_size,
    #     class_cond=args.class_cond,
    # )
    data = load_palm_cup(
        args.raw_dir,
        args.label_dir,
        args.data_type,
        args.batch_size,
        args.large_size,
        args.include_key,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_palm(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_palm_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs

def load_palm_cup(raw_dir, label_dir, data_type, batch_size, large_size, include_key):
    data = load_palm_cup_data(
        raw_dir,
        label_dir,
        data_type,
        batch_size,
        image_size=large_size,
        remove_json=r"C:\Users\mobil\Desktop\25summer\GenPalm\Diff-Palm\cup_deployment\remove.json",
        deterministic=False,
        random_crop=False,
        random_flip=True,
        include_key=include_key,
        save_debug_dir=r"C:\Users\mobil\Desktop\25summer\GenPalm\Diff-Palm\cup_deployment",
    )

    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=128,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def palm_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 128
    res["small_size"] = 128
    res["in_channels"] = 6
    res["out_channels"] = 3
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def create_argparser():
    defaults = dict(
        # data_dir="",
        raw_dir="",
        label_dir="",
        data_type="",
        include_key="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(palm_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
