# Krea 2 LoRA training. Official path: train on RAW, infer on Turbo.

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator, PartialState
from PIL import Image

import train_network
from library import (
    flux_train_utils,
    krea2_sampling,
    krea2_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_base,
    strategy_krea2,
    train_util,
)
from library.device_utils import clean_memory_on_device, init_ipex
from library.utils import setup_logging

setup_logging()
import logging

init_ipex()

logger = logging.getLogger(__name__)


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit,
    vae,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()
    dit = accelerator.unwrap_model(dit)
    dit.switch_block_swap_for_inference()
    if text_encoders is not None:
        text_encoders = [(accelerator.unwrap_model(te) if te is not None else None) for te in text_encoders]

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    args,
                    dit,
                    text_encoders,
                    vae,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
    else:
        per_process_prompts = [prompts[i :: distributed_state.num_processes] for i in range(distributed_state.num_processes)]
        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator,
                        args,
                        dit,
                        text_encoders,
                        vae,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    dit.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    dit,
    text_encoders,
    vae,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt") or ""
    sample_steps = int(prompt_dict.get("sample_steps") or getattr(args, "sample_steps", 28) or 28)
    width = int(prompt_dict.get("width") or getattr(args, "sample_width", 1024) or 1024)
    height = int(prompt_dict.get("height") or getattr(args, "sample_height", 1024) or 1024)
    cfg_scale = float(prompt_dict.get("scale") or getattr(args, "sample_guidance", 5.5) or 5.5)
    seed = prompt_dict.get("seed")
    prompt: str = prompt_dict.get("prompt", "")
    use_turbo = bool(getattr(args, "turbo_dit", False))
    sample_mu = getattr(args, "sample_mu", None)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
    else:
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    logger.info(f"prompt: {prompt}")
    logger.info(f"height: {height} width: {width} steps: {sample_steps} CFG: {cfg_scale}")

    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoders is None:
            raise ValueError("Krea2 preview needs a text encoder or cached prompt outputs")
        tokens_and_masks = tokenize_strategy.tokenize(prpt)
        return encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens_and_masks)

    hiddens, mask = encode_prompt(prompt)
    untxt = untxtmask = None
    if cfg_scale > 1.0:
        untxt, untxtmask = encode_prompt(negative_prompt)

    seq_len = krea2_sampling.pixel_seq_len(height, width)
    if sample_mu is not None and sample_mu != "":
        mu = float(sample_mu)
    elif use_turbo:
        mu = 1.15
        if sample_steps == 28:
            sample_steps = 8
        if cfg_scale == 5.5:
            cfg_scale = 1.0
            untxt = untxtmask = None
    else:
        mu = krea2_sampling.krea2_shift_mu(seq_len)

    dit_is_training = dit.training
    dit.eval()
    images = krea2_sampling.sample_euler(
        dit,
        vae,
        hiddens.to(accelerator.device),
        mask.to(accelerator.device),
        untxt=None if untxt is None else untxt.to(accelerator.device),
        untxtmask=None if untxtmask is None else untxtmask.to(accelerator.device),
        device=accelerator.device,
        dtype=getattr(dit, "dtype", torch.bfloat16),
        width=width,
        height=height,
        steps=sample_steps,
        cfg_scale=cfg_scale,
        seed=0 if seed is None else int(seed),
        mu=mu,
    )
    if dit_is_training:
        dit.train()

    image = images[0]
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict.get("enum", 0)
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)


class Krea2NetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks: bool = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        te_path = getattr(args, "text_encoder", None) or getattr(args, "text_encoder1", None)
        if not te_path:
            raise ValueError("Krea2 requires --text_encoder (Qwen3-VL-4B-Instruct)")
        if not args.vae:
            raise ValueError("Krea2 requires --vae (Qwen-Image VAE)")
        if getattr(args, "fp8_scaled", False) and not getattr(args, "fp8_base", False):
            raise ValueError("Krea2 --fp8_scaled requires --fp8_base")
        if getattr(args, "turbo_dit", False) and getattr(args, "blocks_to_swap", None):
            raise ValueError("Krea2 --turbo_dit is incompatible with --blocks_to_swap")

        sampling = str(getattr(args, "timestep_sampling", "") or "")
        shift = getattr(args, "discrete_flow_shift", None)
        if sampling == "krea2_shift" and shift not in (None, "", 2.5, "2.5"):
            raise ValueError(
                "Krea2 krea2_shift is resolution-aware; leave discrete_flow_shift empty or 2.5. "
                "Do not pair it with a custom fixed shift."
            )

        if args.blocks_to_swap is not None:
            if args.blocks_to_swap > 26:
                logger.warning("Krea2 blocks_to_swap is capped at 26 (28-block DiT).")
                args.blocks_to_swap = 26
            self.is_swapping_blocks = args.blocks_to_swap > 0

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled"
            )
            args.cache_text_encoder_outputs = True

        if args.max_token_length is not None:
            logger.warning("max_token_length is not used in Krea2 training")

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        attn_mode = "torch"
        if getattr(args, "xformers", False):
            attn_mode = "xformers"
        if getattr(args, "attn_mode", None):
            attn_mode = args.attn_mode
        if attn_mode == "sdpa":
            attn_mode = "torch"

        loading_dtype = None if (args.fp8_base or getattr(args, "fp8_scaled", False)) else weight_dtype
        model = krea2_utils.load_krea2_dit(
            args.pretrained_model_name_or_path,
            dtype=torch.bfloat16 if loading_dtype is None else loading_dtype,
            device="cpu",
            fp8_scaled=bool(getattr(args, "fp8_scaled", False)),
            attn_mode=attn_mode,
        )
        if args.fp8_base and not getattr(args, "fp8_scaled", False):
            model.to(weight_dtype)

        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device, supports_backward=True)

        ae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=args.disable_mmap_load_safetensors,
        )
        ae.to(torch.float32 if args.no_half_vae else weight_dtype)
        ae.eval()

        te_path = args.text_encoder or getattr(args, "text_encoder1", None)
        text_encoder = krea2_utils.load_krea2_text_encoder(
            te_path,
            dtype=weight_dtype,
            device="cpu",
            tokenizer_cache_dir=getattr(args, "tokenizer_cache_dir", None),
        )
        return krea2_utils.MODEL_VERSION_KREA2, [text_encoder], ae, model

    def get_tokenize_strategy(self, args):
        return strategy_krea2.Krea2TokenizeStrategy(tokenizer_cache_dir=args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_krea2.Krea2TokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_krea2.Krea2LatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_krea2.Krea2TextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_krea2.Krea2TextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device if vae is not None else None
                org_unet_device = unet.device
                if vae is not None:
                    vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")
                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p is None or p in sample_prompts_te_outputs:
                                continue
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, text_encoders, tokens_and_masks
                            )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            if not self.is_train_text_encoder(args):
                logger.info("move text encoder back to cpu")
                text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet to original device")
                if vae is not None:
                    vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        if latents.dim() == 5:
            latents = latents.squeeze(2)
        noise = torch.randn_like(latents)
        packed_h = latents.shape[2] // 2
        packed_w = latents.shape[3] // 2

        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        hiddens, mask = text_encoder_conds
        if hiddens.dtype != weight_dtype:
            hiddens = hiddens.to(weight_dtype)
        mask = mask.to(dtype=torch.bool, device=accelerator.device)

        packed, pos, attn_mask = krea2_sampling.prepare(
            noisy_model_input, hiddens.shape[1], unet.config.patch, mask
        )
        t = sigmas.reshape(sigmas.shape[0])

        if args.gradient_checkpointing:
            packed.requires_grad_(True)
            if hiddens.dtype.is_floating_point:
                hiddens.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(img=packed, context=hiddens, t=t, pos=pos, mask=attn_mask)

        model_pred = krea2_sampling.unpack(model_pred, packed_h, packed_w, patch=unet.config.patch)
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(
            args, model_pred, noisy_model_input, sigmas
        )
        target = noise - latents
        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False)

    def update_metadata(self, metadata, args):
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_weighting_scheme"] = args.weighting_scheme

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        pass

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def prepare_unet_with_accelerator(self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module):
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)
        model = accelerator.prepare(unet, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()
        return model

    def sample_images(self, accelerator, args, epoch, global_step, device, ae, tokenizer, text_encoder, unet):
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoder)
        sample_images(accelerator, args, epoch, global_step, unet, ae, text_encoders, self.sample_prompts_te_outputs)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    parser.add_argument("--text_encoder", type=str, default=None, help="path to Qwen3-VL-4B-Instruct")
    parser.add_argument("--turbo_dit", action="store_true", help="preview as Turbo (mu=1.15, ~8 steps, CFG 1)")
    parser.add_argument("--sample_mu", type=float, default=None, help="override preview mu; omit for auto")
    parser.add_argument("--sample_guidance", type=float, default=5.5, help="preview CFG (RAW 5.5 / Turbo 1)")
    parser.add_argument("--sample_steps", type=int, default=28, help="preview steps (RAW 28 / Turbo 8)")
    parser.add_argument("--sample_height", type=int, default=1024)
    parser.add_argument("--sample_width", type=int, default=1024)
    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "krea2_shift"],
        default="shift",
        help="Krea2 timestep sampler. shift + discrete_flow_shift=2.5 is the fixed-res default; "
        "krea2_shift is resolution-aware (256px mu=0.5 … 1280px mu=1.15).",
    )
    parser.add_argument("--sigmoid_scale", type=float, default=1.0)
    parser.add_argument(
        "--model_prediction_type",
        choices=["raw", "additive", "sigma_scaled"],
        default="raw",
        help="How to interpret the model prediction. Krea2 default is raw.",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=2.5,
        help="Fixed Euler flow shift when timestep_sampling=shift. Default 2.5.",
    )
    parser.set_defaults(weighting_scheme="none")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT; requires --fp8_base")
    parser.add_argument(
        "--attn_mode",
        choices=["torch", "xformers", "flash", "sageattn", "sdpa"],
        default=None,
        help="Attention implementation. sdpa is treated as torch.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    trainer = Krea2NetworkTrainer()
    trainer.train(args)
