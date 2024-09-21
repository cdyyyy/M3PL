import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from collections import OrderedDict
import math
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from timm.optim import create_optimizer_v2
from utils import cosine_scheduler, build_multi_evaluator
from utils import MultiTrainerX

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": "M3PL",
                      "vision_depth": cfg.TRAINER.M3PL.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.M3PL.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.M3PL.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.M3PL.N_CTX_TEXT,
                      "n_prompts": cfg.TRAINER.M3PL.N_PROMPTS,
                      "init_std": cfg.TRAINER.M3PL.INIT_STD,}
    assert cfg.TRAINER.M3PL.PROMPT_DEPTH_VISION >= 1, "For Vision Prompting, PROMPT_DEPTH_VISION should be >= 1"
    assert cfg.TRAINER.M3PL.PROMPT_DEPTH_TEXT >= 1, "For Language Prompting, PROMPT_DEPTH_TEXT should be >= 1"
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx_text=4):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_ctx_text = n_ctx_text
        self.n_prompts = clip_model.n_prompts

    def forward(self, prompts, tokenized_prompts, is_reg=False, match_ids=None, current_id=0):
        if not self.training:
            prompts = prompts.reshape(-1, prompts.shape[-2], prompts.shape[-1])
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        inputs = [x, match_ids, is_reg, current_id]
        outputs = self.transformer(inputs)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        if self.training:
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        else:
            x = x.reshape(-1, self.n_prompts, x.shape[-2], x.shape[-1])
            x = x[torch.arange(x.shape[0]), :, tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.M3PL.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.M3PL.N_CTX_TEXT
        ctx_init = cfg.TRAINER.M3PL.CTX_INIT
        self.n_prompts = cfg.TRAINER.M3PL.N_PROMPTS
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        init_std = cfg.TRAINER.M3PL.INIT_STD
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialize n_prompts prompts
        ctx_vectors = torch.empty((self.n_prompts, n_ctx, ctx_dim), dtype=dtype)
        nn.init.normal_(ctx_vectors, std=init_std)
        prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f'Number of prompts: {self.n_prompts}')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.M3PL.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_test_prompts(self, ctx, prefix, suffix, labels=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (n_prompts, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if labels is not None:
            prefix = prefix[labels]
            suffix = suffix[labels]

        # each sample generate n_prompts prompts, i.e. (batch_size, n_prompts, n_ctx, ctx_dim)
        # prefix (n_cls, 1, ctx_dim)
        # suffix (n_cls, *, ctx_dim)
        # ctx (n_prompts, n_ctx, ctx_dim)
        prompts = torch.cat(
            [
                prefix.unsqueeze(1).expand(-1, self.n_prompts, -1, -1),  # (n_cls, n_prompts, 1, dim)
                ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1, -1),  # (n_cls, n_prompts, n_ctx, dim)
                suffix.unsqueeze(1).expand(-1, self.n_prompts, -1, -1),  # (n_cls, n_prompts, *, dim)
            ],
            dim=2,
        )

        return prompts
    
    def construct_train_prompts(self, ctx, prefix, suffix, current_id=0):
        # each sample generate n_prompts prompts, i.e. (batch_size, n_ctx, ctx_dim)
        # prefix (n_cls, 1, ctx_dim)
        # suffix (n_cls, *, ctx_dim)
        # ctx (n_prompts, n_ctx, ctx_dim)
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx[current_id].unsqueeze(0).expand(prefix.shape[0], -1, -1),  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        assert prompts.shape[0] == self.n_cls
        # prompts (n_cls, 1 + n_ctx + *, ctx_dim)
        return prompts
    
    def construct_reg_prompts(self, ctx, prefix, suffix=None, labels=None, match_ids=None):
        # full or mix caption
        prefix = prefix[labels]
        suffix = suffix[labels]

        prompts = torch.cat(
            [
                prefix,  # (batch_size, 1, dim)
                ctx[match_ids],  # (batch_size, n_ctx, dim)
                suffix,  # (batch_size, *, dim)
            ],
            dim=1,
        )
        
        return prompts

    def forward(self, labels=None, match_ids=None, current_id=0, is_reg=False):
        ctx = self.ctx   #(n_prompts, n_ctx, ctx_dim)   

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.training:
            if is_reg:
                prompts = self.construct_reg_prompts(ctx, prefix, suffix=suffix, labels=labels, match_ids=match_ids)
            else:
                prompts = self.construct_train_prompts(ctx, prefix, suffix, current_id=current_id)
        else:
            prompts = self.construct_test_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, n_ctx_text=cfg.TRAINER.M3PL.N_CTX_TEXT)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_prompts = cfg.TRAINER.M3PL.N_PROMPTS

    def forward(self, image, label=None, match_ids=None, current_id=0, is_reg=False):
        tokenized_prompts = self.tokenized_prompts.cuda() # (n_cls, n_tkn)
        logit_scale = self.logit_scale.exp()

        if self.training:
            if is_reg:
                tokenized_prompts = tokenized_prompts[label]
                prompts = self.prompt_learner(labels=label, match_ids=match_ids, is_reg=True)
            else:
                prompts = self.prompt_learner(current_id=current_id)
        else:
            # prompts (n_cls, n_prompts, 1 + n_ctx + *, ctx_dim)
            prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts, current_id=current_id, match_ids=match_ids, is_reg=is_reg)
        image_features = self.image_encoder(image.type(self.dtype), current_id=current_id, match_ids=match_ids, is_reg=is_reg)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.training:
            if is_reg:
                # image_features (batch_size, transformer.width)
                # text_features (batch_size, transformer.width)
                flyp_labels = torch.arange(label.shape[0]).long().cuda()
                logits = logit_scale * image_features @ text_features.t()
                logits_text = logit_scale * text_features @ image_features.t()
                loss = (F.cross_entropy(logits, flyp_labels, reduction='sum') + F.cross_entropy(logits_text, flyp_labels, reduction='sum')) * self.n_prompts / 2
            else:
                logits = logit_scale * image_features @ text_features.t()
                loss = F.cross_entropy(logits, label, reduction='sum')
            
            return loss
        
        else:
            image_features = image_features.permute(1, 0, 2)
            text_features = text_features.permute(1, 2, 0)
            # logits (n_prompts, batch_size, n_cls)
            logits = torch.bmm(image_features, text_features)
            logits = logits.permute(1, 0, 2)

            return logits


@TRAINER_REGISTRY.register()
class M3PL(MultiTrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.M3PL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self, args=None, device=None):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_prompts = cfg.TRAINER.M3PL.N_PROMPTS
        self.batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.w_coeff = cfg.TRAINER.M3PL.LAMBDA

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.M3PL.PREC == "fp32" or cfg.TRAINER.M3PL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # clip_model.to(self.device)
        self.dtype = clip_model.dtype

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = create_optimizer_v2(
            model_or_params = self.model,
            opt = cfg.OPTIM.NAME,
            lr = cfg.OPTIM.LR,
            weight_decay = cfg.OPTIM.WEIGHT_DECAY,
            eps = cfg.OPTIM.EPS,
        )
        print(f"Using optimizer: {self.optim.__class__.__name__} learning rate: {cfg.OPTIM.LR} weight decay: {cfg.OPTIM.WEIGHT_DECAY} eps: {cfg.OPTIM.EPS}")
        n_batches = len(self.train_loader_x)
        self.sched = cosine_scheduler(self.optim, cfg.OPTIM.LR, cfg.SCHEDULER.WARMUP_STEPS, cfg.OPTIM.MAX_EPOCH * n_batches, cfg.SCHEDULER.MIN_LR)
        print(f"Using scheduler: {self.sched} warmup_steps: {cfg.SCHEDULER.WARMUP_STEPS} min_lr: {cfg.SCHEDULER.MIN_LR}")
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.M3PL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        self.device_count = torch.cuda.device_count()
        if self.device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={self.device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, match_ids = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.M3PL.PREC
        if prec == "amp":
            raise NotImplementedError("AMP precision is not supported yet")
        else:
            step = self.batch_idx + self.epoch * self.num_batches
            self.sched(step+1)
            optim.zero_grad()
            loss = []
            for i in range(self.n_prompts):
                multi_loss = model(image, label, match_ids=match_ids, current_id=i)
                multi_loss = multi_loss.sum() / self.batch_size
                loss.append(multi_loss.item())
                multi_loss.backward()
            loss_2 = .0
            contrast_loss = model(image, label, match_ids=match_ids, is_reg=True)
            contrast_loss = self.w_coeff * contrast_loss.sum() / self.batch_size
            loss_2 += contrast_loss.item()
            contrast_loss.backward()
            loss_summary = {"total_loss": (sum(loss) + loss_2),  "loss": sum(loss), "contrast_loss": loss_2}
            optim.step()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        match_ids = self.generate_match_ids(label, self.n_prompts).to(self.device)
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, match_ids
    
    def generate_match_ids(self, labels, n_prompts):
        # ensure that when class collision, the corresponding match_id is different
        label_count = torch.bincount(labels)
        match_ids = torch.zeros_like(labels)
        mask = label_count[labels] <= n_prompts
        unique_labels = torch.unique(labels[mask])
        for label in unique_labels:
            label_mask = (labels == label) & mask
            num_labels = label_mask.sum()
            if num_labels > 0:
                match_ids[label_mask] = torch.randperm(n_prompts)[:num_labels]
        mask = label_count[labels] > n_prompts
        unique_labels = torch.unique(labels[mask])
        for label in unique_labels:
            label_mask = (labels == label) & mask
            num_labels = label_mask.sum()
            if num_labels > 0:
                match_ids[label_mask] = torch.randint(0, n_prompts, size=(num_labels,))
        return match_ids

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        """override the original save_model, if use nn.DataParallel, save the model without module."""
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

