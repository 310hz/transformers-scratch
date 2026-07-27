from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
from importlib import import_module
from collections import OrderedDict
import contextlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.accelerator import current_accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from muon import MuonWithAuxAdam
from safetensors.torch import save_file
import wandb
from environs import env
from tqdm import tqdm


ROOT = Path(__file__).parent.parent
DPATH_CHECKPOINTS = ROOT / "checkpoints"
MODULE_MODELS = "models"
FNAME_STATE = "state.pth"
FNAME_MODEL = "model.safetensors"
CPU = torch.device("cpu")

env.read_env()


class Pipeline(ABC):
    def __init__(self, config, state_dict_model=None):
        self.config = config
        self.device = current_accelerator(check_available=True) or CPU
        self.model = self.get_model()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if state_dict_model is not None:
            self.model.load_state_dict(state_dict_model)
        if not self._is_dist(): # Prevent the same model from being placed on multiple devices
            self.model.to(self.device)

    def get_model(self):
        cls = getattr(import_module(f"{MODULE_MODELS}"), self.config.model.name)
        arch = self.config.model.arch or {}
        model = cls(**arch)
        return model

    def setup_train(self, dpath_ckpt=None, test_stream=False):
        self.start_time = datetime.now(tz=ZoneInfo("Asia/Tokyo"))
        config_train = self.config.train # alias

        self.total_steps = config_train.total_steps
        self.grad_accum_steps = config_train.grad_accum_steps
        scheduler_steps = config_train.total_steps // config_train.grad_accum_steps
        warmup_steps = int(config_train.warmup_ratio * scheduler_steps)
        self.max_grad_norm = config_train.max_grad_norm

        self.optimizer = self._get_optimizer()
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_steps,
        )
        self.scaler = torch.amp.GradScaler()

        self.now_steps = 0
        self._setup_checkpoint(dpath_ckpt)
        self.is_dist = self._is_dist()
        self._setup_device()
        self.is_master = self.global_rank == 0
        self.context_autocast = torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

        if self.device.type == "cuda":
            self.model = torch.compile(self.model)

        self.train_loader, self.test_loader = self._get_dataloader(test_stream)
        self.log_interval = config_train.log_interval
        self.eval_interval = config_train.eval_interval
        self.save_interval = config_train.save_interval

        self.metrics = self.get_metrics()

        if self.is_master:
            print(f"Model: {self.config.model.name}")
            print(f"Number of parameters: {self.n_params:,}")
            print(f"Number of devices: {self.world_size}")
            if self.resume:
                print(f"Resumed from checkpoint at {self.dpath_ckpt}")
            else:
                print(f"Checkpoints will be saved to {self.dpath_ckpt}")

            if config_train.wandb_run:
                name = config_train.wandb_run
            else:
                name = (
                    f"[{self.config.model.name} {self.n_params // 1_000_000}M] "
                    f"{self.start_time.strftime('%Y-%m-%d %H:%M')}"
                )

            wandb.login(key=env.str("WANDB_API_KEY"))
            self.wandb_run = wandb.init(
                project=env.str("WANDB_PROJECT_NAME"),
                group=self.config.task.name,
                name=name,
                config=self.config,
            )

    def _setup_checkpoint(self, dpath_ckpt):
        if dpath_ckpt is None:
            datestr = self.start_time.strftime("%Y%m%d_%H%M%S")
            dpath_ckpt = DPATH_CHECKPOINTS / datestr
            dpath_ckpt.mkdir(parents=True, exist_ok=True)
            self.resume = False
        else:
            dpath_ckpt = Path(dpath_ckpt)
            assert dpath_ckpt.exists(),\
                f"The checkpoint directory {dpath_ckpt} does not exist."

            latest_ckpt = dpath_ckpt / FNAME_STATE
            state_dict = torch.load(latest_ckpt, map_location=CPU)
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])
            self.scaler.load_state_dict(state_dict["scaler"])
            self.now_steps = state_dict["now_steps"]
            self.resume = True
        self.dpath_ckpt = dpath_ckpt

    @staticmethod
    def _is_dist() -> bool:
        return (
            dist.is_available()
            and torch.cuda.is_available()
            and env.int("WORLD_SIZE", 1) > 1
        )

    def _setup_device(self):
        if self.is_dist:
            rank = env.int("LOCAL_RANK")
            self.global_rank = env.int("RANK")
            torch.accelerator.set_device_index(rank)
            acc = torch.accelerator.current_accelerator()
            backend = torch.distributed.get_default_backend_for_device(acc)
            dist.init_process_group(backend)
            self.world_size = dist.get_world_size()
            self.device = torch.device(rank)
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.world_size = 1
            self.global_rank = 0
            self.model = self.model.to(self.device)

    def _get_optimizer(self):
        params_adam = []
        params_muon = []

        for name, parameter in self.model.named_parameters():
            if (parameter.ndim >= 2) and "transformer_layers." in name:
                params_muon.append(parameter)
            else:
                params_adam.append(parameter)

        if params_muon and self._is_dist(): # MuonWithAuxAdam only supports distributed training
            optimizer = MuonWithAuxAdam([
                dict(params=params_muon, use_muon=True, **self.config.train.muon),
                dict(params=params_adam, use_muon=False, **self.config.train.adam),
            ])
        else:
            optimizer = torch.optim.AdamW(
                params_adam + params_muon,
                **self.config.train.adam,
            )

        return optimizer

    def _get_dataloader(self, test_stream):
        ds_train, ds_test, get_ds_func = self.get_dataset(test_stream)
        ds_train = ds_train.shuffle(buffer_size=10000)

        if (self.world_size >= 2) and (self.global_rank is not None):
            ds_train = ds_train.shard(
                num_shards=self.world_size,
                index=self.global_rank,
            )
            ds_test = ds_test.shard(
                num_shards=self.world_size,
                index=self.global_rank,
            )

        train_loader = DataLoader(
            get_ds_func(ds_train),
            batch_size=self.config.train.batch_size,
            pin_memory=True,
        )
        test_loader = DataLoader(
            get_ds_func(ds_test),
            batch_size=self.config.train.batch_size,
            pin_memory=True,
        )
        return train_loader, test_loader

    def train(
            self,
            dpath_ckpt=None,
            test_stream=False,
            accelerator="auto",
            devices="auto",
            strategy="auto",
        ):
        self.setup_train(dpath_ckpt, test_stream)

        if self.is_master:
            print("Training started.", flush=True)
        self.model.train()

        is_running = True
        pbar = tqdm(total=self.total_steps, disable=not self.is_master)
        pbar.n = self.now_steps
        pbar.refresh()

        while is_running:
            for batch in self.train_loader:
                pbar.update()
                self.now_steps += 1
                now = self._check_now()

                if (not now.is_updating_step) and self.is_dist:
                    context_nosync = self.model.no_sync()
                else:
                    context_nosync = contextlib.nullcontext()

                with context_nosync, self.context_autocast:
                    predicts, targets = self.forward(batch)
                    self.metrics.update(predicts, targets)
                    loss = self.loss_fn(predicts, targets)
                loss_scaled = self.scaler.scale(loss / self.grad_accum_steps)
                loss_scaled.backward()

                if now.is_updating_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                if now.is_logging_step:
                    self.logging({ "loss": loss.item() }, prefix="train/")
                    self.logging(self.metrics.compute(), prefix="train/")

                if now.is_evaluating_step:
                    self.model.eval()
                    self.logging(self.evaluate(), prefix="test/")
                    self.model.train()

                if now.is_saving_step:
                    if self.is_master:
                        self._save_checkpoint(snapshot=True)

                if now.is_last:
                    is_running = False
                    break

        if self.is_master:
            print("Training finished.", flush=True)
            self.wandb_run.finish()

    def _check_now(self):
        is_last = self.now_steps >= self.total_steps
        if is_last:
            is_updating_step = True
            is_logging_step = True
            is_evaluating_step = True
            is_saving_step = self.save_interval is not None
        else:
            is_updating_step = self.now_steps % self.grad_accum_steps == 0
            is_logging_step = self.now_steps % self.log_interval == 0
            is_evaluating_step = self.now_steps % self.eval_interval == 0
            is_saving_step = (
                self.save_interval
                and self.now_steps % self.save_interval == 0
            )
        now = SimpleNamespace(
            is_last=is_last,
            is_updating_step=is_updating_step,
            is_logging_step=is_logging_step,
            is_evaluating_step=is_evaluating_step,
            is_saving_step=is_saving_step,
        )
        return now


    def _save_checkpoint(self, snapshot=False):
        model_state_dict = self.model.state_dict()
        correct_model_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            key = key.replace("_orig_mod.", "")
                # If the model is wrapped with `torch.compile`,
                # "_orig_mod." is appended to the key
            key = key.replace("module.", "")
                # If the model is wrapped with `DDP`,
                # "module." is appended to the key
            correct_model_state_dict[key] = value

        save_file(correct_model_state_dict, self.dpath_ckpt / FNAME_MODEL)

        state_dict = {
            "model": correct_model_state_dict,
            "now_steps": self.now_steps,
            "config": self.config.asdict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(state_dict, self.dpath_ckpt / FNAME_STATE)

        if snapshot:
            fname_snapshot = f"{self.now_steps:0{len(str(self.total_steps))}d}.pth"
            fpath_snapshot = self.dpath_ckpt / fname_snapshot
            torch.save(state_dict, fpath_snapshot)

    def logging(self, data: dict, prefix: str = ""):
        data = {f"{prefix}{k}": v for k, v in data.items()}
        if self.is_master:
            self.wandb_run.log(data, step=self.now_steps)
            self._save_checkpoint()

    @torch.no_grad()
    def evaluate(self):
        metrics = self.get_metrics()
        for batch in self.test_loader:
            predicts, targets = self.forward(batch)
            metrics.update(predicts, targets)
        result = metrics.compute()
        return result

    @abstractmethod
    def get_dataset(self, test_stream=False):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def loss_fn(self, predicts, targets):
        pass
