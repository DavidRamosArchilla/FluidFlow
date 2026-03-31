from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch

from torch.optim import  AdamW
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity, schedule

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecisionPolicy
from ema_pytorch import EMA

from contextlib import contextmanager, nullcontext
from torch.distributed._composable.fsdp import FSDPModule

from tqdm.auto import tqdm


def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None

def print_profiler_summary(prof):
    averages = prof.key_averages()
    cuda_attr = 'self_device_time_total'

    print("\n=== Top CUDA ops ===")
    print(averages.table(sort_by=cuda_attr, row_limit=20))

    print("\n=== Communication ops (NCCL) ===")
    comm_ops = [e for e in averages if "nccl" in e.key.lower() or "allreduce" in e.key.lower()]
    total_cuda_time = sum(getattr(e, cuda_attr) for e in averages)
    total_comm_time = sum(getattr(e, cuda_attr) for e in comm_ops)

    for e in sorted(comm_ops, key=lambda x: getattr(x, cuda_attr), reverse=True):
        print(f"  {e.key:50s}  cuda_time: {getattr(e, cuda_attr)/1e3:.2f} ms")

    if total_cuda_time > 0:
        print(f"\nTotal CUDA time : {total_cuda_time/1e3:.2f} ms")
        print(f"Total comm time : {total_comm_time/1e3:.2f} ms  ({100*total_comm_time/total_cuda_time:.1f}%)")
        print(f"Compute time    : {(total_cuda_time-total_comm_time)/1e3:.2f} ms  ({100*(1-total_comm_time/total_cuda_time):.1f}%)")

    prof.export_chrome_trace("trace_rank.json")  

@contextmanager
def unsharded(model):
    fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]
    for m in fsdp_modules:
        m.unshard(async_op=False)
    try:
        yield
    finally:
        for m in fsdp_modules:
            m.reshard()

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'bf16',
        split_batches = True,
        max_grad_norm = None,
        use_cpu=False,
        dataset_test=None,
        eta_min_scheduler=None,
        compile_model=False,
        use_fsdop=False
    ):
        super().__init__()

        # accelerator
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=True,  # HYBRID_SHARD equivalent in FSDP2
            mixed_precision_policy=MixedPrecisionPolicy(
                # param_dtype=torch.bfloat16,   # all-gather in bf16
                reduce_dtype=torch.float32,   # reduce-scatter in fp32
            ),
        )
        self.accelerator = Accelerator(
            mixed_precision = mixed_precision_type if amp else 'no',
            cpu=use_cpu,
            dataloader_config=DataLoaderConfiguration(split_batches=split_batches),
            gradient_accumulation_steps=gradient_accumulate_every,
            fsdp_plugin=fsdp_plugin if use_fsdop else None
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
        if dataset_test is not None:
            self.dl_test = DataLoader(dataset_test, batch_size = train_batch_size, shuffle=False, pin_memory=True, num_workers=4)
        else:
            self.dl_test = None

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        # if use_muon:
        #     neural_net = diffusion_model.model
        #     muon_paramaters = list(neural_net.blocks.parameters()) + list(neural_net.t_embedder.parameters()) + list(neural_net.final_layer.parameters())
        #     adam_parameters = list(neural_net.x_embedder.parameters()) + list(neural_net.y_embedder.parameters())
        #     self.opts = [torch.optim.Muon(muon_paramaters, lr=train_lr, weight_decay=1e-3), AdamW(adam_parameters, lr=train_lr, betas=adam_betas, weight_decay=1e-4, fused=True)]
        # else:
        #     self.opts = [AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=1e-4, fused=True)]
        eps = 1e-6 if mixed_precision_type == 'fp16' or mixed_precision_type == 'bf16' else 1e-8
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=1e-2, fused=True, eps=eps)
        # cosine annealing lr scheduler
        self.use_lr_scheduler = eta_min_scheduler is not None
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.train_num_steps, eta_min=eta_min_scheduler)
            self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

        # if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device, dtype=torch.float32) # 

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.cond_dim = diffusion_model.cond_dim
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        if compile_model:
            print("Compiling model...")
            self.model = torch.compile(self.model) # mode="reduce-overhead"
            # self.model.neural_net = torch.compile(self.model.neural_net) # mode="reduce-overhead"
            print("Model compiled")

        self.loss_history = []
        self.test_loss_history = []

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone, model_state_dict):
        lr = self.opt.param_groups[0]['lr']

        data = {
            'step': self.step,
            'model': model_state_dict,  # Use the passed dictionary
            # 'opts': [opt.state_dict() for opt in self.opts],
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.use_lr_scheduler else None,
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'lr': lr,
            'loss_history': torch.tensor(self.loss_history),
            'test_loss_history': torch.tensor(self.test_loss_history)
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)
        try:
            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])
        except:
            state_dict = data['model']
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove "module." prefix if it exists (or contains it)
                new_key = key.replace("module.", "") if "module." in key else key
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)

        self.step = data['step']
        # self.opt.load_state_dict(data['opt'])
        # for i, opt in enumerate(self.opts):
        #     opt.load_state_dict(data["opts"][i])
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            
        if exists(data['loss_history']):
            self.loss_history = data['loss_history'].tolist()

        if "test_loss_history" in data:
            self.test_loss_history = data['test_loss_history'].tolist()
        
        if self.use_lr_scheduler and "scheduler" in data and self.use_lr_scheduler:
            self.scheduler.load_state_dict(data['scheduler'])
        
        if "lr" in data:
            print(f"Setting loaded learning rate to {data['lr']}")
            for param_group in self.opt.param_groups:
                param_group['lr'] = data['lr']

    def train(self, do_profiling=False):
        accelerator = self.accelerator
        device = accelerator.device
        profiler = None
        PROFILE_START_STEP = 25
        PROFILE_ACTIVE_STEPS = 15 
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                if do_profiling and self.step == PROFILE_START_STEP and profiler is None:
                    profiler = profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        schedule=schedule(wait=0, warmup=1, active=PROFILE_ACTIVE_STEPS, repeat=1),
                        record_shapes=True,
                        with_stack=True,
                        on_trace_ready=print_profiler_summary,
                    )
                    profiler.__enter__()
                    if accelerator.is_main_process:
                        print(f"[Profiler] Started at step {self.step}")

                self.model.train()
                total_loss = 0.
                # for _ in range(self.gradient_accumulate_every):
                data = next(self.dl)#.to(device)
                with accelerator.accumulate(self.model):
                    sequence, classes = data[0].to(device), data[1].to(device)
                    with self.accelerator.autocast():
                        loss = self.model(sequence, classes=classes)

                    self.accelerator.backward(loss)
                    # accelerator.wait_for_everyone()
                    if accelerator.sync_gradients:
                        if self.max_grad_norm is not None:
                            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        # Step increments only on actual updates (when gradients are accumulated and ready to do backward)
                        self.step += 1

                    self.opt.step()
                    self.opt.zero_grad()
                    if self.use_lr_scheduler:
                        self.scheduler.step()
                if profiler is not None:
                    profiler.step()
                    if self.step >= PROFILE_START_STEP + PROFILE_ACTIVE_STEPS + 1:  # +1 for warmup
                        profiler.__exit__(None, None, None)
                        profiler = None
                        if accelerator.is_main_process:
                            print("[Profiler] Done. Stopping training.")
                        break  # remove this if you want training to continue after profiling
                # accelerator.wait_for_everyone()
                if accelerator.sync_gradients:
                    with unsharded(self.model):
                        if self.accelerator.is_main_process:
                            self.ema.update()
                    if accelerator.is_main_process:
                        total_loss += loss.detach().float().mean().cpu().item()
                        self.loss_history.append(total_loss)
                        pbar.set_description(f'loss: {total_loss:.5f}')
                        pbar.update(1)
                        # with unsharded(self.model):
                        #     self.ema.update()
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        model_state_dict = accelerator.get_state_dict(self.model)
                        milestone = self.step // self.save_and_sample_every
                        if self.dl_test is not None:
                            samples, sequences = self.eval_model(self.dl_test.dataset, batch_size=self.batch_size)
                            if accelerator.is_main_process:
                                mse = ((samples - sequences) ** 2).mean()
                                test_losses = mse.cpu().item()
                                self.test_loss_history.append(test_losses)
                                torch.save(samples, str(self.results_folder / f'sample-{milestone}.pt'))

                        if accelerator.is_main_process:
                            self.save(milestone, model_state_dict)
                            self.save_loss_plot()
                            self.ema.ema_model.train()

                if accelerator.sync_gradients:
                    accelerator.wait_for_everyone()
 
        if accelerator.is_main_process:
            self.save_loss_plot()
            
        accelerator.print('training complete')
        # profiler.__exit__(None, None, None)
    
    def eval_model(self, dataset_test, batch_size=32, use_autocast=False, **sampling_kwargs):
        # Prepare models
        dl_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        
        # Accelerate handles moving the model to the correct device even on 1 GPU
        model, test_dataloader = self.accelerator.prepare(self.ema.ema_model, dl_test)
        model.eval()
       
        # FIX: Only broadcast if we are in a distributed setup (more than 1 process)
        if self.accelerator.num_processes > 1:
            with torch.no_grad():
                for param in model.parameters():
                    # Broadcast from rank 0 to all other ranks
                    torch.distributed.broadcast(param.data, src=0)
            
            # Wait for broadcast to complete
            self.accelerator.wait_for_everyone()
        
        all_preds = []
        all_seqs = []
        for data in tqdm(test_dataloader, disable=not self.accelerator.is_main_process):
            sequence, classes = data[0], data[1]
            with torch.inference_mode():    
                with (self.accelerator.autocast() if use_autocast else nullcontext()):
                    pred = model.sample(classes=classes, **sampling_kwargs)

                # gather_for_metrics works automatically for both single and multi-GPU
                gathered_pred, sequence = self.accelerator.gather_for_metrics((pred, sequence))
        
            if self.accelerator.is_main_process:
                all_preds.append(gathered_pred.cpu())
                all_seqs.append(sequence.cpu())
                
            del pred
            del gathered_pred
            del classes
            del sequence
            
        if self.accelerator.is_main_process:
            return torch.cat(all_preds, dim=0), torch.cat(all_seqs, dim=0)
        return None, None
    
    def save_loss_plot(self):
        plt.figure()
        plt.plot(self.loss_history, label='Loss')
        if self.test_loss_history:
            test_x_values = list(range(self.save_and_sample_every, self.step+1, self.save_and_sample_every))
            # print(test_x_values, self.test_loss_history)
            plt.plot(test_x_values, self.test_loss_history, label='Test Loss')    
        # Compute moving average
        window_size = 100
        if len(self.loss_history) >= window_size:
            moving_avg = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.loss_history)), moving_avg, label=f'Moving Avg ({window_size})')
        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss Evolution')
        plt.legend()
        plt.savefig(self.results_folder / "loss_evolution.png", bbox_inches="tight", pad_inches=0)
        plt.close()