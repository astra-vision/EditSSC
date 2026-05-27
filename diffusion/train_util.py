import copy
import functools
import os
import numpy as np
import blobfile as bf
import torch as th
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.nn import update_ema
from diffusion.resample import LossAwareSampler, UniformSampler
from utils.common_util import draw_scalar_field2D
from utils import dist_util
from diffusion.nn import mean_flat, decompose_featmaps




INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        triplane_loss_type,
        timestep_respacing,
        training_step,
        model,
        diffusion,
        data,
        val_data,
        ssc_refine,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_cosine_lr=False,
        use_constant_lr=False,
        warmup_steps=0,
        grad_clip=0.0,
        use_ema=True,
        args=None
    ):
        self.args=args
        
        self.triplane_loss_type = triplane_loss_type
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.ssc_refine = ssc_refine
        self.training_step = training_step
        self.timestep_respacing = timestep_respacing

        

        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.use_ema = use_ema
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.use_cosine_lr = use_cosine_lr
        self.use_constant_lr = use_constant_lr
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip

        tblog_dir = os.path.join(logger.get_current().get_dir(), "tblog")
        self.tb = SummaryWriter(tblog_dir)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        # Log trainable parameters (for two-stage training debugging)
        if hasattr(self.model, 'get_trainable_parameters'):
            trainable_params = self.model.get_trainable_parameters()
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_count = sum(p.numel() for p in trainable_params)
            logger.log(f"Two-stage training: {trainable_count}/{total_params} parameters are trainable")

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        # Optimizer automatically skips parameters with requires_grad=False
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        # Setup LR scheduler
        self.lr_scheduler = None
        if self.use_constant_lr:
            logger.log(f"Using constant learning rate: {self.lr} (no scheduling/annealing)")
        elif self.use_cosine_lr:
            num_training_steps = self.lr_anneal_steps if self.lr_anneal_steps > 0 else 100000
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.opt,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.log(f"Using cosine LR schedule with {self.warmup_steps} warmup steps and {num_training_steps} total steps")
        else:
            logger.log(f"Using linear LR annealing over {self.lr_anneal_steps} steps")


        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            if self.use_ema:
                self.ema_params = [
                    self._load_ema_parameters(rate) for rate in self.ema_rate
                ]
            else:
                self.ema_params = []
                logger.log("EMA is disabled - skipping EMA parameter initialization")
        else:
            if self.use_ema:
                self.ema_params = [
                    copy.deepcopy(self.mp_trainer.master_params)
                    for _ in range(len(self.ema_rate))
                ]
            else:
                self.ema_params = []
                logger.log("EMA is disabled - skipping EMA parameter initialization")

        self.use_ddp = False
        self.ddp_model = self.model

    def _process_cond_for_device(self, cond):
        """Process condition dict to move tensors to device."""
        processed_cond = {}
        for k, v in cond.items():
            if k != 'path':
                if isinstance(v, np.ndarray):
                    processed_cond[k] = th.from_numpy(v).to(dist_util.dev())
                elif isinstance(v, th.Tensor):
                    processed_cond[k] = v.to(dist_util.dev())
                else:
                    processed_cond[k] = v
            else:
                processed_cond[k] = [i for i in v] if isinstance(v, list) else v
        return processed_cond

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # self.resume_step = 0

            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

        # Load LR scheduler state if using cosine schedule
        if self.use_cosine_lr and self.lr_scheduler is not None:
            scheduler_checkpoint = bf.join(
                bf.dirname(main_checkpoint), f"scheduler{self.resume_step:06}.pt"
            )
            if bf.exists(scheduler_checkpoint):
                logger.log(f"loading scheduler state from checkpoint: {scheduler_checkpoint}")
                scheduler_state_dict = dist_util.load_state_dict(
                    scheduler_checkpoint, map_location=dist_util.dev()
                )
                self.lr_scheduler.load_state_dict(scheduler_state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond, data = next(self.data)
            
            self.run_step(batch, cond, data)
            if self.step % self.log_interval == 0 :
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1


        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, data=None):
        # data dict contains: dino_features, depth, image, cam_K, T_velo_to_cam
        self.forward_backward(batch, cond, data)

        # Apply gradient clipping if specified
        if self.grad_clip > 0:
            th.nn.utils.clip_grad_norm_(self.mp_trainer.master_params, self.grad_clip)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()


        if self.step % self.log_interval == 0:
            self._sample_and_visualize()



    def _sample_and_visualize(self):
        print("Sampling and visualizing...")
        self.ddp_model.eval()

        batch, cond, data = next(self.val_data)
        cond = self._process_cond_for_device(cond)
        # Add data dict to model_kwargs for sampling
        if data is not None:
            cond['data'] = data

        _shape = [len(cond['path'])] + list(batch.shape[1:])

        
        
        with th.no_grad():
            noise = None
            sample = self.diffusion.p_sample_loop(self.ddp_model, _shape, noise = noise, progress=True, model_kwargs=cond, clip_denoised=False)
        prediction=sample
        sample = sample.detach().cpu().numpy()
        feat_dim = sample.shape[1]

        
        # For triplane models, use existing 2D visualization
        for i in range(sample.shape[0]):
            for c in range(feat_dim//4):
                fig = draw_scalar_field2D(sample[i, c*4])
                self.tb.add_figure(f"sample{i}/channel{c*4}", fig, global_step=self.step)
            
            for c in range(feat_dim//4):
                fig = draw_scalar_field2D(batch[i, c*4].detach().cpu().numpy())
                self.tb.add_figure(f"sample{i}/gt{c*4}", fig, global_step=self.step)
        losses={}
        batch=batch.cuda()

        model_output_xy, model_output_xz, model_output_yz = decompose_featmaps(prediction, self.args.tri_size)
        gt_triplane_xy, gt_triplane_xz, gt_triplane_yz = decompose_featmaps(batch, self.args.tri_size)

        losses["l2_xy_val"] = mean_flat((gt_triplane_xy - model_output_xy)**2)
        
        losses["l2_xz_val"] = mean_flat((gt_triplane_xz - gt_triplane_xz)**2)
        losses["l2_yz_val"] = mean_flat((gt_triplane_yz - gt_triplane_yz)**2)
        
        losses["loss_val"] = losses["l2_xy_val"] + losses["l2_xz_val"] + losses["l2_yz_val"]

               

        self.log_loss_dict( self.diffusion, None, {k: v  for k, v in losses.items()},val=True)

        self.ddp_model.train()


    def forward_backward(self, batch, cond, data=None):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch.to(dist_util.dev())
            micro_cond = self._process_cond_for_device(cond)

            # Add data dict to model_kwargs if available
            if data is not None:
                micro_cond['data'] = data

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,)

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.mp_trainer.backward(loss)

           

            if self.step % 10 == 0:
                self.log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

    def _update_ema(self):
        if not self.use_ema:
            return
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if self.use_constant_lr:
            # Constant learning rate - no adjustment needed
            return
        elif self.use_cosine_lr and self.lr_scheduler is not None:
            # Use cosine schedule with warmup
            self.lr_scheduler.step()
        elif self.lr_anneal_steps:
            # Use linear annealing
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.lr * (1 - frac_done)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])
        if self.step % 10 == 0:
            self.tb.add_scalar("step", self.step + self.resume_step, global_step=self.step)
            self.tb.add_scalar("samples", (self.step + self.resume_step + 1) * self.global_batch, global_step=self.step)
            self.tb.add_scalar("lr", self.opt.param_groups[0]["lr"], global_step=self.step)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        if self.use_ema:
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)
        else:
            # Save the main model when EMA is disabled
            logger.log("EMA is disabled - saving main model instead")
            save_checkpoint(0, self.mp_trainer.master_params)

        # if dist.get_rank() == 0:
        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        # Save LR scheduler state if using cosine schedule
        if self.use_cosine_lr and self.lr_scheduler is not None:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"scheduler{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.lr_scheduler.state_dict(), f)

        # dist.barrier()

    def log_loss_dict(self, diffusion, ts, losses,val=False):
        for key, values in losses.items():
            loss_dict = {}
            logger.logkv_mean(key, values.mean().item())
            loss_dict[f"{key}_mean"] = values.mean().item()
            # Log the quantiles (four quartiles, in particular).
            if not val :
                for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                    quartile = int(4 * sub_t / diffusion.num_timesteps)
                    logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
                    loss_dict[f"{key}_q{quartile}"] = sub_loss
            self.tb.add_scalars(f"{key}", loss_dict, global_step=self.step)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("_")[-1].split(".")[0]
    return int(split)


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
