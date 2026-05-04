from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_builder import dataset_builder
from encoding.networks import  create_autoencoder
from omegaconf import OmegaConf
from encoding.lovasz import lovasz_softmax
from utils.utils import save_remap_lut, point2voxel
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from encoding.ssc_metrics import SSCMetrics
from diffusion.nn import mean_flat
from encoding.lovasz import lovasz_softmax
from utils.utils import make_query



class Trainer:
    def __init__(self, args):
        # etc
        self.args = args
        self.writer = SummaryWriter(os.path.join(args.save_path, 'tb'))
        self.epoch, self.start_epoch = 0, 0
        self.global_step = 0
        self.best_miou = 0

        # dataset
        self.train_dataset, self.val_dataset, self.num_class, class_names = dataset_builder(args)
        
        def collate_skip_none(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                return None
            return torch.utils.data.dataloader.default_collate(batch)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.bs, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, persistent_workers=(self.args.num_workers > 0), collate_fn=collate_skip_none)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, persistent_workers=(self.args.num_workers > 0), collate_fn=collate_skip_none)
        self.iou_class_names = class_names


        # model & optimizer
        if self.args.semantic_training_or_generation :
            self.model = create_autoencoder(args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.lr_scheduler_steps, args.lr_scheduler_decay) if args.lr_scheduler else None
        if self.args.mixed_precision:
            self.grad_scaler = GradScaler()



        if args.resume:
            print(f"Loading checkpoint from {args.resume} for training")
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']

        # loss functions
        self.loss_fns = {}
        # Get weights from original dataset (works even if train_dataset is a Subset)
        dataset_for_weights = self.train_dataset_original if hasattr(self, 'train_dataset_original') else self.train_dataset
        
        ignore_255 = getattr(self.args, 'ignore_255', True)
        if ignore_255:
            self.loss_fns['ce'] = torch.nn.CrossEntropyLoss(weight=dataset_for_weights.weights, ignore_index=255)
        else:
            self.loss_fns['ce'] = torch.nn.CrossEntropyLoss(weight=dataset_for_weights.weights)
        self.loss_fns['lovasz'] = None
        self.loss_fns['vq'] = None
        self.vq_loss_weight = getattr(args, 'vq_loss_weight', 1.0)  # Default VQ loss weight
     

    def train(self):
        for epoch in range(self.args.nb_epochs):
            self.epoch = self.start_epoch + epoch + 1

            print('Training...')
            self._train_model()

            max_train_samples=getattr(self.args, 'max_train_samples', None)
            if max_train_samples is None:

                if epoch % self.args.eval_epoch == 0:
                    print('Evaluation...')
                    self._eval_and_save_model()
            # learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            self.writer.add_scalar('lr_epochwise', self.optimizer.param_groups[0]['lr'], global_step=self.epoch)

    def _loss(self, losses,data):
        empty_label = 0.

        vox = data["voxel_label"].type(torch.LongTensor).cuda()
        query = data["query"].type(torch.FloatTensor).cuda()
        label = data["label"].type(torch.LongTensor).cuda()
        coord = data["coord"].type(torch.LongTensor).cuda()

        full_volume, vq_loss = self.model(vox,return_vq_loss=True)
        # Sample the volume at query points to get preds [B, N, C]
        B, N, _ = query.shape
        batch_indices = torch.arange(B).view(B, 1).expand(B, N).cuda()
        preds = full_volume[batch_indices, :, coord[:, :, 0], coord[:, :, 1], coord[:, :, 2]]
        losses['vq'] = vq_loss
        pred_volume_for_lovasz = full_volume
        losses['ce'] = self.loss_fns['ce'](preds.view(-1, self.num_class), label.view(-1,))
        pred_output_permuted = torch.nn.functional.softmax(pred_volume_for_lovasz, dim=1)
        gt_output = vox.float()
        losses['lovasz'] = lovasz_softmax(pred_output_permuted, gt_output)
        losses['loss'] = losses['ce']
        losses['loss'] = losses['loss'] + losses['lovasz'] + self.vq_loss_weight * losses['vq']


        adaptive_weight = None
        return losses, preds, adaptive_weight

    def _train_model(self):
        self.model.train()

        total_losses = {loss_name: 0. for loss_name in self.loss_fns.keys()}
        total_losses['loss'] = 0.
        evaluator = SSCMetrics(self.num_class, [])
        dataloader_tqdm = tqdm(self.train_dataloader)

        for data in dataloader_tqdm:
            vox = data["voxel_label"].type(torch.LongTensor).cuda()
            coord = data["coord"].type(torch.LongTensor).cuda()
            invalid = data["invalid"].type(torch.LongTensor).cuda()
            b_size = vox.size(0)
            path= data["path"]
            # forward
            losses = {}
            if self.args.mixed_precision:
                with autocast():
                    losses, model_output, adaptive_weight = self._loss(losses, data)
                # optimize
                self.optimizer.zero_grad()
                self.grad_scaler.scale(losses['loss']).backward()
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else :
                losses, model_output, adaptive_weight = self._loss(losses,data)
                self.optimizer.zero_grad()
                losses['loss'].backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
                self.optimizer.step()
            # eval and log each iteration
            if self.global_step % self.args.display_period == 0:
                returns_volume = getattr(self.args, 'ae_returns_volume', False)
                pred_mask = get_pred_mask(model_output, returns_volume=returns_volume)

                masks = torch.from_numpy(evaluator.get_eval_mask(vox.cpu().numpy(), invalid.cpu().numpy()))
                
                output = point2voxel(self.args, pred_mask, coord)
                eval_output = output[masks]
                eval_label = vox[masks]
                this_iou, this_miou = evaluator.addBatch(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))
                
               
                # on display
                dataloader_tqdm.set_postfix({"loss": losses['loss'].detach().item(),"vq": losses['vq'].detach().item(), "ce": losses['ce'].detach().item(),"lovasz": losses['lovasz'].detach().item(),"iou": this_iou, "miou": this_miou})
               

               # on tensorboard
                self.writer.add_scalar('Grad_Norm', grad_norm, global_step=self.global_step)
                self.writer.add_scalar('Train_Performance_stepwise/IoU', this_iou, global_step=self.global_step)
                self.writer.add_scalar('Train_Performance_stepwise/mIoU', this_miou, global_step=self.global_step)
                for loss_name in losses.keys():
                    self.writer.add_scalar(f'Train_Loss_stepwise/loss_{loss_name}', losses[loss_name], self.global_step)

            
            # loss accumulation for logging
            for loss_name in losses.keys():
                total_losses[loss_name] += (losses[loss_name].detach().item() * b_size)

            self.global_step += 1

        # eval for 1 epoch
        _, class_jaccard = evaluator.getIoU()
        m_jaccard = class_jaccard[1:].mean()
        miou = m_jaccard * 100
        conf = evaluator.get_confusion()
        iou = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0] + 1e-8)
        evaluator.reset()

        # log for 1 epoch
        self.writer.add_scalar('Train_Performance_epochwise/IoU', iou, global_step=self.epoch)
        self.writer.add_scalar('Train_Performance_epochwise/mIoU', miou, global_step=self.epoch)
        for class_idx, class_name in enumerate(self.iou_class_names):
            self.writer.add_scalar(f'Train_ClassPerformance_epochwise/class{class_idx + 1}_IoU_{class_name}', class_jaccard[class_idx + 1], global_step=self.epoch)
        for loss_name in losses.keys():
            self.writer.add_scalar(f'Train_Loss_epochwise/loss_{loss_name}', total_losses[loss_name] / len(self.train_dataset), global_step=self.epoch)

        print(f"Epoch: {self.epoch} \t IOU: \t {iou:01f} \t mIoU: \t {miou:01f}")


    @torch.no_grad()
    def _eval_and_save_model(self):
        self.model.eval()

        total_losses = {loss_name: 0. for loss_name in self.loss_fns.keys()}
        total_losses['loss'] = 0.
        evaluator = SSCMetrics(self.num_class, [])
        dataloader_tqdm = tqdm(self.val_dataloader)

        
        for sample_idx, data in enumerate(dataloader_tqdm):
            if data is None:  # batch entièrement skippé (ex: label_rasterization manquant)
                continue

            vox = data["voxel_label"].type(torch.LongTensor).cuda()
            coord = data["coord"].type(torch.LongTensor).cuda()
            invalid = data["invalid"].type(torch.LongTensor).cuda()
            path= data["path"]
            b_size = vox.size(0)  # TODO: check correctness
            assert b_size == 1, 'For accurate logging, please set batch size of validation dataloader to 1.'



            losses = {}
            losses, model_output, adaptive_weight = self._loss(losses, data)
            returns_volume = getattr(self.args, 'ae_returns_volume', False)
            pred_mask =  get_pred_mask(model_output, returns_volume=returns_volume)

            masks = torch.from_numpy(evaluator.get_eval_mask(vox.cpu().numpy(), invalid.cpu().numpy()))
            
            output = point2voxel(self.args, pred_mask, coord)
            eval_output = output[masks]
            eval_label = vox[masks]
            this_iou, this_miou = evaluator.addBatch(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))

            # log on display for each sample
            dataloader_tqdm.set_postfix({"loss": losses['loss'].detach().item(), "iou": this_iou, "miou": this_miou})

            for loss_name in losses.keys():
                total_losses[loss_name] += (losses[loss_name] * b_size)

            idx = path[0].split('/')[-1].split('.')[0]
            folder = path[0].split('/')[-3]
            # Use original dataset for learning_map_inv (works even if train_dataset is a Subset)
            dataset_for_attrs = self.train_dataset_original if hasattr(self, 'train_dataset_original') else self.train_dataset
            save_remap_lut(self.args, output, folder, idx, dataset_for_attrs.learning_map_inv, True)

        # eval for all validation samples
        _, class_jaccard = evaluator.getIoU()
        m_jaccard = class_jaccard[1:].mean()
        miou = m_jaccard * 100
        conf = evaluator.get_confusion()
        iou = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0] + 1e-8)
        evaluator.reset()

        self.writer.add_scalar('Val_Performance_epochwise/IoU', iou, global_step=self.epoch)
        self.writer.add_scalar('Val_Performance_epochwise/mIoU', miou, global_step=self.epoch)
        for class_idx, class_name in enumerate(self.iou_class_names):
            self.writer.add_scalar(f'Val_ClassPerformance_epochwise/class{class_idx + 1}_IoU_{class_name}', class_jaccard[class_idx + 1], global_step=self.epoch)
        for loss_name in losses.keys():
            self.writer.add_scalar(f'Val_Loss_epochwise/loss_{loss_name}', total_losses[loss_name] / len(self.val_dataset), global_step=self.epoch)
        print(f"Epoch: {self.epoch} \t IOU: \t {iou:01f} \t mIoU: \t {miou:01f}")

        if self.best_miou < miou:
            self.best_miou = miou
            checkpoint = {'optimizer': self.optimizer.state_dict(), 'model': self.model.state_dict(), 'epoch': self.epoch}  # TODO: save scheduler
            torch.save(checkpoint, self.args.save_path + "/" + str(self.epoch) + "_miou=" + str(f"{miou:.3f}") + '.pt')




def get_pred_mask(model_output, separate_decoder=False, returns_volume=False):
    preds = model_output
    if returns_volume:
        # Volume output: [B, X, Y, Z, C]
        pred_prob = torch.softmax(preds, dim=4)  # Softmax over class dimension
        pred_mask = pred_prob.argmax(dim=4).float()  # [B, X, Y, Z]
    else:
        # Query points output: [B, N, C]
        pred_prob = torch.softmax(preds, dim=2)
        pred_mask = pred_prob.argmax(dim=2).float()  # [B, N]
    return pred_mask

def mean_flat_f(dense_triplane,sparse_triplane):
    """
    Take the mean over all non-batch dimensions.
    """
    l2_xy = mean_flat((dense_triplane[0] - sparse_triplane[0])**2)
    l2_xz = mean_flat((dense_triplane[1] - sparse_triplane[1])**2)
    l2_yz = mean_flat((dense_triplane[2] - sparse_triplane[2])**2)
    loss = l2_xy + l2_xz + l2_yz
    return loss

