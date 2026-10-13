import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import albumentations as A
import numpy as np
import os
import random
import logging
import argparse
from tqdm import tqdm
import shutil
import hashlib
from importlib.metadata import distributions
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict

from models.SemiEchoTracker import SemiEchoTracker
from models.dinov2 import get_dinov2_model
from dataset.echo_dataset import make_dataset


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='SemiEchoTracker')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Path to the dataset')
    parser.add_argument('--data_type', type=str, default='PLAX', help='Data type')
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224), help='Image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--frame_num', type=int, default=10, help='Number of frames')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--tracker_lr_multiplier', type=float, default=1.0,
                        help='Learning-rate multiplier for tracker-only parameters')
    parser.add_argument('--detector_lr_multiplier', type=float, default=1.0,
                        help='Learning-rate multiplier for detector/shared parameters')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # loss weight
    parser.add_argument('--velocity', type=float, default=0.1, help='Weight of velocity loss')
    parser.add_argument('--tracking', type=float, default=0.1, help='Weight of tracking loss')
    parser.add_argument('--tracking_supervised', type=float, default=1.0, help='Weight of supervised tracker endpoint loss')
    parser.add_argument('--mutual_consistency', type=float, default=0.1, help='Weight of mutual consistency loss')
    parser.add_argument('--endpoint_supervision_weight', type=float, default=1.0,
                        help='Weight for detector endpoint supervision in the total loss')
    parser.add_argument('--agent_num', type=int, default=32, help='number of agent tokens')
    parser.add_argument('--attention_layers', type=int, default=3,
                        help='Number of trainable EchoAgentAttention blocks after frozen DINO')
    parser.add_argument('--feature_branch_mode', type=str, default='shared',
                        choices=['shared', 'split_attention'],
                        help='shared uses one attention branch; split_attention gives detector/tracker separate attention branches')
    parser.add_argument('--coord_head', type=str, default='conv_mlp_keypoint_cnn_gcn_noxy',
                        choices=['pooled_gcn', 'fullres_gcn', 'query_attn',
                                 'conv_mlp', 'conv_mlp_gcn', 'conv_mlp_spatial_gcn',
                                 'conv_mlp_keypoint_gcn', 'conv_mlp_keypoint_cnn_gcn',
                                 'conv_mlp_keypoint_cnn_gcn_noxy'],
                        help='Direct coordinate head; fullres_gcn keeps the native 16x16 feature grid')
    parser.add_argument('--graph_mode', type=str, default='split', choices=['split', 'shared'],
                        help='Use separate detector/tracker landmark graphs or share one graph')
    parser.add_argument('--training_mode', type=str, default='semi', choices=['semi', 'detector_only'],
                        help='semi uses detector+tracker losses; detector_only trains only the frame-wise detector on endpoints')
    parser.add_argument('--mode', type=str, default=None, choices=['semi', 'detector_only'],
                        help='Alias for --training_mode, useful for ablation scripts.')
    parser.add_argument('--detector_only', action='store_true',
                        help='Alias for --training_mode detector_only')
    parser.add_argument('--aug_mode', type=str, default='standard', choices=['standard', 'weak', 'none'],
                        help='Training augmentation strength')
    parser.add_argument('--rotation_limit', type=float, default=30.0,
                        help='Clip-level rotation limit in degrees for training')
    parser.add_argument('--rotation_prob', type=float, default=0.5,
                        help='Clip-level rotation probability for training')
    
    # training settings
    parser.add_argument('--warmup_epochs', type=int, default=30, help='Number of warmup epochs')
    parser.add_argument('--cons_epoch', type=int, default=0, help='Epoch to start consistency loss between detector and tracker')
    parser.add_argument('--consistency_ramp_epochs', type=int, default=0,
                        help='Linearly ramp detector-tracker consistency weight over this many epochs after --cons_epoch')
    parser.add_argument('--detach_tracker_teacher', action='store_true',
                        help='Detach tracker coordinates in detector-tracker consistency so tracker teaches detector')
    parser.add_argument('--detach_tracker_inputs', action='store_true',
                        help='Detach shared features, detector seeds, and shared adjacency before the tracker branch')
    parser.add_argument('--consistency_gate', type=str, default='none', choices=['none', 'fb'],
                        help='Mask detector-tracker consistency by tracker reliability')
    parser.add_argument('--consistency_fb_threshold', type=float, default=8.0,
                        help='Pixel threshold for forward/backward tracker agreement when --consistency_gate fb')
    parser.add_argument('--consistency_mask_normalization', type=str, default='selected',
                        choices=['selected', 'total'],
                        help='Normalize masked consistency by selected points or by all candidate points')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short results directory name for long ablation configs')
    parser.add_argument('--best_metric', type=str, default=None,
                        choices=['err_detection_endpoint', 'err_init', 'full_sequence_tracking', 'err_tracking'],
                        help='Optional validation metric used to save best_model.pth')
    parser.add_argument('--num_workers', type=int, default=6, help='DataLoader worker processes')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Accumulate gradients over this many mini-batches before each optimizer step')
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adamw')
    parser.add_argument('--seed', type=int, default=33)
    parser.add_argument('--backbone_source', choices=['guided', 'public'], default='guided')
    parser.add_argument('--backbone_repo', type=str)
    parser.add_argument('--backbone_checkpoint', type=str)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--resume', type=str, help='Resume an exact configuration from latest_model.pth')
    args = parser.parse_args(argv)
    args.image_size = tuple(args.image_size)
    if args.mode is not None:
        args.training_mode = args.mode
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error('batch_size and grad_accum_steps must be positive')
    if args.detector_only:
        args.training_mode = 'detector_only'
    return args


def process_features(dinov2_model, inputs, B, T, H, W):
    """Extract DINOv2 patch tokens + CLS token.

    Args:
        dinov2_model: frozen DINOv2 backbone
        inputs: [B*T, C, H, W] image tensor
    Returns:
        extra_feature: [B, T, C_dino, H//14, W//14]
        cls_token:     [B, T, C_dino]
    """
    with torch.no_grad():
        extra_feature_dict = dinov2_model.forward_features(inputs)
        extra_feature = extra_feature_dict['x_norm_patchtokens']
        cls_token = extra_feature_dict['x_norm_clstoken']           # [B*T, C]
        cls_token = cls_token.reshape(B, T, -1)

        extra_feature = extra_feature.reshape(B * T, H // 14, W // 14, -1)
        extra_feature = extra_feature.permute(0, 3, 1, 2).contiguous()   # [B*T, C, H//14, W//14]
        extra_feature = extra_feature.reshape(B, T, -1, H // 14, W // 14)
    return extra_feature, cls_token


def coordinate_loss_elementwise(pred, target):
    """论文的逐点欧氏距离；FP32 避免混合精度下小误差丢失。"""
    return torch.linalg.vector_norm(pred.float() - target.float(), dim=-1)


def coordinate_loss(pred, target):
    return coordinate_loss_elementwise(pred, target).mean()


def masked_coordinate_loss(pred, target, point_weights, args):
    distances = coordinate_loss_elementwise(pred, target)
    if point_weights is None:
        return distances.mean()
    weights = point_weights.to(distances)
    denominator = distances.numel() if args.consistency_mask_normalization == 'total' else weights.sum().clamp_min(1)
    return (distances * weights).sum() / denominator


def velocity_consistency_loss(forward_offsets, backward_offsets):
    # 返回的反向 offsets 已按时间正序排列；两侧对齐中间帧的到达速度。
    forward_speed = torch.linalg.vector_norm(forward_offsets[:, :-1].float(), dim=-1)
    backward_speed = torch.linalg.vector_norm(backward_offsets[:, 1:].float(), dim=-1)
    return (forward_speed - backward_speed).abs().mean()


def co_training_loss(detector, forward, backward, weights, args):
    return (masked_coordinate_loss(detector, forward, weights, args)
            + masked_coordinate_loss(detector, backward, weights, args))


def prediction_errors_mm(pred, target, batch, data_type):
    difference = pred.float() - target.float()
    if data_type == 'echoxflow':
        linear = batch['input_to_mm'].to(device=pred.device, dtype=torch.float32)[:, :2, :2]
        difference = torch.einsum('bij,btnj->btni', linear, difference)
    else:
        scale = torch.stack([batch['rescale_x'], batch['rescale_y']], dim=-1).to(difference)
        difference = difference * scale[:, None, None] * (0.6 if data_type == 'PLAX' else 0.67)
    return torch.linalg.vector_norm(difference, dim=-1)


def labelled_predictions(pred, batch):
    if 'label_frame_indices' in batch:
        indices = batch['label_frame_indices'].to(pred.device)
    elif batch['landmarks'].shape[1] == 2:
        indices = torch.tensor([0, pred.shape[1]-1], device=pred.device).expand(pred.shape[0], -1)
    else:
        indices = torch.arange(pred.shape[1], device=pred.device).expand(pred.shape[0], -1)
    return pred.gather(1, indices[:, :, None, None].expand(-1, -1, pred.shape[2], 2))


def tracker_agreement_weights(forward_coords, backward_coords, args):
    if args.consistency_gate == 'none':
        return None
    agreement = torch.norm(forward_coords.detach() - backward_coords.detach(), dim=-1)
    if args.consistency_gate == 'fb':
        return (agreement <= args.consistency_fb_threshold).float()
    raise ValueError(f'Unknown consistency_gate: {args.consistency_gate}')


def migrate_split_graph_state_dict(state_dict):
    """Map older checkpoints with one shared graph to split graph params."""
    state_dict = dict(state_dict)
    old_key = 'sttgnn.adj'
    detector_key = 'sttgnn.detector_adj'
    tracker_key = 'sttgnn.tracker_adj'
    if old_key in state_dict:
        old_adj = state_dict.pop(old_key)
        state_dict.setdefault(detector_key, old_adj.clone())
        state_dict.setdefault(tracker_key, old_adj.clone())
    return state_dict


def is_tracker_parameter(name):
    tracker_prefixes = (
        'tracker_feature_module.',
        'tracker_attention_modules.',
        'sttgnn.gc1.',
        'sttgnn.gc2.',
        'sttgnn.gc3.',
        'sttgnn.tracker_adj',
        'sttgnn.pred_head.',
        'sttgnn.gru_projector.',
        'sttgnn.gru_forward.',
        'sttgnn.gru_backward.',
        'sttgnn.gru_deprojector.',
    )
    return name.startswith(tracker_prefixes)



def train_or_validate(model, dinov2_model, data_loader, optimizer, device, epoch,
                     writer, scaler, args, is_training=True):
    model.train(is_training)
    dinov2_model.eval()
    totals, errors = defaultdict(float), defaultdict(list)
    sample_count = 0
    detector_only = not is_training or args.training_mode == 'detector_only'
    accum = args.grad_accum_steps if is_training else 1
    split = 'train' if is_training else 'val'
    progress = tqdm(data_loader, desc=f'{split} {epoch + 1}/{args.epochs}',
                    disable=not sys.stderr.isatty(), mininterval=10)
    for batch_index, batch in enumerate(progress):
        if is_training and batch_index % accum == 0:
            optimizer.zero_grad(set_to_none=True)
            group_start = batch_index * data_loader.batch_size
            group_samples = min(accum * data_loader.batch_size, len(data_loader.dataset) - group_start)
        inputs = batch['video'].to(device, non_blocking=True).float()
        landmarks = batch['landmarks'].to(device, non_blocking=True).float()
        B, T, C, H, W = inputs.shape
        extra_feature, cls_token = process_features(dinov2_model, inputs.reshape(B*T, C, H, W), B, T, H, W)
        del inputs
        with torch.set_grad_enabled(is_training):
            amp_enabled = is_training and str(device).startswith('cuda') and args.precision != 'fp32'
            with torch.autocast(device_type=torch.device(device).type, enabled=amp_enabled,
                                dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16):
                if detector_only:
                    detector = model(extra_feature, cls_token, infer_mode=True)
                else:
                    seeds = landmarks[:, [0, -1]] / landmarks.new_tensor([W-1, H-1])
                    forward, backward, forward_offsets, backward_offsets, detector = model(
                        extra_feature, cls_token, tracker_seed_coords=seeds,
                        detach_tracker_inputs=args.detach_tracker_inputs)
            if not torch.isfinite(detector).all():
                raise FloatingPointError(f'Non-finite detector output: epoch={epoch}, samples={batch["patient_name"]}')
            losses = dict(landmarks=coordinate_loss(detector[:, 0], landmarks[:, 0])
                                   + coordinate_loss(detector[:, -1], landmarks[:, -1]))
            loss = losses['landmarks'] * args.endpoint_supervision_weight
            if not detector_only:
                losses['tracking_supervised'] = (coordinate_loss(forward[:, -1], landmarks[:, -1])
                                                  + coordinate_loss(backward[:, 0], landmarks[:, 0]))
                losses['tracking'] = coordinate_loss(forward[:, 1:-1], backward[:, 1:-1])
                losses['velocity'] = velocity_consistency_loss(forward_offsets, backward_offsets)
                loss = loss + losses['tracking_supervised'] * args.tracking_supervised
                loss = loss + losses['tracking'] * args.tracking + losses['velocity'] * args.velocity
                if epoch >= args.cons_epoch:
                    fwd, bwd = forward[:, 1:-1], backward[:, 1:-1]
                    if args.detach_tracker_teacher:
                        fwd, bwd = fwd.detach(), bwd.detach()
                    weights = tracker_agreement_weights(fwd, bwd, args)
                    losses['consistency'] = co_training_loss(detector[:, 1:-1], fwd, bwd, weights, args)
                    ramp = min(1., (epoch - args.cons_epoch + 1) / max(1, args.consistency_ramp_epochs))
                    loss = loss + losses['consistency'] * args.mutual_consistency * ramp
            losses['total'] = loss
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError(f'Non-finite loss: epoch={epoch}, samples={batch["patient_name"]}, losses={losses}')
            if is_training:
                scaler.scale(loss * B / group_samples).backward()
                if (batch_index + 1) % accum == 0 or batch_index + 1 == len(data_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
        with torch.no_grad():
            detector_error = prediction_errors_mm(labelled_predictions(detector, batch), landmarks, batch, args.data_type)
            errors['err_init'].append(detector_error.cpu().numpy().reshape(-1))
            errors['err_detection_endpoint'].append(detector_error[:, [0, -1]].cpu().numpy().reshape(-1))
            if not detector_only:
                tracked = torch.stack([backward[:, 0], forward[:, -1]], dim=1)
                tracking_error = prediction_errors_mm(tracked, landmarks[:, [0, -1]], batch, args.data_type)
                errors['err_tracking'].append(tracking_error.cpu().numpy().reshape(-1))
        sample_count += B
        for name, value in losses.items():
            totals['loss' if name == 'total' else 'loss_' + name] += value.item() * B
    metrics = {key: value / sample_count for key, value in totals.items()}
    metrics.update({key: float(np.concatenate(value).mean()) for key, value in errors.items()})
    metrics.setdefault('err_tracking', metrics['err_init'])
    metrics['full_sequence_tracking'] = metrics['err_init']
    metrics['videos'] = sample_count
    metrics['reference_points'] = sum(len(value) for value in errors['err_init'])
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'{split}/{key}', value, epoch)
    logging.info('Epoch %d/%d %s: loss=%.4f detector MRE=%.4f mm endpoint MRE=%.4f mm videos=%d',
                 epoch+1, args.epochs, split, metrics['loss'], metrics['err_init'],
                 metrics['err_detection_endpoint'], sample_count)
    return metrics


def train(model, dinov2_model, train_loader, optimizer, device, epoch, writer, scaler, args):
    """Training function wrapper"""
    return train_or_validate(model, dinov2_model, train_loader, optimizer, device, 
                           epoch, writer, scaler, args, is_training=True)


def validate(model, dinov2_model, val_loader, device, epoch, writer, args):
    """Validation function wrapper"""
    return train_or_validate(model, dinov2_model, val_loader, None, device,
                           epoch, writer, None, args, is_training=False)


def get_augmentations(args):
    """Get data augmentation transforms.

    Train uses `A.ReplayCompose` so that the same random augmentation is applied
    to every frame in a clip (consistent temporal appearance); the first frame
    records the replay params and subsequent frames replay them.
    """
    if args.aug_mode == 'standard':
        train_transforms = [
            A.Resize(args.image_size[0], args.image_size[1]),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(p=0.2),
        ]
    elif args.aug_mode == 'weak':
        train_transforms = [
            A.Resize(args.image_size[0], args.image_size[1]),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        ]
    else:
        train_transforms = [
            A.Resize(args.image_size[0], args.image_size[1]),
        ]

    return {
        "train": A.ReplayCompose(train_transforms),
        "val": A.Compose([
            A.Resize(args.image_size[0], args.image_size[1]),
        ])
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    import cv2
    cv2.setNumThreads(1)


def make_model(args):
    return SemiEchoTracker(agent_num=args.agent_num, nodes_num=11 if args.data_type == 'PLAX' else 3,
                           image_size=args.image_size[0], coord_head=args.coord_head,
                           graph_mode=args.graph_mode, attention_layers=args.attention_layers,
                           feature_branch_mode=args.feature_branch_mode)


def load_backbone(args):
    return get_dinov2_model(source=args.backbone_source, hub_dir=args.backbone_repo,
                           checkpoint=args.backbone_checkpoint, device=args.device)


def make_optimizer(model, args):
    detector_params, tracker_params = [], []
    for name, parameter in model.named_parameters():
        (tracker_params if is_tracker_parameter(name) else detector_params).append(parameter)
    optimizer_class = optim.Adam if args.optimizer == 'adam' else optim.AdamW
    return optimizer_class([
        dict(params=detector_params, lr=args.lr * args.detector_lr_multiplier, name='detector_shared'),
        dict(params=tracker_params, lr=args.lr * args.tracker_lr_multiplier, name='tracker'),
    ], lr=args.lr, weight_decay=args.weight_decay)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def backbone_code_hash(path):
    if path is None:
        return None
    root = Path(path)
    digest = hashlib.sha256()
    for source in sorted(root.rglob('*.py')):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def source_files(root):
    # 显式白名单，避免递归包含 results、权重、虚拟环境及数据。
    files = list(root.glob('*.py'))
    for directory in ('models', 'dataset', 'tests'):
        files.extend((root / directory).rglob('*.py'))
    return sorted(files)


def atomic_checkpoint(checkpoint, path):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(), loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def main():
    args = parse_args()
    seed_everything(args.seed)
    root = Path(__file__).resolve().parent
    results = root / 'results' / args.data_type / (args.run_tag or f'seed{args.seed}_{args.training_mode}')
    if args.resume is None and results.exists() and any(results.iterdir()):
        raise FileExistsError(f'Run already exists; use --resume explicitly: {results}')
    results.mkdir(parents=True, exist_ok=True)
    setup_logging(results)
    logging.info('Arguments: %s', vars(args))
    aug = get_augmentations(args)
    train_dataset = make_dataset(root_dir=args.data_dir, transform=aug['train'], splits='train',
                                 frame_num=args.frame_num, target_size=args.image_size, data_type=args.data_type,
                                 rotation_limit=args.rotation_limit, rotation_prob=args.rotation_prob)
    val_dataset = make_dataset(root_dir=args.data_dir, transform=aug['val'], splits='val',
                               frame_num=args.frame_num, target_size=args.image_size, data_type=args.data_type)
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                       worker_init_fn=seed_worker, generator=generator,
                       prefetch_factor=2 if args.num_workers else None, persistent_workers=False)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    model = make_model(args).to(args.device)
    backbone = load_backbone(args)
    optimizer = make_optimizer(model, args)
    if 0 < args.warmup_epochs < args.epochs:
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=.1, total_iters=args.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs),
        ], milestones=[args.warmup_epochs])
    elif args.warmup_epochs:
        scheduler = LinearLR(optimizer, start_factor=.1, total_iters=args.epochs)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024,
                                 enabled=str(args.device).startswith('cuda') and args.precision == 'fp16')
    sources = {str(path.relative_to(root)): file_sha(path) for path in source_files(root)}
    provenance = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                      source_sha256=sources, data_manifest_sha256=getattr(train_dataset, 'manifest_sha256', None),
                      backbone_sha256=file_sha(args.backbone_checkpoint) if args.backbone_checkpoint else None,
                      backbone_code_sha256=backbone_code_hash(args.backbone_repo),
                      backbone_repo=args.backbone_repo, command=sys.argv,
                      python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                      effective_batch_size=args.batch_size * args.grad_accum_steps,
                      train_videos=len(train_dataset), val_videos=len(val_dataset))
    start_epoch, best_value, best_epoch = 0, float('inf'), None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        previous_args = checkpoint['args']
        for key, value in vars(args).items():
            if key != 'resume' and value != previous_args[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        for key in ('source_sha256', 'data_manifest_sha256', 'backbone_sha256', 'backbone_code_sha256'):
            if provenance[key] != checkpoint['provenance'][key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        restore_rng(checkpoint['rng_state'], generator)
        start_epoch = checkpoint['epoch'] + 1
        best_value, best_epoch = checkpoint['best_value'], checkpoint['best_epoch']
        logging.info('Resuming epoch %d', start_epoch + 1)
    else:
        for source in source_files(root) + [root / 'README.md']:
            destination = results / 'code' / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (results / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
        (results / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        packages = sorted(f'{dist.metadata["Name"]}=={dist.version}' for dist in distributions())
        (results / 'environment.txt').write_text('\n'.join(packages) + '\n')
        if hasattr(train_dataset, 'manifest_path'):
            shutil.copy2(train_dataset.manifest_path, results / 'data_manifest.json')
    writer = SummaryWriter(log_dir=str(results / 'tensorboard'))
    best_metric = args.best_metric or ('err_detection_endpoint' if args.training_mode == 'detector_only' else 'err_init')
    logging.info('Training starts: %d parameters; effective batch=%d; precision=%s; select by %s',
                 sum(p.numel() for p in model.parameters()), provenance['effective_batch_size'], args.precision, best_metric)
    try:
        for epoch in range(start_epoch, args.epochs):
            started = time.monotonic()
            learning_rate = scheduler.get_last_lr()[0]
            train_metrics = train(model, backbone, train_loader, optimizer, args.device, epoch, writer, scaler, args)
            val_metrics = validate(model, backbone, val_loader, args.device, epoch, writer, args)
            scheduler.step()
            improved = val_metrics[best_metric] < best_value
            if improved:
                best_value, best_epoch = val_metrics[best_metric], epoch
            checkpoint = dict(epoch=epoch, args=vars(args), provenance=provenance,
                              model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                              scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
                              rng_state=rng_state(generator), train_metrics=train_metrics, val_metrics=val_metrics,
                              best_value=best_value, best_epoch=best_epoch, best_metric=best_metric)
            atomic_checkpoint(checkpoint, results / 'latest_model.pth')
            if improved:
                atomic_checkpoint(checkpoint, results / 'best_model.pth')
                logging.info('Best detector checkpoint: epoch=%d MRE=%.4f mm', epoch+1, best_value)
            with (results / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch+1, learning_rate=learning_rate,
                                             seconds=time.monotonic()-started, train=train_metrics, val=val_metrics),
                                        allow_nan=False) + '\n')
            writer.flush()
        logging.info('Training completed: %d epochs; best epoch=%d, MRE=%.4f mm', args.epochs, best_epoch+1, best_value)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
