"""按检查点配置评估检测器；毫米误差、逐点预测与参考一起保存。"""

import argparse
import json
from pathlib import Path
import sys

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.echo_dataset import make_dataset
from train import (backbone_code_hash, file_sha, labelled_predictions, load_backbone, make_model,
                   prediction_errors_mm, process_features)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Run directory or checkpoint file')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_type', choices=['PLAX', 'A4C', 'echoxflow'])
    parser.add_argument('--image_size', type=int, nargs=2)
    parser.add_argument('--frame_num', type=int)
    parser.add_argument('--agent_num', type=int)
    parser.add_argument('--attention_layers', type=int)
    parser.add_argument('--feature_branch_mode', choices=['shared', 'split_attention'])
    parser.add_argument('--coord_head')
    parser.add_argument('--graph_mode', choices=['split', 'shared'])
    parser.add_argument('--backbone_repo', help='Relocated backbone source with identical contents')
    parser.add_argument('--backbone_checkpoint', help='Relocated backbone weights with identical hash')
    parser.add_argument('--no_visuals', action='store_true')
    return parser.parse_args()


def resolve_checkpoint(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'best_model.pth'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_configuration(checkpoint, cli):
    config = dict(checkpoint['args'])
    for key, default in (('backbone_source', 'guided'), ('backbone_repo', None), ('backbone_checkpoint', None)):
        config.setdefault(key, default)
    for key in ('data_type', 'image_size', 'frame_num', 'agent_num', 'attention_layers',
                'feature_branch_mode', 'coord_head', 'graph_mode'):
        override = getattr(cli, key)
        if override is not None:
            expected = list(config[key]) if key == 'image_size' else config[key]
            if override != expected:
                raise ValueError(f'Checkpoint configuration mismatch: {key}={override}, expected {expected}')
    config['device'] = cli.device
    for key in ('data_dir', 'backbone_repo', 'backbone_checkpoint'):
        if getattr(cli, key) is not None:
            config[key] = getattr(cli, key)
    return argparse.Namespace(**config)


def summarize_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if not errors.size or not np.isfinite(errors).all():
        raise ValueError('Evaluation requires finite, nonempty errors')
    return dict(mre_mm=float(errors.mean()), std_mm=float(errors.std(ddof=0)),
                videos=errors.shape[0], frames=errors.shape[0]*errors.shape[1], points=errors.size,
                sdr_percent={f'{threshold}mm': float((errors <= threshold).mean()*100)
                             for threshold in (2, 3, 4)})


def main():
    cli = parse_args()
    checkpoint_path = resolve_checkpoint(cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint_configuration(checkpoint, cli)
    dataset = make_dataset(root_dir=args.data_dir, data_type=args.data_type, splits=cli.split,
                           frame_num=args.frame_num, target_size=args.image_size,
                           transform=A.Compose([A.Resize(*args.image_size)]), rotation_prob=0)
    provenance = checkpoint.get('provenance', {})
    if args.data_type == 'echoxflow':
        if dataset.manifest_sha256 != provenance['data_manifest_sha256']:
            raise ValueError('Data version differs from the training checkpoint')
        if file_sha(args.backbone_checkpoint) != provenance['backbone_sha256']:
            raise ValueError('Backbone weights differ from the training checkpoint')
        if backbone_code_hash(args.backbone_repo) != provenance['backbone_code_sha256']:
            raise ValueError('Backbone source differs from the training checkpoint')
    model = make_model(args).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    backbone = load_backbone(args)
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers)
    output = Path(cli.output_dir)
    if not output.is_absolute():
        output = checkpoint_path.parent / output
    output = output / cli.split
    output.mkdir(parents=True, exist_ok=True)
    collected = {key: [] for key in ('pred', 'reference', 'error_mm', 'label_frame_indices', 'patient_name')}
    if args.data_type == 'echoxflow':
        collected.update({key: [] for key in ('input_to_mm', 'frame_indices', 'timestamps_seconds')})
    with torch.inference_mode():
        for batch in loader:
            video = batch['video'].to(args.device).float()
            B, T, C, H, W = video.shape
            features, cls = process_features(backbone, video.reshape(B*T, C, H, W), B, T, H, W)
            # 推理只输入图像特征，没有真实关键点种子或平滑。
            predictions = model(features, cls, infer_mode=True)
            reference = batch['landmarks'].to(args.device).float()
            errors = prediction_errors_mm(labelled_predictions(predictions, batch), reference, batch, args.data_type)
            if not torch.isfinite(predictions).all() or not torch.isfinite(errors).all():
                raise FloatingPointError(f'Non-finite predictions: {batch["patient_name"]}')
            collected['pred'].append(predictions.cpu().numpy())
            collected['reference'].append(reference.cpu().numpy())
            collected['error_mm'].append(errors.cpu().numpy())
            collected['patient_name'].extend(batch['patient_name'])
            frame_labels = batch.get('label_frame_indices')
            if frame_labels is None:
                indices = [0, T-1] if reference.shape[1] == 2 else list(range(T))
                frame_labels = torch.tensor(indices).expand(B, -1)
            collected['label_frame_indices'].append(frame_labels.numpy())
            for key in ('input_to_mm', 'frame_indices', 'timestamps_seconds'):
                if key in collected:
                    collected[key].append(batch[key].numpy())
            if not cli.no_visuals:
                import matplotlib.pyplot as plt
                for i, sid in enumerate(batch['patient_name']):
                    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
                    for ax, t, ref_t in zip(axes, (0, T-1), (0, reference.shape[1]-1)):
                        ax.imshow(video[i, t, 0].cpu().numpy(), cmap='gray')
                        xy = predictions[i, t].cpu().numpy()
                        gt = reference[i, ref_t].cpu().numpy()
                        ax.scatter(xy[:, 0], xy[:, 1], c='lime', label='prediction')
                        ax.scatter(gt[:, 0], gt[:, 1], c='red', marker='x', label='reference')
                        ax.set_axis_off()
                    axes[0].legend()
                    fig.savefig(output / f'{sid}.png', dpi=120)
                    plt.close(fig)
    arrays = {key: np.asarray(value) if key == 'patient_name' else np.concatenate(value)
              for key, value in collected.items()}
    errors = arrays['error_mm']
    metrics = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=file_sha(checkpoint_path),
                   epoch=checkpoint['epoch']+1, split=cli.split, prediction_source='detector',
                   data_manifest_sha256=provenance.get('data_manifest_sha256'),
                   reference='software contours; estimated ED/ES' if args.data_type == 'echoxflow' else args.data_type,
                   std_definition='population standard deviation over individual point errors (ddof=0)',
                   point_order=['apex_proxy', 'base_image_left', 'base_image_right'] if args.data_type == 'echoxflow' else None,
                   endpoint=summarize_errors(errors[:, [0, -1]]), all_frames=summarize_errors(errors),
                   per_point={str(i): summarize_errors(errors[:, :, i:i+1]) for i in range(errors.shape[-1])},
                   evaluation_command=sys.argv)
    if errors.shape[1] > 2:
        metrics['middle'] = summarize_errors(errors[:, 1:-1])
    np.savez_compressed(output / 'predictions.npz', **arrays)
    metrics['predictions_sha256'] = file_sha(output / 'predictions.npz')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: metrics[key] for key in ('epoch', 'endpoint', 'middle', 'all_frames') if key in metrics}, indent=2))


if __name__ == '__main__':
    main()
