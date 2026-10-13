import os
import torch
import torch.nn as nn


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _candidate_pretrain_dirs():
    env_dir = os.environ.get('SEMI_ECHO_PRETRAIN_DIR')
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    candidates.extend([
        os.path.abspath(os.path.join(_THIS_DIR, '..', '..', 'pretraining')),
        os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', 'pretraining')),
        os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..', 'pretraining')),
    ])
    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            yield path


def _resolve_pretrain_dir():
    for path in _candidate_pretrain_dirs():
        if os.path.isdir(path):
            return path
    checked = '\n'.join(_candidate_pretrain_dirs())
    raise FileNotFoundError(
        'Cannot find DINOv2 pretraining directory. Set SEMI_ECHO_PRETRAIN_DIR. '
        f'Checked:\n{checked}'
    )


def get_dinov2_model(source='guided', hub_dir=None, checkpoint=None, device='cpu'):
    if source not in ('guided', 'public'):
        raise ValueError(f'Unsupported DINOv2 source: {source}')
    if source == 'public' and (hub_dir is None or checkpoint is None):
        raise ValueError('Public DINOv2 requires explicit hub_dir and checkpoint')
    hub_dir = hub_dir or _resolve_pretrain_dir()
    ckpt_path = checkpoint or os.path.join(hub_dir, 'teacher_checkpoint.pth')

    if not os.path.isdir(hub_dir):
        raise FileNotFoundError(f'DINOv2 hub directory not found: {hub_dir}')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'DINOv2 teacher checkpoint not found: {ckpt_path}')

    model = torch.hub.load(hub_dir, 'dinov2_vits14', source='local', pretrained=False)
    pretrained = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if source == 'guided':
        state = {key.removeprefix('backbone.'): value for key, value in pretrained['teacher'].items()
                 if not key.startswith(('dino_head.', 'ibot_head.'))}
        model.pos_embed = nn.Parameter(torch.zeros_like(state['pos_embed']))
    else:
        state = pretrained
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    print(f'DINOv2 ({source}) loaded strictly from {ckpt_path}')
    return model.to(device)
