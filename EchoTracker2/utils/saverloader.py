import torch
import os, pathlib
import numpy as np
import random


ECHO_RELEASE_EXTRA_KEYS = {
    f"resnet_torch.block_groups.{i}.blocks.1.proj_conv.weight" for i in range(4)
}


def load_echo_checkpoint(path, model):
    """严格加载 EchoTracker2，只容许已核实的发布权重额外键。"""
    path = pathlib.Path(path)
    if path.is_dir():
        candidates = list(path.glob("model-*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No model checkpoint in {path}")
        path = max(candidates, key=lambda p: int(p.stem.split("-")[-1]))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
    extra = set(state) - set(model.state_dict())
    if extra and extra != ECHO_RELEASE_EXTRA_KEYS:
        raise ValueError(f"Unexpected EchoTracker2 checkpoint keys: {sorted(extra)}")
    model.load_state_dict({k: v for k, v in state.items() if k not in extra}, strict=True)
    print(f"Loaded {path}; explicitly ignored release keys: {sorted(extra)}", flush=True)
    return checkpoint


def save_echo_checkpoint(path, model, optimizer, scheduler, *, epoch, global_step,
                         best_score, metadata, loader_generator):
    """保存可从下一轮继续的训练状态；文件原子替换。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch, "global_step": global_step, "best_score": list(best_score),
        "metadata": metadata,
        "random_state": random.getstate(),
        "numpy_state": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch_state": torch.get_rng_state(),
        "cuda_states": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "loader_state": loader_generator.get_state() if loader_generator is not None else None,
    }
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_echo_training(checkpoint, optimizer, scheduler, loader_generator):
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    random.setstate(checkpoint["random_state"])
    s = checkpoint["numpy_state"]
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), *s[2:]))
    torch.set_rng_state(checkpoint["torch_state"])
    if checkpoint["cuda_states"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_states"])
    if loader_generator is not None:
        loader_generator.set_state(checkpoint["loader_state"])

def save(ckpt_dir, optimizer, model, global_step, scheduler=None, model_ema=None, keep_latest=1, model_name='model', save_ts=False):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    prev_ckpts = list(pathlib.Path(ckpt_dir).glob('%s-*' % model_name))
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_ckpts) > keep_latest-1:
        for f in prev_ckpts[keep_latest-1:]:
            f.unlink()
    model_path = '%s/%s-%09d.pth' % (ckpt_dir, model_name, global_step)
    
    ckpt = {'optimizer_state_dict': optimizer.state_dict()}
    ckpt['model_state_dict'] = model.state_dict()
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if model_ema is not None:
        ckpt['ema_model_state_dict'] = model_ema.state_dict()
    torch.save(ckpt, model_path)
    if save_ts:
        torch.jit.script(model).save('%s/torchscript.pth' % (ckpt_dir))
        print("TorchScript model is saved: %s/torchscript.pth"% (ckpt_dir))
        #torch.jit.script(model).save('torchscript.pth')
    print("saved a checkpoint: %s" % (model_path))

def load(ckpt_dir, model, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='model', ignore_load=None):
    print('reading ckpt from %s' % ckpt_dir)
    if not os.path.exists(ckpt_dir):
        print('...there is no full checkpoint here!')
        print('-- note this function no longer appends "saved_checkpoints/" before the ckpt_dir --')
    else:
        ckpt_names = [i for i in os.listdir(ckpt_dir)
                      if i.startswith('%s-' % model_name) and i.endswith('.pth')]
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            if step==0:
                step = max(steps)
            model_name = '%s-%01d.pth' % (model_name, step)
            path = os.path.join(ckpt_dir, model_name)
            print('...found checkpoint %s'%(path))

            if ignore_load is not None:
                
                print('ignoring', ignore_load)
                checkpoint = torch.load(path)['model_state_dict']

                model_dict = model.state_dict()

                # 1. filter out ignored keys
                pretrained_dict = {k: v for k, v in checkpoint.items()}
                for ign in ignore_load:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not ign in k}
                    
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict, strict=False)
            else:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if model_ema is not None:
                model_ema.load_state_dict(checkpoint['ema_model_state_dict']) 
        else:
            print('...there is no full checkpoint here!')
    return step
