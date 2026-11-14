"""Authenticate public initialization exports and restore the original model setup."""
from pathlib import Path
import json
import torch

from trustml.convnext_cifar100.model import ConvStem, TransferModel
from trustml.convnext_gtsrb.model import block_name
from trustml.convnext_gtsrb.data import digest
from .io import sha


def load_selected(assets_root, selected_path, expected_hash):
    """Accept the executed checkpoint or the authenticated evaluation-weight export.

    The exported checkpoint has identical model tensors and no optimizer/RNG
    state. Its transport hash differs from the original full checkpoint hash.
    """
    if Path(selected_path).is_file():
        from trustml.ssg_gtsrb.private import load_selected as original_loader
        return original_loader(assets_root,selected_path,expected_hash)
    import timm
    if timm.__version__!='0.9.16':
        raise RuntimeError('architecture lock requires timm==0.9.16')
    repo=Path(__file__).resolve().parents[4]
    manifest=json.loads((repo/'weights/manifest.json').read_text())
    item=next(x for x in manifest['assets'] if x['role']=='public_pretrain')
    path=repo/'weights/files'/item['asset']
    if not path.is_file():
        raise FileNotFoundError('Download the public-pretrain asset from weights-v1 into weights/files; see the repository README, Additional experiments')
    if item['source_full_checkpoint_sha256']!=expected_hash or sha(path)!=item['export_sha256']:
        raise ValueError('public initialization provenance/hash mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['format']!='trustml-gtsrb-eval-weights-v1':
        raise ValueError('unexpected exported weight format')
    torch.manual_seed(42)
    backbone=timm.create_model('convnext_tiny',pretrained=False,drop_path_rate=0.,drop_rate=0.)
    backbone.stem=ConvStem();backbone.reset_classifier(43)
    model=TransferModel(backbone)
    model.load_state_dict(payload['model'],strict=True)
    if not all(torch.equal(v,payload['model'][k]) for k,v in model.state_dict().items()):
        raise RuntimeError('public tensors were not loaded exactly')
    trainable={};blocks={};offset=0
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('model.stages.3.','model.head.','model.norm_pre.')))
        if p.requires_grad:
            block=block_name(name)
            trainable[name]=dict(shape=list(p.shape),numel=p.numel(),block=block)
            blocks.setdefault(block,[]).append([offset,offset+p.numel()]);offset+=p.numel()
    if offset!=15505195:
        raise RuntimeError('GTSRB trainable parameter count changed')
    architecture=dict(trainable_mask_hash=digest(trainable),trainable_tensors=trainable,
        ordered_trainable_tensor_names=list(trainable),trainable_parameter_count=offset,
        blocks=blocks,checkpoint_sha256=expected_hash,public_export_sha256=item['export_sha256'],
        newly_initialized=[],classifier_reinitialized=False,public_pretrain_epoch=50,
        normalization_count=1,timm_version=timm.__version__)
    return model.eval(),architecture


def verify_cifar_initialization(path, expected_source_hash):
    """Accept only the executed checkpoint or its tensor-identical public export."""
    original = "53211c5f45c3a2a9171448fade104c0d3c8fd7649482827063240b3808796262"
    exported = "53f358fe8a0e7b87e0dbbf8b0814d885e6bb82ba5452baa22521a58bf2d68909"
    actual = sha(Path(path))
    if expected_source_hash != original or actual not in (original, exported):
        raise ValueError("frozen CIFAR initialization hash mismatch")
    return dict(source_checkpoint_sha256=original, loaded_file_sha256=actual)
