"""Existing project's detached-target TRADES attack, without pretrain package imports.

Semantics mirror trustml.pretrain.attacks.trades_adversarial (uniform start).
"""
import torch
import torch.nn.functional as F


def trades_adversarial(model,pixels,clean_probs,*,epsilon,step_size,steps,generator):
    original=pixels.detach()
    was_training=model.training
    model.eval()
    try:
        uniform=torch.rand(original.shape,dtype=original.dtype,device=original.device,generator=generator)
        adversarial=(original+(2*uniform-1)*epsilon).clamp(0,1).detach()
        for _ in range(steps):
            adversarial.requires_grad_(True)
            loss=F.kl_div(model(adversarial).log_softmax(-1),clean_probs.detach(),reduction="batchmean")
            gradient=torch.autograd.grad(loss,adversarial,only_inputs=True)[0]
            adversarial=adversarial.detach()+step_size*gradient.sign()
            adversarial=(original+(adversarial-original).clamp(-epsilon,epsilon)).clamp(0,1).detach()
    finally:
        model.train(was_training)
    return adversarial.detach()
