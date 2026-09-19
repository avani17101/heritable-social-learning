"""Normalized evidence cross entropy; coefficients alter target mixtures."""
import torch


def social_weight(gamma, cue, available):
    q = torch.where(cue, .85, .15)
    return q.pow(4 * gamma) * available


def evidence_loss(logits, private, feedback, social, cue, available, parameters):
    alpha, beta, gamma = parameters.unbind(-1)
    weight = beta * social_weight(gamma, cue, available)
    logp = logits.log_softmax(-1)
    def nll(labels):
        return -logp.gather(-1, labels[...,None]).squeeze(-1)
    return (alpha * (nll(private) + nll(feedback)) + weight * nll(social)) / (2*alpha + weight)
