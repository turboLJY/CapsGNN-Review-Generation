import torch
from torch.nn import functional


def masked_cross_entropy(prediction, target, mask):
    """
    Returns:
        loss: An average loss value masked by the length.
    """

    # (batch * max_len, num_classes)
    prediction_flat = prediction.view(-1, prediction.size(-1))

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)

    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(prediction_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # mask: (batch, max_len)
    loss = losses.masked_select(mask).mean()

    return loss


def margin_loss(scores, target, mask, loss_lambda=0.5):
    """
    :param scores: aspect capsules, [batch_size, seq_len, n_targets, capsule_size]
    :param target: ground-truth, [batch_size, seq_len]
    :param mask: ground-truth mask, [batch_size, seq_len]
    :param loss_lambda: Regularization parameter.
    :return L_c: Classification loss.
    """
    scores = scores.view(-1, scores.size(2), scores.size(3))

    v_mag = torch.sqrt((scores**2).sum(dim=2, keepdim=True))  # [batch_size, seq_len, n_targets, 1]

    zero = torch.zeros(1).to(v_mag.device)
    m_plus = 0.9
    m_minus = 0.1
    max_l = torch.max(m_plus - v_mag, zero).view(scores.size(0), -1)**2
    max_r = torch.max(v_mag - m_minus, zero).view(scores.size(0), -1)**2

    T_c = target.view(-1, 1)
    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1).view(*target.size())
    L_c = L_c.masked_select(mask).mean()
    return L_c
