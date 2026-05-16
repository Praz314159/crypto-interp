"""Ablation experiments for embedding analysis.

Validate that the model is actually using the key multiplicative-Fourier
components by:
  1. Projecting W_E onto the multiplicative basis.
  2. Keeping ONLY the top-K key components (zeroing the rest); re-evaluate loss.
     Expectation: loss stays near zero. → embedding really uses these components.
  3. Keeping EVERYTHING EXCEPT those top-K components; re-evaluate loss.
     Expectation: loss crashes to chance or worse. → those components are
     necessary.
"""

import copy

import torch
import torch.nn.functional as F


def project_back(W_E_values: torch.Tensor, basis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Project W_E onto the basis, keep only basis indices where mask is True, reconstruct.

    Args:
        W_E_values: (d_model, p) tensor.
        basis: (p, p) orthonormal basis where each ROW is a basis vector.
        mask: (p,) boolean — True for basis indices to keep.

    Returns:
        W_E_filtered of shape (d_model, p) — the reconstruction using only the kept basis rows.
    """
    # coef[k, d] = sum_a basis[k, a] * W_E[d, a]
    coef = torch.einsum("kp,dp->kd", basis, W_E_values.to(basis.dtype))
    coef_filtered = coef * mask[:, None].to(coef.dtype)
    # reconstruct: W_E_new[d, a] = sum_k basis[k, a] * coef_filtered[k, d]
    return torch.einsum("kp,kd->dp", basis, coef_filtered).to(W_E_values.dtype)


@torch.no_grad()
def evaluate_loss(model, ds) -> tuple[float, float, float]:
    """Compute (train_loss, test_loss, total_accuracy) for the current model."""
    device = next(model.parameters()).device
    inputs = ds.inputs.to(device)
    labels = ds.labels.to(device)
    logits = model(inputs)[:, -1, : ds.n_answer_tokens]
    logits_64 = logits.to(torch.float64)
    logprobs = F.log_softmax(logits_64, dim=-1)
    sel = torch.gather(logprobs, index=labels[:, None], dim=-1).squeeze(-1)
    losses = -sel
    train_loss = float(losses[ds.train_mask].mean())
    test_loss = float(losses[ds.test_mask].mean())
    acc = float((logits.argmax(-1) == labels).float().mean())
    return train_loss, test_loss, acc


def ablate_embedding(model, basis: torch.Tensor, keep_mask: torch.Tensor):
    """Return a model copy whose W_E (restricted to value tokens) is reconstructed
    from only the basis indices where keep_mask is True. The '=' token's embedding
    is left untouched.
    """
    new_model = copy.deepcopy(model)
    p = basis.shape[0]
    with torch.no_grad():
        W_E = new_model.embed.W_E  # (d_model, vocab)
        W_E_values = W_E[:, :p]
        W_E_new = project_back(W_E_values, basis, keep_mask)
        W_E[:, :p] = W_E_new.to(W_E.dtype)
    return new_model
