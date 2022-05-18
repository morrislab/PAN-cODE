import numpy as np
import torch


def log_normal_pdf(x, mean, logvar):
    """Compute log pdf of data under gaussian specified by parameters.
    Implementation taken from: https://github.com/rtqichen/torchdiffeq.
    Args:
        x (torch.Tensor): Observed data points.
        mean (torch.Tensor): Mean of gaussian distribution.
        logvar (torch.Tensor): Log variance of gaussian distribution.
    Returns:
        torch.Tensor: Log probability of data under specified gaussian.
    """
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)

    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    """Compute analytic KL divergence between two gaussian distributions.
    Computes analytic KL divergence between two multivariate gaussians which
    are parameterized by the given mean and variances. All inputs must have
    the same dimension.
    Implementation taken from: https://github.com/rtqichen/torchdiffeq.
    Args:
        mu1 (torch.Tensor): Mean of first gaussian distribution.
        lv1 (torch.Tensor): Log variance of first gaussian distribution.
        mu2 (torch.Tensor): Mean of second gaussian distribution.
        lv2 (torch.Tensor): Log variance of second gaussian distribution.
    Returns:
        torch.Tensor: Analytic KL divergence between given distributions.
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def get_analytic_elbo(x, pred_x, qz0_mean, qz0_logvar, noise_std=0.1, kl_weight=1., stl=False):
    """Compute the ELBO.

    Computes the evidence lower bound (ELBO) for a given prediction,
    ground truth, and latent initial state.

    Supports KL annealing, where the KL term can gradually be increased
    during training, as described in: https://arxiv.org/abs/1903.10145.

    Supports the Sticking The Landing gradient estimator for lower
    variance, as described in: https://arxiv.org/abs/1703.09194.

    Args:
        x (torch.Tensor): Input data.
        pred_x (torch.Tensor): Data reconstructed by latent NODE.
        qz0_mean (torch.Tensor): Latent initial state means.
        qz0_logvar (torch.Tensor): Latent initial state variances.
        noise_std (float, optional): Variance of gaussian pdf.
        kl_weight (float, optional): Weight for KL term.
        stl (boolean): Uses Sticking-The-Landing estimator.

    Returns:
        torch.Tensor: ELBO score.
    """
    noise_std_ = torch.zeros(pred_x.size(), device=x.device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_)

    logpx = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)

    if stl:
        qz0_mean = qz0_mean.detach()
        qz0_logvar = qz0_logvar.detach()

    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size(), device=x.device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)

    return torch.mean(-logpx + kl_weight * analytic_kl, dim=0)


def get_iwae_elbo(x, pred_x, z0, qz0_mean, qz0_logvar, noise_std=1., transpose=True, stl=False):
    """Compute the IWAE ELBO.

    Computes the Importance weighted ELBO as described in: https://arxiv.org/abs/1509.00519.

    Supports the Sticking The Landing gradient estimator for lower
    variance, as described in: https://arxiv.org/abs/1703.09194.

    Args:
        x (torch.Tensor): Input data. (batch_size, t, 2)
        pred_x (torch.Tensor): Data reconstructed by latent NODE. (N_trajectories, batch_size, t, 2) if transpose
        z0 (torch.Tensor): (batch_size, N_trajectories, latent_dim)
        qz0_mean (torch.Tensor): Latent initial state means. (batch_size, N_trajectories, latent_dim) or (batch_size, latent_dim)
        qz0_logvar (torch.Tensor): Latent initial state variances. (batch_size, N_trajectories, latent_dim) or (batch_size, latent_dim)
        noise_std (float, optional): Variance of gaussian pdf.
        transpose (boolean): Transposes input prior to processing.
        stl (boolean): Uses Sticking-The-Landing estimator.

    Returns:
        torch.Tensor: IW ELBO.
    """
    if transpose:
        pred_x = pred_x.transpose(0, 1)
    if len(x.size()) == 3:
        batch_size = x.shape[0]
        x = torch.repeat_interleave(x, z0.shape[1], 0)
        x = x.view(batch_size, z0.shape[1], *x.shape[1:])

    if qz0_mean.ndim == 2: # (batch_size, latent_dim)
        qz0_mean = qz0_mean.unsqueeze(0).transpose(0, 1).repeat(1, z0.shape[1], 1) # (batch_size, N_trajectories, latent_dim)
    if qz0_logvar.ndim == 2:
        qz0_logvar = qz0_logvar.unsqueeze(0).transpose(0, 1).repeat(1, z0.shape[1], 1)

    noise_std_ = torch.zeros(x.size()).to(x.device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_)

    zero_mean = zero_logvar = torch.zeros_like(z0, device=x.device)

    log_pxCz0 = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)
    log_pz0 = log_normal_pdf(z0, zero_mean, zero_logvar).sum(-1)

    if stl:
        log_qz0Cx = log_normal_pdf(z0, qz0_mean.detach(), qz0_logvar.detach()).sum(-1)
    else:
        log_qz0Cx = log_normal_pdf(z0, qz0_mean, qz0_logvar).sum(-1)

    unnorm_weight = log_pxCz0 + log_pz0 - log_qz0Cx

    unnorm_weight_detach = unnorm_weight.detach()
    total_weight = torch.logsumexp(unnorm_weight_detach, -1).unsqueeze(-1)
    log_norm_weight = unnorm_weight_detach - total_weight
    iw_elbo = -torch.mean(torch.sum(torch.exp(log_norm_weight) * unnorm_weight, -1))
    return iw_elbo
