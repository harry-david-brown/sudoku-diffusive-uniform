"""GIDD on Sudoku: masked / hybrid discrete diffusion.

A port of von Rutte et al., "Generalized Interpolating Discrete Diffusion" (ICML 2025, arXiv 2503.04482),
following the reference code at github.com/dvruette/gidd (loss.py, diffusion_process.py, sampling.py,
eval/self_correction.py).

  - one output head: x_theta, a distribution over the clean digit
  - hybrid noise: mostly MASK, with a share p_u of random digits (Eq. 17-18); p_u = 0 is masked diffusion
  - the noise level t is an input to the network
  - loss: the GIDD ELBO, w_t * (KL + IS), with the paper's reweighting
  - samplers: confidence-ordered unmasking, ancestral sampling, self-correction

Sudoku-specific choices (not from the paper): given cells are never noised, never in the loss and never
changed; the network is told which cells are givens; the ancestral sampler ends with a final denoise.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK = 10          # token id of MASK (digits are 1..9, 0 is unused)
N_DIGITS = 9       # N in the paper: number of non-mask tokens
N_STATES = 10      # state index 0..8 = digit 1..9, state 9 = MASK
T_EPS = 1e-4


# ── Noise schedule (paper Sec. 4, Eq. 17-18; reference HybridDiffusion) ────────
class HybridSchedule:
    """q_t(z|x) = alpha_t * x + beta_t*pi_t, with
       alpha_t = (1-t)/C_t,  beta_t*pi_t = (t*m + c_t*u)/C_t,
       c_t = B * t^(g/2) * (1-t)^(g/2),  C_t = 1 + c_t,  B = 2^g * p_u/(1-p_u).
       p_u is the share of random digits at t = 1/2. p_u = 0 is masked diffusion."""

    def __init__(self, p_uniform=0.0, gamma=1.0):
        self.p_uniform = max(math.exp(-20), p_uniform)   # same floor as the reference
        self.gamma = gamma
        self.B = 2 ** gamma * self.p_uniform / (1 - self.p_uniform)
        self.mask_vec = torch.zeros(N_STATES); self.mask_vec[N_DIGITS] = 1
        self.unif_vec = (1 - self.mask_vec) / N_DIGITS

    def c(self, t):
        return self.B * t.pow(self.gamma / 2) * (1 - t).pow(self.gamma / 2)

    def alpha_betapi(self, t):
        """t: [B] -> alpha [B,1], beta_pi [B,N_STATES]"""
        t = t[:, None]
        c = self.c(t); C = 1 + c
        alpha = (1 - t) / C
        beta_pi = (t * self.mask_vec.to(t) + c * self.unif_vec.to(t)) / C
        return alpha, beta_pi

    def probs_at_t(self, prs, t):
        """prs: [B,81,N_STATES] distribution over the clean token -> q_t(.|prs)"""
        alpha, beta_pi = self.alpha_betapi(t)
        return prs * alpha[:, :, None] + beta_pi[:, None, :]

    def sample_zt(self, x_state, t):
        """x_state: [B,81] clean state indices -> noisy state indices"""
        probs = self.probs_at_t(F.one_hot(x_state, N_STATES).to(t.dtype), t)
        return torch.multinomial(probs.reshape(-1, N_STATES), 1).reshape(x_state.shape)

    def weights(self, t, z_state, x_state, weighting='dynamic', max_w=2.0):
        """ELBO weight w_t(z_t, x) (Eq. 13, closed form from App. H.8 / reference
        get_weights) and the training weight (Sec. 4.1)."""
        t = t[:, None].float()                      # float32: MPS has no float64
        g, B = self.gamma, self.B
        c = B * t.pow(g / 2) * (1 - t).pow(g / 2)
        c_prime = (g / 2) * (1 - 2 * t) / (t * (1 - t)) * c
        is_mask = (z_state == N_DIGITS).to(t.dtype)
        is_x = (z_state == x_state).to(t.dtype)
        w_x = (c + (1 - t) * c_prime) / N_DIGITS / ((1 - t) * (1 - t + c / N_DIGITS))
        w_u = (c + (1 - t) * c_prime) / ((1 - t) * c)
        w_m = 1 / ((1 - t) * t)
        elbo_w = is_x * w_x + is_mask * w_m + (1 - is_x - is_mask) * w_u
        if weighting == 'elbo':
            train_w = elbo_w.clamp(max=1e4)
        elif weighting == 'clip':
            train_w = elbo_w.clamp(0.0, 1.0)                       # Eq. 19, w_max = 1
        else:                                                      # Eq. 20: 2 / 1 / (B/N) e^{-lambda/2}
            alpha = (1 - t) / (1 + c)
            x_scale = (B / N_DIGITS) * ((1 - alpha) / alpha).sqrt()
            train_w = ((1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale).clamp(0.0, max_w)
        return elbo_w.float(), train_w.float()


def tokens_to_state(tok):        # digit d -> d-1, MASK -> 9
    return torch.where(tok == MASK, torch.full_like(tok, N_DIGITS), tok - 1)

def state_to_tokens(st):
    return torch.where(st == N_DIGITS, torch.full_like(st, MASK), st + 1)


# ── Model ──────────────────────────────────────────────────────────────────────
class SudokuGIDD(nn.Module):
    """Same encoder as the masked/two-head models (4 layers, d=128, 4 heads), plus
    a noise-level embedding and a given-cell flag. One head: logits over digits 1-9."""

    def __init__(self, embed_dim=128, num_heads=4, num_layers=4, seq_len=81):
        super().__init__()
        self.embedding = nn.Embedding(11, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.given_embedding = nn.Embedding(2, embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, N_DIGITS)
        self.embed_dim, self.seq_len = embed_dim, seq_len

    def time_embedding(self, t):
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * 1000 * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, z_tok, t, given):
        pos = torch.arange(self.seq_len, device=z_tok.device).unsqueeze(0)
        h = self.embedding(z_tok) + self.pos_embedding(pos) + self.given_embedding(given.long())
        h = h + self.time_mlp(self.time_embedding(t))[:, None, :]
        return self.output(self.transformer(h))          # [B,81,9]


# ── Loss (Theorem 3.7; reference GiddLoss) ─────────────────────────────────────
def gidd_loss(logits, x_state, z_state, t, unknown, sched, weighting='dynamic'):
    """Returns (training loss, ELBO, KL part, IS part), each averaged over unknown cells."""
    x_hat = F.pad(logits.softmax(-1), (0, 1))                         # zero mass on MASK
    x = F.one_hot(x_state, N_STATES).to(x_hat.dtype)
    log_q = sched.probs_at_t(x, t).log().clamp(min=-1e6)              # true marginal
    log_p = sched.probs_at_t(x_hat, t).log().clamp(min=-1e6)          # model marginal
    kl = (log_q.exp() * (log_q - log_p)).sum(-1)
    log_ratio = (log_q - log_p).gather(-1, z_state.unsqueeze(-1)).squeeze(-1)
    isd = log_ratio.exp() - log_ratio - 1
    elbo_w, train_w = sched.weights(t, z_state, x_state, weighting)
    m = unknown.float(); denom = m.sum().clamp(min=1)
    loss = (train_w * (kl + isd) * m).sum() / denom
    elbo = (elbo_w * (kl + isd) * m).sum() / denom
    return loss, elbo.detach(), (train_w * kl * m).sum().detach() / denom, (train_w * isd * m).sum().detach() / denom


def sample_t(batch_size, device):
    """Low-discrepancy t ~ U(eps, 1-eps), as in the reference."""
    t = (torch.arange(batch_size, device=device) / batch_size + torch.rand(1, device=device)).fmod(1.0)
    return (1 - 2 * T_EPS) * t + T_EPS


def make_noisy(puzzles, solutions, t, sched):
    """Noise the unknown cells only. Returns z tokens, z states, x states, unknown mask."""
    unknown = puzzles == 0
    x_state = solutions - 1
    z_state = torch.where(unknown, sched.sample_zt(x_state, t), x_state)
    return state_to_tokens(z_state), z_state, x_state, unknown


# ── Samplers ───────────────────────────────────────────────────────────────────
def _start(puzzles):
    given = puzzles != 0
    return torch.where(given, puzzles, torch.full_like(puzzles, MASK)), given


@torch.no_grad()
def sample_confidence(model, puzzles, k=5, margin=False):
    """The masked-diffusion sampler from Sudoku-Diffusive-Masked: commit the k most
    confident masked cells per pass. Cannot revise. Used for the p_u = 0 check."""
    z, given = _start(puzzles)
    n_unknown = (~given).sum(1).clamp(min=1).float()
    while (z == MASK).any():
        masked = z == MASK
        t = (masked.sum(1).float() / n_unknown).clamp(T_EPS, 1 - T_EPS)
        probs = model(z, t, given).softmax(-1)
        conf, pred = probs.max(-1)
        if margin:                      # Kim et al. 2025, "top probability margin": top-1 minus top-2
            top2 = probs.topk(2, dim=-1).values
            conf = top2[..., 0] - top2[..., 1]
        conf = conf.masked_fill(~masked, -1.0)
        top = conf.topk(min(k, 81), dim=1)
        take = torch.zeros_like(masked).scatter(1, top.indices, top.values >= 0)
        z = torch.where(take, pred + 1, z)
    return z


@torch.no_grad()
def sample_ancestral(model, puzzles, sched, steps=64):
    """Ancestral sampling (Eq. 16; reference GiddSampler). Every unknown cell is
    resampled at every step, so with p_u > 0 a filled cell can change."""
    z, given = _start(puzzles)
    B = puzzles.shape[0]
    ts = (1 - 2 * T_EPS) * torch.linspace(0, 1, steps + 1, device=puzzles.device) + T_EPS
    for i in range(steps, 0, -1):
        t, s = ts[i].expand(B), ts[i - 1].expand(B)
        x_hat = F.pad(model(z, t, given).softmax(-1), (0, 1))
        z_state = tokens_to_state(z)
        q_s, q_t = sched.probs_at_t(x_hat, s), sched.probs_at_t(x_hat, t)
        q_zt = q_t.gather(-1, z_state.unsqueeze(-1))
        a_t, bp_t = sched.alpha_betapi(t); a_s, bp_s = sched.alpha_betapi(s)
        a_ts = (a_t / a_s)[:, :, None]
        bp_ts_at_zt = (bp_t - (a_t / a_s) * bp_s)[:, None, :].expand(-1, 81, -1).gather(-1, z_state.unsqueeze(-1))
        q_ts = a_ts * F.one_hot(z_state, N_STATES) + bp_ts_at_zt        # q_{t|s}(z_t | z_s), as a function of z_s
        q_st = (q_ts * q_s / q_zt).clamp(min=0)
        new = torch.multinomial(q_st.reshape(-1, N_STATES) + 1e-12, 1).reshape(z.shape)
        z = torch.where(given, z, state_to_tokens(new))
    # Final denoise (not in the reference): at t = eps the chain still carries a little
    # noise (c_t ~ sqrt(t), about 0.5% random digits at p_u = 0.2). Harmless for text,
    # fatal for Sudoku, so read off the model's clean prediction for every unknown cell.
    pred = model(z, ts[0].expand(B), given).argmax(-1) + 1
    return torch.where(given, z, pred)


@torch.no_grad()
def self_correct(model, z, given, temp=0.1, t0=0.01, max_steps=128, patience=32):
    """Self-correction step (Sec. 3.4, App. C Algorithm 1; reference correction_step).
    Resample every unknown cell, commit the single most confident change, repeat."""
    B = z.shape[0]
    t = torch.full((B,), t0, device=z.device)
    best = torch.zeros(B, device=z.device); stale = torch.zeros(B, device=z.device)
    active = torch.ones(B, dtype=torch.bool, device=z.device)
    for _ in range(max_steps):
        logits = model(z, t, given)
        p = (logits / temp).softmax(-1)
        prop = torch.multinomial(p.reshape(-1, N_DIGITS), 1).reshape(z.shape) + 1
        score = ((prop != z) & ~given).float() * p.gather(-1, (prop - 1).unsqueeze(-1)).squeeze(-1)
        val, idx = score.max(1)
        change = active & (val > 0)
        z_new = z.scatter(1, idx[:, None], prop.gather(1, idx[:, None]))
        z = torch.where(change[:, None], z_new, z)
        acc = (z == logits.argmax(-1) + 1).float().mean(1)             # self-accuracy
        improved = acc > best
        best = torch.where(improved, acc, best)
        stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
        active = active & change & (stale <= patience)
        if not active.any():
            break
    return z
