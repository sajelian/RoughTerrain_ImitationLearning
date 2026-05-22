"""Flow-Q-Learning agent.

Three networks:
  * π_β  (FlowMLP)            — behavior flow, trained with flow-matching loss
  * π_φ  (OneStepFlowPolicy)  — one-step deploy policy, distilled from π_β + Q
  * Q1, Q2 (QNetwork)         — clipped-double-Q with target networks

Each `update(batch)` runs one gradient step on each network and one soft-target sync.
Reference: Park et al., "Flow Q-Learning" (https://arxiv.org/abs/2502.02538).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .dataset import FQLBatch
from .networks import FlowMLP, OneStepFlowPolicy, QNetwork


@dataclass
class FQLConfig:
    obs_dim: int
    action_dim: int

    # Network sizes (paper Table 5: [512,512,512,512] + GELU)
    flow_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    onestep_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    q_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    activation: str = "gelu"
    time_embed_dim: int = 64

    # Optimisers
    flow_lr: float = 3e-4
    onestep_lr: float = 3e-4
    q_lr: float = 3e-4

    # Q-learning
    gamma: float = 0.99
    target_tau: float = 0.005
    grad_clip: float = 1.0

    # One-step distillation (paper Eq. 9: L_π = -Q + alpha * ||μ_ω - μ_θ||²)
    flow_integration_steps: int = 10        # Euler steps when generating BC targets
    alpha: float = 3.0                      # BC coefficient — paper's recommended range
                                            # with Q normalization: {0.03, 0.1, 0.3, 1, 3, 10}
    q_weight_normalize: bool = True         # paper *recommends* this for new tasks (App. B)

    # Double-Q aggregation. Paper: mean by default; min for adroit / antmaze-{large,giant}.
    # box_up (humanoid + manipulation) is closer to the default → "mean".
    q_aggregator: str = "mean"              # "mean" or "min"
    layer_norm_q: bool = True               # paper applies LN to value nets

    # Stability
    target_q_clip: float | None = None      # optional clip on Q target (None = off)

    device: str = "cuda"


class FQLAgent:
    def __init__(self, cfg: FQLConfig):
        self.cfg = cfg
        d = cfg.device

        self.flow = FlowMLP(
            obs_dim=cfg.obs_dim, action_dim=cfg.action_dim,
            hidden_dims=list(cfg.flow_hidden_dims), time_embed_dim=cfg.time_embed_dim,
            activation=cfg.activation,
        ).to(d)
        self.onestep = OneStepFlowPolicy(
            obs_dim=cfg.obs_dim, action_dim=cfg.action_dim,
            hidden_dims=list(cfg.onestep_hidden_dims), activation=cfg.activation,
        ).to(d)
        self.q1 = QNetwork(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim,
                           hidden_dims=list(cfg.q_hidden_dims), activation=cfg.activation,
                           layer_norm=cfg.layer_norm_q).to(d)
        self.q2 = QNetwork(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim,
                           hidden_dims=list(cfg.q_hidden_dims), activation=cfg.activation,
                           layer_norm=cfg.layer_norm_q).to(d)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

        self.flow_opt = torch.optim.Adam(self.flow.parameters(), lr=cfg.flow_lr)
        self.onestep_opt = torch.optim.Adam(self.onestep.parameters(), lr=cfg.onestep_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.q_lr,
        )

        # EMA of |advantage| for q_weight normalisation
        self._adv_abs_ema = 1.0

    # --- losses ---

    def flow_loss(self, batch: FQLBatch) -> torch.Tensor:
        """Linear-interpolant flow-matching loss.

        a_t = (1 - t) * z + t * a   →   v_target = a - z   (constant along the path)
        """
        s, a = batch.obs, batch.action
        B = a.shape[0]
        z = torch.randn_like(a)
        t = torch.rand(B, 1, device=a.device)
        a_t = (1.0 - t) * z + t * a
        v_pred = self.flow(a_t, t.squeeze(-1), s)
        return F.mse_loss(v_pred, a - z)

    def _agg(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Aggregate the double-Q values. Paper default: mean; 'min' for harder tasks."""
        if self.cfg.q_aggregator == "min":
            return torch.min(q1, q2)
        return 0.5 * (q1 + q2)

    def q_loss(self, batch: FQLBatch) -> tuple[torch.Tensor, dict]:
        s, a, r, s_next, d = batch.obs, batch.action, batch.reward, batch.next_obs, batch.done
        with torch.no_grad():
            z_next = torch.randn(s_next.shape[0], self.cfg.action_dim, device=s_next.device)
            a_next = self.onestep(s_next, z_next)
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_next = self._agg(q1_next, q2_next)
            target = r + self.cfg.gamma * (1.0 - d) * q_next
            if self.cfg.target_q_clip is not None:
                target = target.clamp(-self.cfg.target_q_clip, self.cfg.target_q_clip)

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        return loss, {
            "q_target_mean": target.mean().item(),
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
        }

    def onestep_loss(self, batch: FQLBatch) -> tuple[torch.Tensor, dict]:
        """One-step policy loss (paper Eq. 9):  L_π = -Q(s, μ_ω(s,z)) + α * ||μ_ω(s,z) - μ_θ(s,z)||²

        μ_ω: one-step policy; μ_θ: full flow ODE; same z for both (so the BC term is an
        upper bound on the 2-Wasserstein distance between the two push-forwards).
        """
        s = batch.obs
        B = s.shape[0]
        z = torch.randn(B, self.cfg.action_dim, device=s.device)

        # BC target: integrate the flow under the *same* z
        a_target = self.flow.integrate(s, z, num_steps=self.cfg.flow_integration_steps)

        # One-step prediction
        a_phi = self.onestep(s, z)
        bc = F.mse_loss(a_phi, a_target)

        # Q-improvement signal (gradient flows through a_phi). Paper Eq. 9 uses
        # mean(Q1, Q2) for the actor objective (App. B, "Value learning").
        q_phi = self._agg(self.q1(s, a_phi), self.q2(s, a_phi))
        if self.cfg.q_weight_normalize:
            # Paper Appendix B: lam = stop_grad(1 / mean(|q|)).  Batch-wise, not EMA.
            lam = (1.0 / q_phi.detach().abs().mean().clamp(min=1e-6))
            policy_loss = -lam * q_phi.mean()
        else:
            policy_loss = -q_phi.mean()

        loss = policy_loss + self.cfg.alpha * bc
        return loss, {
            "bc_loss": bc.item(),
            "policy_q": q_phi.mean().item(),
        }

    # --- update step ---

    def soft_update_targets(self):
        tau = self.cfg.target_tau
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
                pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
            for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
                pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def update(self, batch: FQLBatch) -> dict:
        # 1. Flow (BC for π_β)
        fl = self.flow_loss(batch)
        self.flow_opt.zero_grad(set_to_none=True)
        fl.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.cfg.grad_clip)
        self.flow_opt.step()

        # 2. Q (TD with double-Q target)
        ql, q_info = self.q_loss(batch)
        self.q_opt.zero_grad(set_to_none=True)
        ql.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.cfg.grad_clip,
        )
        self.q_opt.step()

        # 3. One-step (BC distillation from flow + Q-improvement)
        ol, o_info = self.onestep_loss(batch)
        self.onestep_opt.zero_grad(set_to_none=True)
        ol.backward()
        torch.nn.utils.clip_grad_norm_(self.onestep.parameters(), self.cfg.grad_clip)
        self.onestep_opt.step()

        # 4. Target soft-update
        self.soft_update_targets()

        return {"flow_loss": fl.item(), "q_loss": ql.item(), "onestep_loss": ol.item(), **q_info, **o_info}

    # --- inference ---

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic_z: bool = False) -> torch.Tensor:
        """Sample an action from the one-step policy."""
        if deterministic_z:
            z = torch.zeros(obs.shape[0], self.cfg.action_dim, device=obs.device)
        else:
            z = torch.randn(obs.shape[0], self.cfg.action_dim, device=obs.device)
        return self.onestep(obs, z)

    @torch.no_grad()
    def act_flow(self, obs: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """Sample from the iterative flow (slower; useful for comparison/eval)."""
        steps = num_steps or self.cfg.flow_integration_steps
        z = torch.randn(obs.shape[0], self.cfg.action_dim, device=obs.device)
        return self.flow.integrate(obs, z, num_steps=steps)

    # --- checkpointing ---

    def state_dict(self) -> dict:
        return {
            "flow": self.flow.state_dict(),
            "onestep": self.onestep.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "flow_opt": self.flow_opt.state_dict(),
            "onestep_opt": self.onestep_opt.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "_adv_abs_ema": self._adv_abs_ema,
        }

    def load_state_dict(self, state: dict):
        self.flow.load_state_dict(state["flow"])
        self.onestep.load_state_dict(state["onestep"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        self.flow_opt.load_state_dict(state["flow_opt"])
        self.onestep_opt.load_state_dict(state["onestep_opt"])
        self.q_opt.load_state_dict(state["q_opt"])
        self._adv_abs_ema = state.get("_adv_abs_ema", 1.0)
