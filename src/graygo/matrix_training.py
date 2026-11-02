"""Real-trainer PPO matrix adapter and reproducible A2 preflight.

Each record is one learner side of a one-round game. Opponents are selected
before either action is drawn; historical actions never enter learner updates.
For this role-only adapter, two probability vectors are an exact frozen policy.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .manifest import make_manifest, source_digest, write_report
from .models import masked_log_probs
from .ppo import PPOBatch, PPOConfig, generalized_advantages, update_ppo
from .solving import solve_matrix, value_gaps
from .storage import CheckpointStore, DEFAULT_RESERVE

TASK_VERSION = "graygo/matrix-role/1"


def matrix_payoffs(name):
    if name == "asymmetric":
        return np.array([[1., -1.], [-1., .5]])
    if name == "rps":
        return np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
    if name == "matching-pennies":
        return np.array([[1., -1.], [-1., 1.]])
    if name == "pure82":
        result = -np.ones((82, 82)); result[0] = 1
        return result
    raise ValueError(f"unknown matrix task: {name}")


class MatrixActorCritic(nn.Module):
    """Compact public-role adapter; loss/update implementation is shared with boards."""
    def __init__(self, actions, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, actions)
        self.critic = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, features, buckets):
        if buckets.shape != (len(features), 2):
            raise ValueError("current/history critic bucket matrix required")
        encoded = self.encoder(features)
        return self.actor(encoded), self.critic(torch.cat((encoded,buckets),dim=1)).squeeze(-1)

    def policy_logits(self, features):
        return self.actor(self.encoder(features))


@dataclass(frozen=True)
class MatrixPPOConfig:
    task: str = "asymmetric"
    seed: int = 0
    records: int = 2_000_000
    rollout: int = 1024
    eval_every: int = 102_400
    entropy: str = "taper"
    device: str = "cpu"
    hidden: int = 64
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch: int = 128
    time_cap_seconds: float = 24 * 3600

    def __post_init__(self):
        matrix_payoffs(self.task)
        if self.records < 2 or self.records % 2 or self.rollout < 2 or self.rollout % 2:
            raise ValueError("positive even record/rollout counts required for role balance")
        if self.eval_every < 2 or self.hidden < 1 or self.epochs < 1 or self.minibatch < 1:
            raise ValueError("invalid matrix training configuration")
        if self.entropy not in ("taper", "fixed-0.1"):
            raise ValueError("entropy protocol must be taper or fixed-0.1")
        if self.learning_rate <= 0 or self.time_cap_seconds <= 0:
            raise ValueError("positive learning-rate/time budget required")

    def entropy_at(self, records):
        if self.entropy == "fixed-0.1":
            return .1
        return .01 + (.001 - .01) * min(1., records / (self.records / 2))


class MatrixPPOTrainer:
    def __init__(self, config, run_dir, *, resume=False, reserve_bytes=DEFAULT_RESERVE):
        self.config = config
        self.root = Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = CheckpointStore(self.root, reserve_bytes=reserve_bytes)
        self.device = torch.device(config.device)
        self.payoffs = matrix_payoffs(config.task)
        self.solution = solve_matrix(self.payoffs)  # evaluation only; never used as training labels
        self.action_counts = self.payoffs.shape
        self.max_actions = max(self.action_counts)
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.model = MatrixActorCritic(self.max_actions, config.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.own_rng = torch.Generator(device=self.device).manual_seed(config.seed + 101)
        self.opponent_rng = torch.Generator(device=self.device).manual_seed(config.seed + 102)
        self.shuffle_rng = torch.Generator().manual_seed(config.seed + 103)
        self.scheduler_rng = np.random.default_rng(config.seed + 104)
        self.records = self.cycles = self.optimizer_steps = self.episodes = 0
        self.per_role = [0, 0]
        self.current_opponents = self.historical_opponents = 0
        self.history = []  # exact two-role probability tables, strictly older than current actor
        self.snapshots = []  # deployment mixture, separate from training opponents
        self.evaluations = []
        self.next_eval = config.eval_every
        self.elapsed = 0.
        self._active_started = None
        self.last_saved = -1
        manifest_path = self.root / "run.json"
        if resume:
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest["config"] != asdict(config):
                raise ValueError("resume configuration differs from the original plan")
            if self.manifest["source_sha256"] != source_digest():
                raise ValueError("source changed since run creation; use an explicitly new run lineage")
            self._restore()
        else:
            if manifest_path.exists():
                raise FileExistsError("run already exists; use resume rather than overwrite")
            self.manifest = make_manifest(seed=config.seed, command={"operation": "matrix-ppo", "config": asdict(config)})
            self.manifest.update({"config": asdict(config), "task_schema": TASK_VERSION,
                                  "encoding_version": TASK_VERSION, "rule_version": None,
                                  "baseline_version": None, "payoffs": self.payoffs.tolist(),
                                  "model": {"type": "MatrixActorCritic", "hidden": config.hidden,
                                            "shared_parameters_across_roles": True, "critic_only_buckets": ["current","history"]},
                                  "opponent_scheduler": {"current": .5, "history": .5,
                                                         "archive": "pre-update exact role probability table each rollout"},
                                  "counter_unit": "one learner-side action; one physical one-round game per record",
                                  "pid": __import__("os").getpid()})
            base_settings=asdict(PPOConfig(epochs=config.epochs,minibatch=config.minibatch))
            base_settings.pop("entropy_weight")
            self.manifest.update({"ppo_base_settings":base_settings,
                                  "optimizer":{"class":"Adam","defaults":dict(self.optimizer.defaults)},
                                  "entropy_schedule":{"protocol":config.entropy,"clock":"fresh_records",
                                      "start":config.entropy_at(0),"end":config.entropy_at(config.records),
                                      "decay_records":config.records/2 if config.entropy=="taper" else None},
                                  "rng_stream_seeds":{"model":config.seed,"own_actions":config.seed+101,
                                      "opponent_actions":config.seed+102,"shuffle":config.seed+103,"scheduler":config.seed+104}})
            write_report(manifest_path, self.manifest, reserve_bytes=reserve_bytes)

    def _inputs(self, roles):
        features = torch.nn.functional.one_hot(roles, 2).float()
        counts = torch.tensor(self.action_counts, device=self.device)[roles]
        legal = torch.arange(self.max_actions, device=self.device)[None] < counts[:, None]
        return features, legal

    @torch.no_grad()
    def policies(self):
        features, legal = self._inputs(torch.arange(2, device=self.device))
        logits = self.model.policy_logits(features)
        probs = masked_log_probs(logits, legal).exp().cpu().double().numpy().copy()
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    @torch.no_grad()
    def collect(self, count):
        if count % 2:
            raise ValueError("rollout must preserve role balance")
        roles = torch.arange(count, device=self.device) % 2
        features, legal = self._inputs(roles)
        # Freeze the current opponent distributions before either commitment.
        current = self.policies()
        opponent_roles = 1 - roles.cpu().numpy()
        opponents = current[opponent_roles].copy()
        selected_history = np.zeros(count, dtype=bool)
        if self.history:
            selected_history = self.scheduler_rng.random(count) < .5
            indices = self.scheduler_rng.integers(0, len(self.history), int(selected_history.sum()))
            history = np.stack(self.history)
            opponents[selected_history] = history[indices, opponent_roles[selected_history]]
        buckets = torch.as_tensor(np.stack((~selected_history,selected_history),axis=1),device=self.device,dtype=torch.float32)
        logits, values = self.model(features,buckets)
        log_probs = masked_log_probs(logits, legal)
        own = torch.multinomial(log_probs.exp(), 1, generator=self.own_rng).squeeze(1)
        other = torch.multinomial(torch.as_tensor(opponents, device=self.device, dtype=torch.float32),
                                  1, generator=self.opponent_rng).squeeze(1)
        matrix = torch.as_tensor(self.payoffs, device=self.device, dtype=torch.float32)
        black = torch.where(roles == 0, own, other)
        white = torch.where(roles == 0, other, own)
        reward_black = matrix[black, white]
        rewards = torch.where(roles == 0, reward_black, -reward_black)
        advantages, returns = generalized_advantages(rewards[None], values[None],
                                                     torch.ones((1, count), dtype=torch.bool, device=self.device),
                                                     torch.zeros(count, device=self.device))
        batch = PPOBatch({"features": features, "buckets": buckets}, own, legal, log_probs.gather(1, own[:, None]).squeeze(1),
                         advantages[0], returns[0], torch.ones(count, dtype=torch.bool, device=self.device),
                         torch.ones(count, dtype=torch.bool, device=self.device))
        counts = {"current": count - int(selected_history.sum()), "history": int(selected_history.sum())}
        return batch, current, counts

    def _evaluate(self):
        current = self.policies()
        if self.records >= self.config.records / 2:
            self.snapshots.append({"records": self.records, "probabilities": current.tolist()})
        latest = value_gaps(self.payoffs, current[0, :self.action_counts[0]],
                            current[1, :self.action_counts[1]], self.solution.value)
        mixture = None
        if self.snapshots:
            mixed = np.mean([s["probabilities"] for s in self.snapshots], axis=0)
            mixed /= mixed.sum(axis=1, keepdims=True)
            mixture = list(value_gaps(self.payoffs, mixed[0, :self.action_counts[0]],
                                     mixed[1, :self.action_counts[1]], self.solution.value))
        result = {"records": self.records, "rollout_cycles": self.cycles,
                  "optimizer_steps": self.optimizer_steps, "latest_gaps": list(latest),
                  "mixture_gaps": mixture, "policies": current.tolist(),
                  "mixture_components": len(self.snapshots), "elapsed_seconds": self.elapsed_now()}
        self.evaluations.append(result)
        return result

    def elapsed_now(self):
        return self.elapsed + (time.monotonic() - self._active_started if self._active_started is not None else 0.)

    def _state(self):
        return {"format_version": 1, "config": asdict(self.config), "task_schema": TASK_VERSION,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "records": self.records, "cycles": self.cycles, "optimizer_steps": self.optimizer_steps,
                "episodes": self.episodes, "per_role": self.per_role,
                "current_opponents": self.current_opponents, "historical_opponents": self.historical_opponents,
                "history": torch.tensor(np.stack(self.history), dtype=torch.float64) if self.history else torch.empty((0,2,self.max_actions),dtype=torch.float64),
                "snapshots": self.snapshots, "evaluations": self.evaluations, "next_eval": self.next_eval,
                "elapsed": self.elapsed_now(), "own_rng": self.own_rng.get_state(),
                "opponent_rng": self.opponent_rng.get_state(), "shuffle_rng": self.shuffle_rng.get_state(),
                "scheduler_rng": self.scheduler_rng.bit_generator.state, "torch_rng": torch.get_rng_state()}

    def _save(self, *, weights=False):
        if weights:
            target = self.root / "weights" / f"{self.records:012d}"
            if target.exists():
                self.store.verify(target)
                previous = torch.load(target / "model.pt", weights_only=True, map_location=self.device)
                if previous["config"] != asdict(self.config) or any(not torch.equal(previous["model"][k],v) for k,v in self.model.state_dict().items()):
                    raise RuntimeError("permanent archive generation conflicts with current model")
            else:
                buffer = io.BytesIO()
                torch.save({"config": asdict(self.config), "model": self.model.state_dict()}, buffer)
                raw = buffer.getvalue()
                self.store.save(self.records, lambda p: (p / "model.pt").write_bytes(raw),
                                metadata={"task_schema": TASK_VERSION, "config": asdict(self.config)},
                                estimated_peak_bytes=len(raw)+65536, kind="weights")
        if self.last_saved == self.records:
            return
        buffer = io.BytesIO(); torch.save(self._state(), buffer); raw = buffer.getvalue()
        self.store.save(self.records, lambda p: (p / "trainer.pt").write_bytes(raw),
                        metadata={"task_schema": TASK_VERSION, "records": self.records,
                                  "source_sha256": self.manifest["source_sha256"]},
                        estimated_peak_bytes=len(raw)+65536)
        self.last_saved = self.records

    def _restore(self):
        path = self.store.latest()
        state = torch.load(path / "trainer.pt", weights_only=True, map_location=self.device)
        if state["format_version"] != 1 or state["config"] != asdict(self.config) or state["task_schema"] != TASK_VERSION:
            raise ValueError("checkpoint schema/configuration mismatch")
        self.model.load_state_dict(state["model"]); self.optimizer.load_state_dict(state["optimizer"])
        for field in ("records","cycles","optimizer_steps","episodes","per_role","current_opponents",
                      "historical_opponents","snapshots","evaluations","next_eval","elapsed"):
            setattr(self, field, state[field])
        self.history = [p.copy() for p in state["history"].cpu().numpy()]
        self.own_rng.set_state(state["own_rng"].cpu()); self.opponent_rng.set_state(state["opponent_rng"].cpu())
        self.shuffle_rng.set_state(state["shuffle_rng"].cpu())
        self.scheduler_rng.bit_generator.state = state["scheduler_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        self.last_saved = self.records

    def result(self, status):
        final = self.evaluations[-5:]
        enough = len(final) == 5 and self.records == self.config.records
        def passes(field):
            return bool(enough and all(e[field] is not None and max(e[field]) <= .05 for e in final))
        return {"status": status, "task": self.config.task, "seed": self.config.seed,
                "entropy_protocol": self.config.entropy, "records": self.records,
                "physical_episodes": self.episodes, "opponent_decisions": self.episodes,
                "per_role_records": self.per_role, "rollout_cycles": self.cycles,
                "optimizer_steps": self.optimizer_steps, "current_opponent_records": self.current_opponents,
                "historical_opponent_records": self.historical_opponents,
                "value": self.solution.value, "matrix_residual_gap": self.solution.residual_gap,
                "latest_passed": passes("latest_gaps"), "mixture_passed": passes("mixture_gaps"),
                "has_five_final_evaluations": enough,
                "formal_protocol": self.config.records == 2_000_000 and self.config.rollout == 1024
                    and self.config.eval_every == 102_400 and self.config.epochs == 4
                    and self.config.minibatch == 128 and self.config.learning_rate == 3e-4,
                "evaluations": self.evaluations,
                "elapsed_seconds": self.elapsed_now(), "source_sha256": self.manifest["source_sha256"]}

    def run(self, *, until_records=None, stop_requested=lambda: False):
        stop = self.config.records if until_records is None else until_records
        if stop < self.records or stop > self.config.records or stop % 2:
            raise ValueError("invalid rollout stop")
        if stop != self.config.records and stop % self.config.rollout:
            raise ValueError("partial stop must be at a full rollout boundary")
        self._active_started = time.monotonic()
        status = "complete"
        try:
            while self.records < stop:
                requested = stop_requested()
                if requested:
                    status = requested if isinstance(requested, str) else "interrupted"
                    break
                if self.elapsed_now() >= self.config.time_cap_seconds:
                    status = "budget_limited"; break
                self.store.ensure_space()
                count = min(self.config.rollout, self.config.records - self.records)
                batch, old_policy, opponents = self.collect(count)
                update_config = PPOConfig(epochs=self.config.epochs, minibatch=self.config.minibatch,
                                          entropy_weight=self.config.entropy_at(self.records))
                stats = update_ppo(self.model, self.optimizer, batch, update_config, shuffle_generator=self.shuffle_rng)
                self.history.append(old_policy)  # strictly pre-update behavior snapshot
                self.records += count; self.cycles += 1; self.optimizer_steps += stats["updates"]
                self.episodes += count
                self.per_role[0] += count//2; self.per_role[1] += count//2
                self.current_opponents += opponents["current"]; self.historical_opponents += opponents["history"]
                if self.records >= self.next_eval or self.records == self.config.records:
                    while self.next_eval <= self.records:
                        self.next_eval += self.config.eval_every
                    evaluation = self._evaluate(); self._save(weights=True)
                    write_report(self.root / "progress.json", self.result("running"), reserve_bytes=self.store.reserve_bytes)
                    print(json.dumps({"event":"matrix-evaluation", "task":self.config.task,
                                      "seed":self.config.seed, "entropy":self.config.entropy, **evaluation}), flush=True)
            if self.records < self.config.records and status == "complete":
                status = "segment_paused"
            self._save(weights=self.records == self.config.records)
            if self.records == self.config.records:
                self.store.finalize()
            result = self.result(status)
            write_report(self.root / "result.json", result, reserve_bytes=self.store.reserve_bytes)
            return result
        finally:
            self.elapsed = self.elapsed_now(); self._active_started = None


def preflight_summary(results):
    """One protocol AND one deployment representation must pass both tasks/all seeds."""
    passed = []
    for protocol in ("taper","fixed-0.1"):
        for representation in ("latest","mixture"):
            required = [r for r in results if r["entropy_protocol"] == protocol]
            expected = {(task, seed) for task in ("asymmetric","rps") for seed in (0,1,2)}
            keys = {(r["task"],r["seed"]) for r in required}
            if keys == expected and len(required) == 6 and all(r["status"] == "complete" and r[representation+"_passed"] for r in required):
                passed.append({"entropy_protocol":protocol,"representation":representation})
    expected_all = {(task,protocol,seed) for task in ("asymmetric","rps")
                    for protocol in ("taper","fixed-0.1") for seed in (0,1,2)}
    actual = {(r["task"],r["entropy_protocol"],r["seed"]) for r in results}
    complete = len(results)==12 and actual==expected_all and all(
        r["status"]=="complete" and r["has_five_final_evaluations"] and r["formal_protocol"] for r in results)
    return {"route":("ppo" if passed else "nfsp") if complete else "pending",
            "passing_configurations":passed, "trials":len(results), "complete":complete,
            "results":results}
