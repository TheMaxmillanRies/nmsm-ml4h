'Opt-in, from-scratch mean-objective experiments. Production training is unchanged.'
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import LossWeights, compute_loss, masked_mean
from .model import JointCausalTransformer, allowed_transitions
from .training import fit_feature_encoder, forward_batch, make_dataset, model_config, move_batch

MASKS = {
    "state": "next_mask", "covariate_values": "continuous_value_mask",
    "mean_mse": "continuous_value_mask", "scale_nll": "continuous_value_mask",
    "observations": "observation_loss_mask", "binary_covariates": "binary_loss_mask",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def objective_parts(model, batch, graph, cov_weight, objective):
    if objective not in ("nll", "mean_focused"):
        raise ValueError("Unknown continuous objective")
    if not math.isfinite(cov_weight) or cov_weight < 0:
        raise ValueError("Invalid covariate weight")
    outputs = forward_batch(model, batch)
    parts = compute_loss(outputs, batch, graph, LossWeights(cov_weight))
    residual = batch["target_continuous_values"] - outputs["covariate_location"]
    parts["mean_mse"] = masked_mean(residual.square(), batch["continuous_value_mask"])


    with torch.no_grad():
        condition = model.state_emb(batch["target_states"].clamp(0, model.num_states - 1))
        if not model.config.condition_covariates_on_next_state:
            condition = torch.zeros_like(condition)
        trunk = model.covariate_trunk(torch.cat((outputs["hidden"].detach(), condition), dim=-1))
    sigma = F.softplus(model.covariate_scale_head(trunk).clamp(-8., 6.)) + 1e-3
    scale_nll = .5 * (residual.detach() / sigma).square() + sigma.log() + .5 * math.log(2 * math.pi)
    parts["scale_nll"] = masked_mean(scale_nll, batch["continuous_value_mask"])
    parts["original_total"] = parts["total"]
    if objective == "mean_focused":
        parts["total"] = parts["state"] + cov_weight * (
            parts["mean_mse"] + parts["scale_nll"] + parts["observations"] + parts["binary_covariates"]
        )
    return parts, outputs


def component_weights(objective, cov_weight):
    if objective not in ("nll", "mean_focused"):
        raise ValueError("Unknown objective")
    result = {"state": 1., "observations": cov_weight, "binary_covariates": cov_weight}
    if objective == "nll":
        result["covariate_values"] = cov_weight
    else:
        result.update(mean_mse=cov_weight, scale_nll=cov_weight)
    return result


def counts(batch):
    return {key: int(batch[mask].sum()) for key, mask in MASKS.items()}


def microbatches(batch, size):
    for start in range(0, len(batch["ids"]), size):
        yield {key: value[start:start + size] for key, value in batch.items()}


def accumulated_objective(parts, micro_counts, full_counts, objective, weight):


    return sum(
        factor * parts[key] * micro_counts[key] / max(full_counts[key], 1)
        for key, factor in component_weights(objective, weight).items()
    )


class ResourcePause(RuntimeError):
    'Stop this experiment; never kill or wait for unrelated work.'


def available_memory():
    if psutil is not None:
        return psutil.virtual_memory().available
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read available memory")


def process_rss():
    if psutil is not None:
        memory = psutil.Process().memory_info()
        return max(memory.rss, getattr(memory, "private", 0))
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read process resident memory")


def cpu_percent(startup=False):
    if psutil is not None:
        return psutil.cpu_percent(.2 if startup else None)

    return min(100., 100. * os.getloadavg()[0] / max(os.cpu_count() or 1, 1))


def allocation_budget(total, free, reserve, cap, margin):
    budget = min(cap, free - reserve - margin)
    if budget < 512 * 1024**2 or budget >= total:
        raise ResourcePause("Insufficient GPU headroom for the configured reserve and safety margin")
    return budget


class ComputeBudget:
    def __init__(self, device, *, min_gpu_free_gib=6., max_gpu_gib=6.,
                 min_ram_free_gib=12., max_ram_gib=12., gpu_margin_gib=1.,
                 max_cpu_percent=85., check_seconds=1.):
        requested = torch.device(device)
        self.device = torch.device("cuda:0") if requested.type == "cuda" and requested.index is None else requested
        self.reserve = min_gpu_free_gib * 2**30
        self.cap = max_gpu_gib * 2**30
        self.margin = gpu_margin_gib * 2**30
        self.min_ram = min_ram_free_gib * 2**30
        self.max_ram = max_ram_gib * 2**30
        self.max_cpu = max_cpu_percent
        self.interval = check_seconds
        self.last = -math.inf
        self.high_cpu_checks = 0
        self.samples = []
        self.peak_rss = self.peak_cuda_reserved = 0
        if psutil is not None:
            psutil.cpu_percent(None)

    def initialize(self):
        self.check(force=True, startup=True)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            budget = allocation_budget(total, free, self.reserve, self.cap, self.margin)
            torch.cuda.set_per_process_memory_fraction(budget / total, self.device)
            self.allocator_cap = budget
        self.check(force=True)

    def check(self, force=False, startup=False):
        if not force and time.monotonic() - self.last < self.interval:
            return
        self.last = time.monotonic()
        rss = process_rss()
        available = available_memory()
        cpu = cpu_percent(startup)
        self.peak_rss = max(self.peak_rss, rss)
        sample = {"time": time.time(), "cpu_percent": cpu, "ram_free_gib": available / 2**30,
                  "process_ram_gib": rss / 2**30}
        if available < self.min_ram or rss > self.max_ram:
            raise ResourcePause("RAM guard: " + json.dumps(sample))
        self.high_cpu_checks = self.high_cpu_checks + 1 if cpu > self.max_cpu else 0
        if self.high_cpu_checks >= 3 or (startup and cpu > self.max_cpu):
            raise ResourcePause("CPU is busy: " + json.dumps(sample))
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            self.peak_cuda_reserved = max(self.peak_cuda_reserved, reserved)
            sample.update(gpu_free_gib=free / 2**30, cuda_reserved_gib=reserved / 2**30)
            if free < self.reserve + self.margin or reserved > self.cap:
                raise ResourcePause("GPU headroom guard: " + json.dumps(sample))
        self.samples.append(sample)
        self.samples = self.samples[-100:]

    def hook(self, module, inputs):
        self.check()

    def summary(self):
        return {"device": str(self.device), "peak_process_ram_gib": self.peak_rss / 2**30,
                "peak_cuda_reserved_gib": self.peak_cuda_reserved / 2**30, "recent_samples": self.samples,
                "min_gpu_free_gib": self.reserve / 2**30, "gpu_margin_gib": self.margin / 2**30,
                "max_cuda_allocator_gib": getattr(self, "allocator_cap", 0) / 2**30}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_factual(model, loader, graph, device, cov_weight, guard):
    model.eval()
    sums, totals = {k: 0. for k in MASKS}, {k: 0 for k in MASKS}
    variable = None
    for raw in loader:
        guard.check()
        batch = move_batch(raw, device)
        parts, outputs = objective_parts(model, batch, graph, cov_weight, "nll")
        n = counts(raw)
        for key in sums:
            sums[key] += float(parts[key]) * n[key]
            totals[key] += n[key]
        valid = batch["continuous_value_mask"]
        error = outputs["covariate_location"] - batch["target_continuous_values"]
        sigma = outputs["covariate_log_scale"].exp()
        values = torch.stack((valid.sum((0, 1)), torch.where(valid, error, 0.).sum((0, 1)),
                              torch.where(valid, error.square(), 0.).sum((0, 1)),
                              ((error.abs() <= 1.96 * sigma) & valid).sum((0, 1)))).cpu().double().numpy()
        variable = values if variable is None else variable + values
    if not totals["state"]:
        raise ValueError("No validation state targets")
    result = {k: sums[k] / totals[k] if totals[k] else 0. for k in sums}
    result["standardized_covariate_rmse"] = math.sqrt(result["mean_mse"])
    if not all(math.isfinite(x) for x in result.values()):
        raise FloatingPointError("Non-finite factual validation metric")
    return {**result, **{f"count_{k}": v for k, v in totals.items()}}, variable


def train_from_scratch(train, validation, config, objective, directory, seed,
                       microbatch_size, patience, guard, run_id):
    'Exact epoch-boundary resume, including optimizer and random states.'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = {"run_id": run_id, "config": asdict(config), "objective": objective,
            "seed": seed, "microbatch_size": microbatch_size, "patience": patience,
            "checkpoint_metric": "state", "initialization": "random_from_scratch"}
    spec_id = fingerprint(spec)
    if (directory / "spec.json").exists():
        if fingerprint(json.loads((directory / "spec.json").read_text())) != spec_id:
            raise ValueError("Cannot resume a fit with a different specification")
    else:
        write_json(directory / "spec.json", spec)
    if (directory / "complete.json").exists():
        result = json.loads((directory / "complete.json").read_text())
        if result["best_sha256"] != file_hash(directory / "best.pt"):
            raise ValueError("Completed checkpoint hash mismatch")
        return result
    guard.check(force=True)
    seed_all(seed)
    encoder = fit_feature_encoder(train, config)
    write_json(directory / "feature_encoder.json", {"fit_scope": "inner-training only", **encoder.state_dict()})
    train_set = make_dataset(train, config, encoder, "original")
    valid_set = make_dataset(validation, config, encoder, "original")
    from .data import collate_trajectories
    generator = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                        generator=generator, num_workers=0, collate_fn=collate_trajectories)
    valid_loader = DataLoader(valid_set, batch_size=microbatch_size, shuffle=False,
                              generator=torch.Generator().manual_seed(seed + 2),
                              num_workers=0, collate_fn=collate_trajectories)
    architecture = model_config(config, encoder, "original")
    model = JointCausalTransformer(architecture).to(guard.device)
    graph = allowed_transitions(config.num_states, config.terminal_states, config.allowed_edges).to(guard.device)
    hook = model.state_head.register_forward_pre_hook(guard.hook)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    initial_hash = state_hash(model)
    start_epoch, best, stale, history = 0, float("inf"), 0, []
    started = time.monotonic()
    accumulated_seconds = 0.

    def payload(epoch):
        return {"model": model.state_dict(), "architecture": asdict(architecture),
                "feature_encoder": encoder.state_dict(), "config": asdict(config),
                "variant": "original", "objective": objective, "epoch": epoch,
                "optimizer": optimizer.state_dict(), "rng": rng_state(generator),
                "best_score": best, "stale": stale, "history": history,
                "initial_state_sha256": initial_hash, "spec_id": spec_id,
                "checkpoint_metric": "state", "runtime_seconds": accumulated_seconds + time.monotonic() - started}

    last_path = directory / "last.pt"
    if last_path.exists():
        prior = torch.load(last_path, map_location="cpu", weights_only=False)
        if prior["spec_id"] != spec_id:
            raise ValueError("Resume checkpoint specification mismatch")
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(guard.device)
        restore_rng(prior["rng"], generator)
        start_epoch, best, stale, history = prior["epoch"], prior["best_score"], prior["stale"], prior["history"]
        accumulated_seconds = prior["runtime_seconds"]
        if prior["initial_state_sha256"] != initial_hash:
            raise ValueError("Recreated random initialization differs from resumed fit")
        del prior
    else:
        save_checkpoint(last_path, payload(0))
    try:
        for epoch in range(start_epoch + 1, config.epochs + 1):
            if stale >= patience:
                break
            model.train()
            batch_digest = hashlib.sha256()
            epoch_sums, epoch_counts = {k: 0. for k in MASKS}, {k: 0 for k in MASKS}
            for raw in loader:
                guard.check()
                batch_digest.update(json.dumps(raw["ids"]).encode())
                full_counts = counts(raw)
                optimizer.zero_grad(set_to_none=True)
                for micro in microbatches(raw, microbatch_size):
                    guard.check()
                    micro_counts = counts(micro)
                    batch = move_batch(micro, guard.device)
                    parts, _ = objective_parts(model, batch, graph, config.lambda_cov, objective)
                    loss = accumulated_objective(parts, micro_counts, full_counts, objective, config.lambda_cov)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite objective")
                    loss.backward()
                    for key in MASKS:
                        epoch_sums[key] += float(parts[key].detach()) * micro_counts[key]
                        epoch_counts[key] += micro_counts[key]
                    del loss, parts, batch
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
            scores, _ = evaluate_factual(model, valid_loader, graph, guard.device, config.lambda_cov, guard)
            improved = scores["state"] < best - 1e-5
            best, stale = (scores["state"], 0) if improved else (best, stale + 1)
            row = {"epoch": epoch, "batch_order_sha256": batch_digest.hexdigest(),
                   "training": {k: epoch_sums[k] / max(epoch_counts[k], 1) for k in MASKS}, "validation": scores}
            history.append(row)
            checkpoint = payload(epoch)
            if improved:

                save_checkpoint(directory / "best.pt", {k: v for k, v in checkpoint.items() if k not in ("optimizer", "rng", "history")})
            save_checkpoint(last_path, checkpoint)
            write_json(directory / "history.json", history)
            print("FULL_MEAN_EPOCH " + json.dumps({"fit": str(directory), "epoch": epoch,
                  "validation_state": scores["state"], "validation_covariate_rmse": scores["standardized_covariate_rmse"],
                  "stale": stale, "elapsed_seconds": checkpoint["runtime_seconds"]}), flush=True)
        best_checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        result = {"status": "complete", "best_epoch": best_checkpoint["epoch"], "epochs_run": len(history),
                  "epoch_cap": config.epochs, "stopped_by_patience": stale >= patience,
                  "best_at_training_boundary": best_checkpoint["epoch"] == len(history),
                  "initial_state_sha256": initial_hash, "spec_id": spec_id,
                  "runtime_seconds": accumulated_seconds + time.monotonic() - started,
                  "best_sha256": file_hash(directory / "best.pt"), "history": history}
        write_json(directory / "complete.json", result)
        return result
    finally:
        hook.remove()
        write_json(directory / "resources.json", guard.summary())
