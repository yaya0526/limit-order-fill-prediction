"""Independent, deterministic market scenarios for transfer stress testing.

The simulator deliberately does not read or estimate parameters from the source
workbook.  It implements a small queue-depletion mechanism with four declared
market regimes so frozen models can be evaluated under a different data-
generating process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd


HORIZONS = (5, 30, 60)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    mechanism: str
    spread_values: tuple[int, ...]
    spread_probabilities: tuple[float, ...]
    depth_scale: float
    volatility_bps: float
    flow_count_scale: float
    flow_volume_scale: float
    distance_probabilities: tuple[float, ...]
    hidden_queue_scale: float = 0.0


SCENARIOS = (
    ScenarioConfig(
        name="liquid_consistent",
        description=(
            "Tighter spreads, deeper displayed liquidity, and faster order flow; "
            "fills follow visible queue depletion."
        ),
        mechanism="queue_depletion",
        spread_values=(1, 2, 3),
        spread_probabilities=(0.82, 0.15, 0.03),
        depth_scale=1.25,
        volatility_bps=0.65,
        flow_count_scale=1.25,
        flow_volume_scale=1.30,
        distance_probabilities=(0.62, 0.23, 0.10, 0.04, 0.01),
    ),
    ScenarioConfig(
        name="wide_thin_volatile",
        description=(
            "Wider spreads, thinner displayed depth, higher volatility, and slower "
            "order flow with the same visible queue-depletion mechanism."
        ),
        mechanism="queue_depletion",
        spread_values=(2, 3, 4, 5),
        spread_probabilities=(0.25, 0.35, 0.25, 0.15),
        depth_scale=0.55,
        volatility_bps=2.20,
        flow_count_scale=0.70,
        flow_volume_scale=0.70,
        distance_probabilities=(0.35, 0.28, 0.20, 0.12, 0.05),
    ),
    ScenarioConfig(
        name="flow_reversal",
        description=(
            "Aggressive displayed flow triggers replenishment; observed flow against "
            "the order is therefore a misleading fill signal."
        ),
        mechanism="flow_reversal",
        spread_values=(1, 2, 3),
        spread_probabilities=(0.62, 0.28, 0.10),
        depth_scale=0.95,
        volatility_bps=1.15,
        flow_count_scale=1.30,
        flow_volume_scale=1.45,
        distance_probabilities=(0.55, 0.25, 0.13, 0.05, 0.02),
    ),
    ScenarioConfig(
        name="hidden_queue",
        description=(
            "Displayed depth is low but unobserved priority sits ahead of the order, "
            "breaking the visible-queue assumption."
        ),
        mechanism="hidden_queue",
        spread_values=(1, 2, 3),
        spread_probabilities=(0.70, 0.24, 0.06),
        depth_scale=0.65,
        volatility_bps=0.85,
        flow_count_scale=1.00,
        flow_volume_scale=1.00,
        distance_probabilities=(0.60, 0.23, 0.11, 0.05, 0.01),
        hidden_queue_scale=0.75,
    ),
)


def scenario_manifest() -> list[dict]:
    """Return the immutable public scenario declaration."""
    return [asdict(scenario) for scenario in SCENARIOS]


def scenario_manifest_hash() -> str:
    """Hash the exact scenario declaration for auditability."""
    encoded = json.dumps(
        scenario_manifest(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _market_activity(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.choice(
        np.array(["low", "medium", "high"]),
        size=size,
        p=np.array([0.30, 0.55, 0.15]),
    )


def _order_sizes(rng: np.random.Generator, size: int) -> np.ndarray:
    small = rng.choice(np.array([1, 2, 3, 5, 10, 25]), size=size)
    block = np.rint(rng.lognormal(mean=5.15, sigma=0.75, size=size)).astype(int)
    return np.where(rng.random(size) < 0.76, small, block).clip(1, 4000)


def simulate_scenario(
    config: ScenarioConfig,
    rows: int = 4_000,
    random_seed: int = 20261026,
) -> pd.DataFrame:
    """Generate one independently configured scenario.

    Labels come from a queue-depletion clock rather than from a fitted model.
    The function is deterministic for a given scenario, row count, and seed.
    """
    if rows < 100:
        raise ValueError("At least 100 rows are required for a stable stress scenario.")
    seed_offset = sum((index + 1) * ord(char) for index, char in enumerate(config.name))
    rng = np.random.default_rng(random_seed + seed_offset)

    side = rng.choice(np.array(["buy", "sell"]), size=rows)
    seconds_from_open = rng.uniform(300, 23_100, size=rows)
    session_phase = np.where(
        seconds_from_open < 2_400,
        "opening",
        np.where(seconds_from_open > 20_700, "closing", "continuous"),
    )
    activity = _market_activity(rng, rows)
    activity_factor = pd.Series(activity).map(
        {"low": 0.62, "medium": 1.0, "high": 1.65}
    ).to_numpy()

    spread_ticks = rng.choice(
        np.asarray(config.spread_values),
        size=rows,
        p=np.asarray(config.spread_probabilities),
    )
    volatility = np.clip(
        rng.lognormal(mean=np.log(config.volatility_bps), sigma=0.28, size=rows),
        0.20,
        6.0,
    )
    mid_price = np.clip(
        140.0
        + 0.00012 * seconds_from_open
        + rng.normal(0, 0.65 + 0.22 * volatility, size=rows),
        132.0,
        154.0,
    )
    tick = 0.01
    best_bid = mid_price - spread_ticks * tick / 2
    best_ask = mid_price + spread_ticks * tick / 2

    book: dict[str, np.ndarray] = {}
    bid_volumes = []
    ask_volumes = []
    for level in range(1, 6):
        book[f"bid_price_l{level}"] = best_bid - (level - 1) * tick
        book[f"ask_price_l{level}"] = best_ask + (level - 1) * tick
        level_taper = 1 + 0.08 * (level - 1)
        bid_volume = np.maximum(
            1,
            np.rint(
                rng.gamma(
                    shape=4.0,
                    scale=430 * config.depth_scale * level_taper,
                    size=rows,
                )
            ),
        ).astype(int)
        ask_volume = np.maximum(
            1,
            np.rint(
                rng.gamma(
                    shape=4.0,
                    scale=430 * config.depth_scale * level_taper,
                    size=rows,
                )
            ),
        ).astype(int)
        book[f"bid_volume_l{level}"] = bid_volume
        book[f"ask_volume_l{level}"] = ask_volume
        bid_volumes.append(bid_volume)
        ask_volumes.append(ask_volume)

    bid_depth = np.vstack(bid_volumes).sum(axis=0)
    ask_depth = np.vstack(ask_volumes).sum(axis=0)
    buy = side == "buy"
    same_side_depth = np.where(buy, bid_depth, ask_depth)
    opposite_side_depth = np.where(buy, ask_depth, bid_depth)
    five_level_side_imbalance = (
        same_side_depth - opposite_side_depth
    ) / (same_side_depth + opposite_side_depth)
    top_imbalance = (bid_volumes[0] - ask_volumes[0]) / (
        bid_volumes[0] + ask_volumes[0]
    )
    microprice = (
        best_ask * bid_volumes[0] + best_bid * ask_volumes[0]
    ) / (bid_volumes[0] + ask_volumes[0])

    order_size = _order_sizes(rng, rows)
    partial_mask = rng.random(rows) < 0.11
    partial_fraction = rng.uniform(0.05, 0.75, size=rows)
    quantity_filled = np.where(
        partial_mask,
        np.floor(order_size * partial_fraction),
        0,
    ).astype(int)
    quantity_filled = np.minimum(quantity_filled, np.maximum(order_size - 1, 0))
    same_price_total = np.maximum(
        20,
        np.rint(
            np.where(buy, bid_volumes[0], ask_volumes[0])
            + rng.gamma(shape=2.0, scale=260 * config.depth_scale, size=rows)
        ),
    ).astype(int)
    queue_fraction = rng.beta(2.25, 1.15, size=rows)
    queue_ahead = np.maximum(1, np.rint(queue_fraction * same_price_total)).astype(int)
    normalized_queue = queue_ahead / same_price_total
    distance = rng.choice(
        np.arange(5), size=rows, p=np.asarray(config.distance_probabilities)
    )

    count_multiplier = config.flow_count_scale * activity_factor
    same_submissions = rng.poisson(5.8 * count_multiplier)
    opposite_submissions = rng.poisson(5.8 * count_multiplier)
    same_cancellations = rng.poisson(3.2 * count_multiplier)
    opposite_cancellations = rng.poisson(3.2 * count_multiplier)
    modifications = rng.poisson(1.2 * count_multiplier)
    trade_count = rng.poisson(7.8 * count_multiplier)
    against_shape = 1.4 + 0.24 * trade_count
    with_shape = 1.2 + 0.20 * trade_count
    aggressive_against = rng.gamma(
        shape=against_shape,
        scale=58 * config.flow_volume_scale,
    )
    aggressive_with = rng.gamma(
        shape=with_shape,
        scale=52 * config.flow_volume_scale,
    )
    time_since_trade = np.minimum(
        rng.exponential(1_050 / np.maximum(count_multiplier, 0.1)),
        18_000,
    )

    remaining_size = np.maximum(order_size - quantity_filled, 1)
    visible_required = queue_ahead + 0.50 * remaining_size
    cancel_release_rate = 3.2 * same_cancellations * activity_factor
    visible_trade_rate = 0.085 * aggressive_against
    required_volume = visible_required.astype(float)
    effective_rate = 1.0 + visible_trade_rate + 0.45 * cancel_release_rate

    if config.mechanism == "flow_reversal":
        # Displayed aggressive flow attracts replenishment instead of consuming
        # the observed queue.  This is deliberately antagonistic to the source
        # model's order-flow assumption.
        required_volume = required_volume + 1.35 * aggressive_against
        effective_rate = 1.0 + 0.052 * aggressive_with + 0.20 * cancel_release_rate
    elif config.mechanism == "hidden_queue":
        hidden_queue = config.hidden_queue_scale * same_side_depth * rng.lognormal(
            mean=-0.15, sigma=0.55, size=rows
        )
        required_volume = required_volume + hidden_queue
        effective_rate = 0.72 * effective_rate
    elif config.mechanism != "queue_depletion":
        raise ValueError(f"Unknown simulation mechanism: {config.mechanism}")

    touch_delay = (
        distance
        * (5.0 + 4.2 * spread_ticks)
        / np.sqrt(np.maximum(volatility, 0.15) / 0.8)
    )
    depletion_time = required_volume / np.maximum(effective_rate, 0.25)
    fill_time = (touch_delay + depletion_time) * rng.lognormal(
        mean=0.0, sigma=0.24, size=rows
    )

    frame = pd.DataFrame(
        {
            "scenario": config.name,
            "order_id": [f"SIM-{config.name}-{index:05d}" for index in range(rows)],
            "trading_day": pd.Timestamp("2030-01-02"),
            "seconds_from_open": seconds_from_open,
            "session_phase": session_phase,
            "side": side,
            "order_age_s": np.clip(rng.exponential(11.0, size=rows), 0.05, 75.0),
            "order_size": order_size,
            "price_distance_ticks": distance,
            "quantity_filled_before_t0": quantity_filled,
            "queue_ahead_qty": queue_ahead,
            "same_price_total_qty": same_price_total,
            "normalized_queue_ahead": normalized_queue,
            "mid_price": mid_price,
            "spread_ticks": spread_ticks,
            "relative_spread_bps": spread_ticks * tick / mid_price * 10_000,
            "top_level_imbalance": top_imbalance,
            "five_level_side_imbalance": five_level_side_imbalance,
            "same_side_depth_5": same_side_depth,
            "opposite_side_depth_5": opposite_side_depth,
            "microprice": microprice,
            "volatility_60s_bps": volatility,
            "same_side_submissions_10s": same_submissions,
            "opposite_side_submissions_10s": opposite_submissions,
            "same_side_cancellations_10s": same_cancellations,
            "opposite_side_cancellations_10s": opposite_cancellations,
            "modifications_10s": modifications,
            "aggressive_trade_volume_against_order_10s": aggressive_against,
            "aggressive_trade_volume_with_order_side_10s": aggressive_with,
            "trade_count_10s": trade_count,
            "time_since_last_trade_ms": time_since_trade,
            "market_activity_group": activity,
            "simulated_fill_time_seconds": fill_time,
        }
    )
    for key, values in book.items():
        frame[key] = values
    for horizon in HORIZONS:
        frame[f"eligible_{horizon}s"] = 1
        frame[f"target_fill_{horizon}s"] = (fill_time <= horizon).astype(int)
    return frame


def transfer_status(
    pr_auc_lift: float,
    brier_skill: float,
    minimum_pr_lift: float = 0.02,
) -> str:
    """Classify transfer using criteria declared before scenario scoring."""
    if not np.isfinite(pr_auc_lift) or pr_auc_lift <= minimum_pr_lift:
        return "failure"
    if np.isfinite(brier_skill) and brier_skill > 0:
        return "ranking_and_calibration_transfer"
    return "ranking_only_transfer"
