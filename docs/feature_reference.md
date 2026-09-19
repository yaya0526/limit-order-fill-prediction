# Feature Reference

## Single-Symbol Synthetic Limit-Order-Book Dataset

This document defines the variables supplied in `AI_Order_Execution.xlsx` and the raw historical fields described by its `JSON_Schema` sheet. It also defines a reproducible set of summary features that can be constructed from the 50-snapshot history.

The dataset contains synthetic teaching data for one symbol, AAPL. Definitions in this document describe the supplied dataset and should not be interpreted as empirical claims about real AAPL trading.

## 1. Conventions

### Observation time

`t0` is the order-level observation time. Every predictor in the `Orders` sheet is measured at or before `t0`. Outcomes are observed after `t0`.

### Book levels

- Level 1 is the best quote.
- Bid prices decrease from L1 to L5.
- Ask prices increase from L1 to L5.
- `tick_size` is USD 0.01 for every row.

### Order-side orientation

| Resting order | Same side | Opposite side | Aggressive volume against the order |
|---|---|---|---|
| Buy | Bid | Ask | Aggressive sell volume consuming bids. |
| Sell | Ask | Bid | Aggressive buy volume consuming asks. |

### Time windows

Fields ending in `_10s` summarize the 10-second interval immediately preceding `t0`. `volatility_60s_bps` summarizes the 60-second interval ending at `t0`.

## 2. Order context and queue state

| Field | Type / unit | Definition |
|---|---|---|
| `side` | Category | Direction of the resting order. `buy` orders rest on the bid side; `sell` orders rest on the ask side. |
| `seconds_from_open` | Decimal / seconds | Seconds elapsed between the 09:30 market open and `t0`. Range: 0-23,400. |
| `session_phase` | Category | Intraday phase at `t0`: `opening` for the first 30 minutes, `continuous` for the middle session, and `closing` for the final hour. |
| `order_age_s` | Decimal / seconds | Seconds elapsed between the order's original submission and `t0`. |
| `limit_price` | Decimal / USD per share | Fixed limit price of the resting order at `t0`. The value is aligned to `tick_size`. |
| `order_size` | Integer / shares | Original submitted order quantity. Range: 1-5,000 shares. |
| `price_distance_ticks` | Integer / ticks | Distance from the order price to the same-side best quote at `t0`. For a buy order: `(bid_price_l1 - limit_price) / tick_size`. For a sell order: `(limit_price - ask_price_l1) / tick_size`. Zero means that the order rests at the best quote. Range: 0-4. |
| `quantity_filled_before_t0` | Integer / shares | Quantity executed between original submission and `t0`. It is strictly smaller than `order_size` because the order remains active at `t0`. |
| `queue_ahead_qty` | Integer / shares | Displayed quantity ahead of the studied order at the same limit price at `t0`. |
| `same_price_total_qty` | Integer / shares | Total displayed quantity at the studied order's price, including quantity ahead, the order's remaining displayed quantity, and quantity behind it. |
| `normalized_queue_ahead` | Decimal / ratio | Relative queue position: `queue_ahead_qty / same_price_total_qty`. Range: `[0, 1)`. |

## 3. Five-level limit order book at t0

### Bid and ask prices

| Fields | Type / unit | Definition |
|---|---|---|
| `bid_price_l1` to `bid_price_l5` | Decimal / USD per share | Displayed bid prices at levels 1-5. L1 is the highest bid; deeper bid prices decrease in tick increments. |
| `ask_price_l1` to `ask_price_l5` | Decimal / USD per share | Displayed ask prices at levels 1-5. L1 is the lowest ask; deeper ask prices increase in tick increments. |

### Bid and ask volumes

| Fields | Type / unit | Definition |
|---|---|---|
| `bid_volume_l1` to `bid_volume_l5` | Integer / shares | Displayed bid quantity at each of the five price levels. |
| `ask_volume_l1` to `ask_volume_l5` | Integer / shares | Displayed ask quantity at each of the five price levels. |

The raw five-level book therefore contains 20 columns: five bid prices, five bid volumes, five ask prices, and five ask volumes.

## 4. Derived LOB state at t0

Define:

- `P_bid = bid_price_l1`
- `P_ask = ask_price_l1`
- `V_bid = bid_volume_l1`
- `V_ask = ask_volume_l1`
- `D_bid = sum(bid_volume_l1, ..., bid_volume_l5)`
- `D_ask = sum(ask_volume_l1, ..., ask_volume_l5)`

| Field | Type / unit | Exact definition |
|---|---|---|
| `mid_price` | Decimal / USD per share | Arithmetic midpoint: `(P_bid + P_ask) / 2`. |
| `spread_ticks` | Integer / ticks | Best-quote spread in ticks: `(P_ask - P_bid) / tick_size`. Range: 1-4. |
| `relative_spread_bps` | Decimal / basis points | Best-quote spread relative to the midpoint: `10,000 * (P_ask - P_bid) / mid_price`. |
| `top_level_imbalance` | Decimal / ratio | L1 bid-ask volume imbalance: `(V_bid - V_ask) / (V_bid + V_ask)`. Range: `[-1, 1]`. This field is not oriented by the studied order's side. |
| `same_side_depth_5` | Integer / shares | Five-level depth on the studied order's side. It equals `D_bid` for a buy order and `D_ask` for a sell order. |
| `opposite_side_depth_5` | Integer / shares | Five-level depth opposite the studied order. It equals `D_ask` for a buy order and `D_bid` for a sell order. |
| `five_level_side_imbalance` | Decimal / ratio | Side-oriented five-level imbalance: `(same_side_depth_5 - opposite_side_depth_5) / (same_side_depth_5 + opposite_side_depth_5)`. Range: `[-1, 1]`. |
| `microprice` | Decimal / USD per share | Top-level volume-weighted price: `(V_bid * P_ask + V_ask * P_bid) / (V_bid + V_ask)`. It lies between the best bid and best ask. |
| `volatility_60s_bps` | Decimal / basis points | Synthetic realized-volatility signal over the 60 seconds ending at `t0`. The workbook does not specify the sampling frequency, return convention, or annualization rule used to generate this supplied field. |

## 5. Recent order flow before t0

### Submissions, cancellations, and modifications

| Field | Type / unit | Definition |
|---|---|---|
| `same_side_submissions_10s` | Integer / messages | New limit-order submissions on the studied order's side during the 10 seconds before `t0`. |
| `opposite_side_submissions_10s` | Integer / messages | New limit-order submissions on the side opposite the studied order during the 10 seconds before `t0`. |
| `same_side_cancellations_10s` | Integer / messages | Cancellation messages on the studied order's side during the 10 seconds before `t0`. This is an aggregate side-level count; it does not state that each cancellation occurred ahead of the studied order or at its price. |
| `opposite_side_cancellations_10s` | Integer / messages | Cancellation messages on the side opposite the studied order during the 10 seconds before `t0`. |
| `modifications_10s` | Integer / messages | Order-modification messages on both sides during the 10 seconds before `t0`. |

### Trades and activity

| Field | Type / unit | Definition |
|---|---|---|
| `aggressive_trade_volume_against_order_10s` | Integer / shares | Aggressive volume that consumed resting liquidity on the studied order's side during the 10 seconds before `t0`. For a resting buy order this is aggressive sell volume; for a resting sell order this is aggressive buy volume. |
| `aggressive_trade_volume_with_order_side_10s` | Integer / shares | Aggressive volume in the same trading direction as the studied order during the 10 seconds before `t0`. For a resting buy order this is aggressive buy volume consuming asks; for a resting sell order this is aggressive sell volume consuming bids. |
| `trade_count_10s` | Integer / trades | Number of trades during the 10 seconds before `t0`. |
| `trade_intensity_per_s` | Decimal / trades per second | `trade_count_10s / 10`. This is a deterministic rescaling of `trade_count_10s`. |
| `same_side_cancel_intensity_per_s` | Decimal / cancellations per second | `same_side_cancellations_10s / 10`. This is a deterministic rescaling of `same_side_cancellations_10s`. |
| `time_since_last_trade_ms` | Decimal / milliseconds | Milliseconds elapsed between the most recent trade and `t0`. |
| `market_activity_group` | Category | Activity regime derived from recent order-flow intensity: `low`, `medium`, or `high`. The workbook does not specify the numerical thresholds used to construct the categories. |

## 6. Historical JSON structure

The history data contain one object per order and join to the `Orders` sheet using `order_id`. Each order has exactly 50 snapshots in chronological order, all strictly earlier than the order's original submission time.

### Order-level JSON fields

| JSON field | Type / unit | Definition |
|---|---|---|
| `orders[].order_id` | Text | Join key corresponding to `Orders.order_id`. |
| `orders[].symbol` | Category | `AAPL` for every order. |
| `orders[].order_type` | Category | Synthetic generation mechanism: `observed` or `hypothetical_pegged`. This is metadata, not an order-book state variable. |
| `orders[].submission_timestamp` | ISO-8601 timestamp | Original order submission time. It precedes `observation_timestamp` by `order_age_s`. |
| `orders[].lookback_count` | Integer | Number of historical snapshots. Always 50. |
| `orders[].history` | Array of 50 objects | Chronological sequence of pre-submission LOB snapshots. |

### Snapshot fields

| JSON field | Type / unit | Definition |
|---|---|---|
| `sequence_step` | Integer | Position within the history. Step 1 is oldest; step 50 is closest to submission. |
| `event_timestamp` | ISO-8601 timestamp | Timestamp of the LOB event represented by the snapshot. It is strictly earlier than `submission_timestamp`. |
| `milliseconds_before_submission` | Integer / milliseconds | Negative time offset from submission. The closest snapshot has value -1 ms. |
| `event_type` | Category | Event producing the snapshot: `limit_submission`, `cancellation`, `market_trade`, or `modification`. |
| `event_side` | Category | Side changed by the event: `bid` or `ask`. |
| `event_level` | Integer | Affected LOB level, from 1 to 5. |
| `bid_prices` | Array[5] / USD per share | Bid prices from L1 to L5. |
| `bid_volumes` | Array[5] / shares | Displayed bid quantities from L1 to L5. |
| `ask_prices` | Array[5] / USD per share | Ask prices from L1 to L5. |
| `ask_volumes` | Array[5] / shares | Displayed ask quantities from L1 to L5. |
| `mid_price` | Decimal / USD per share | Mean of the best bid and best ask in the snapshot. |
| `spread_ticks` | Integer / ticks | Best-quote spread in the snapshot. |
| `top_level_imbalance` | Decimal / ratio | L1 bid-ask volume imbalance for the snapshot. |
| `five_level_bid_ask_imbalance` | Decimal / ratio | `(total bid depth - total ask depth) / (total bid depth + total ask depth)` across five levels. This raw JSON field is bid-ask oriented, not studied-order-side oriented. |
| `microprice` | Decimal / USD per share | Top-level volume-weighted microprice for the snapshot. |
| `volatility_60s_bps` | Decimal / basis points | Synthetic recent realized-volatility signal recorded at the snapshot. |

### Snapshot change fields

Each change field is measured relative to the preceding snapshot. All change arrays contain zeros at `sequence_step = 1` because no earlier snapshot exists in the stored history.

| JSON field | Type / unit | Definition |
|---|---|---|
| `changes.bid_price_ticks` | Array[5] / ticks | Change in each bid price level relative to the preceding snapshot. |
| `changes.bid_volume` | Array[5] / shares | Change in displayed bid volume at each level relative to the preceding snapshot. |
| `changes.ask_price_ticks` | Array[5] / ticks | Change in each ask price level relative to the preceding snapshot. |
| `changes.ask_volume` | Array[5] / shares | Change in displayed ask volume at each level relative to the preceding snapshot. |

## 7. Reproducible history-summary definitions

The following variables are constructed from the 50 snapshots. They are not pre-existing columns in the `Orders` sheet. The definitions below make the project brief's required history summaries explicit and reproducible.

Let snapshot index `j` run from 1 to 50. Define:

- `mid_j` as the snapshot midpoint;
- `spread_j` as the snapshot spread in ticks;
- `imbalance_j` as `five_level_bid_ask_imbalance`;
- `bid_depth_j = sum(bid_volumes_j)`;
- `ask_depth_j = sum(ask_volumes_j)`;
- `micro_gap_j_bps = 10,000 * (microprice_j - mid_j) / mid_j`.

### Price movement

| Constructed feature | Definition |
|---|---|
| `mid_return_1_50_bps` | `10,000 * (mid_50 - mid_1) / mid_1`. |
| `best_bid_change_count` | Number of transitions `j = 2,...,50` for which `bid_price_j[1] != bid_price_(j-1)[1]`. |
| `best_ask_change_count` | Number of transitions `j = 2,...,50` for which `ask_price_j[1] != ask_price_(j-1)[1]`. |
| `net_best_quote_direction` | Sum of signed L1 bid and ask price changes in ticks across transitions 2-50. If side orientation is used, multiply price direction by `+1` for buy and `-1` for sell and document that convention. |

### Spread behavior

| Constructed feature | Definition |
|---|---|
| `spread_mean_50` | Arithmetic mean of `spread_j` over the 50 snapshots. |
| `spread_max_50` | Maximum value of `spread_j`. |
| `spread_final` | `spread_50`. |
| `spread_one_tick_fraction` | Fraction of snapshots for which `spread_j == 1`. |

### Imbalance behavior

| Constructed feature | Definition |
|---|---|
| `imbalance_mean_50` | Arithmetic mean of `imbalance_j`. |
| `imbalance_std_50` | Population standard deviation of `imbalance_j`, using denominator 50 (`ddof = 0`). |
| `imbalance_min_50` | Minimum value of `imbalance_j`. |
| `imbalance_max_50` | Maximum value of `imbalance_j`. |
| `imbalance_final` | `imbalance_50`. |
| `imbalance_trend_50` | Ordinary-least-squares slope from regressing `imbalance_j` on `sequence_step j`. |

### Depth change

| Constructed feature | Definition |
|---|---|
| `bid_depth_change_1_50` | `bid_depth_50 - bid_depth_1`. |
| `ask_depth_change_1_50` | `ask_depth_50 - ask_depth_1`. |
| `same_side_depth_change_1_50` | For buy orders, `bid_depth_50 - bid_depth_1`; for sell orders, `ask_depth_50 - ask_depth_1`. |
| `opposite_side_depth_change_1_50` | For buy orders, `ask_depth_50 - ask_depth_1`; for sell orders, `bid_depth_50 - bid_depth_1`. |

### Microprice signal

| Constructed feature | Definition |
|---|---|
| `micro_gap_final_bps` | `micro_gap_50_bps`. |
| `micro_gap_mean_50_bps` | Arithmetic mean of `micro_gap_j_bps`. |
| `micro_gap_trend_50` | Ordinary-least-squares slope from regressing `micro_gap_j_bps` on `sequence_step j`. |

### Event composition

| Constructed feature | Definition |
|---|---|
| `event_count_<type>` | Number of snapshots whose `event_type` equals `<type>`, for each of `limit_submission`, `cancellation`, `market_trade`, and `modification`. |
| `event_count_<type>_<side>` | Number of snapshots with the specified `event_type` and `event_side`, for bid and ask separately. |

### Event timing

- `lookback_span_ms`: `event_timestamp_50 - event_timestamp_1`, expressed in milliseconds.
- `mean_inter_event_ms`: arithmetic mean of the 49 consecutive timestamp differences.
- `last_event_to_submission_ms`: `submission_timestamp - event_timestamp_50`, expressed in milliseconds. It should equal 1 ms under the documented schema.
