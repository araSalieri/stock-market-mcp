# Dip is its own Verdict, with classifier-picked levels

The only buy was a momentum buy, so when a stock in an uptrend pulled back below its mean the classifier answered hold, and the user never got a Signal to buy the pullback. On a ticker not held the verdict question now also offers dip: price stretched below its mean (near or under the lower Bollinger Band, RSI oversold) while the longer trend still points up. Dip is never offered on a held Position, so the server never suggests averaging down against a cut.

A dip's Target is where the snap-back is likely to stop, a chart level, not a fixed multiple of ATR. So every not-held Signal is also asked to pick a dip Target from the levels above the Price and a dip Stop from those below, from the same candidates a cut uses (ADR-0010) with the sides swapped, or none. A dip is stored with its picked Stop and Target as its plan, scored like a buy against them, and never scored when either is missing. Its Reward Ratio varies, so it is judged against a Dip Break-even, as a cut is (ADR-0011). The Dip Minimum Confidence follows the same band rule over dips on all Intervals, since dips start with no record and one Interval alone would take long to fill.

## Considered Options

- Widen buy's criteria to cover pullbacks. Rejected: two strategies in one Hit Rate, so the user can't tell which one works, and buy's ATR Target doesn't fit mean reversion.
- A mode argument on get_signal (momentum or mean reversion). Rejected: the agent would have to guess the setup up front, and the classifier would still be asked one strategy at a time.
- Ask the dip level questions only after a dip Verdict. Rejected: a second classifier call per dip; asking them every time keeps one call per Signal.

## Consequences

- Every not-held Signal is asked two more choice questions, even when the Verdict is buy or hold.
- `PROMPT_VERSION` goes to 5, so every Hit Rate and Minimum Confidence, buy's and cut's too, starts over at its default.
- The dip's picked levels are stored in the long Stop and Target columns, so no schema change is needed.
- Buy's Minimum Confidence and Trade Plan are fed and built only by buys. `atr_multiple` and `reward_ratio` shape only buy's Trade Plan and are ignored on a dip.
- `get_signal` on a ticker not held adds `dip_min_confidence`.
