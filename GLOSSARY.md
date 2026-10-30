# Stock Market MCP

A server that gives coding agents market data, indicators, and advisory trading guidance. It never places orders.

## Language

### Signal

**Signal**:
An advisory answer to "is now the time to buy, sell, or hold this stock?", made of a Verdict, a Confidence, a rationale, the data date, the Price, the Hit Rate for its ticker and Interval (none until enough are scored), and a Trade Plan when the Verdict is buy or sell.
_Avoid_: Recommendation, call, alert

**Verdict**:
One of buy, sell, or hold – the direction the Signal leans.
_Avoid_: Decision, action

**Confidence**:
The classifier's own 0–1 estimate that the Verdict is right. Self-reported, not a measured hit rate.
_Avoid_: Probability, certainty, score

**Minimum Confidence**:
The lowest Confidence at which a buy or sell gets a Trade Plan. Set by the server from Hit Rates: 0.6 until enough Signals are scored.
_Avoid_: Threshold, cut-off

**Price**:
The last close in the data the Signal was made from; during the session that is the close of the bar still forming. Every Signal has one; on buy or sell it equals the Entry.
_Avoid_: Current price, last price, quote

**Outcome**:
What happened to a buy or sell Signal within 20 bars of its Interval, judged against the default Trade Plan: Target (reached first), Stop (reached first, or on the same bar as the Target), or Expired (neither). Scored even when the Signal was below the Minimum Confidence. A hold Signal's Outcome is Quiet (neither the long nor the short default Target reached) or Missed (one was).
_Avoid_: Result, win/loss

**Hit Rate**:
For buy and sell, Targets divided by Targets plus Stops over a set of scored Signals, Expired left out; for hold, Quiet divided by all scored holds. The measured counterpart to Confidence.
_Avoid_: Win rate, accuracy

**Interval**:
The length of one bar in the price history a Signal is made from: 1m, 5m, 15m, 1h, or 1d.
_Avoid_: Timeframe, resolution, granularity

### Trade Plan

**Trade Plan**:
The three prices for one Direction: Entry, Stop, and Target. Part of a Signal; a hold Verdict, or a Signal below the Minimum Confidence, has none. Advisory only.
_Avoid_: Setup, strategy, levels

**Direction**:
Long (profit if price rises) or short (profit if price falls). Set by the Verdict: buy is long, sell is short.
_Avoid_: Side, bias

**Entry**:
The price at which the Trade Plan opens the position.
_Avoid_: Buy price, trigger

**Stop**:
The price at which the Trade Plan is wrong and the position should close at a loss. Below Entry for long, above for short.
_Avoid_: Stop-loss price, invalidation

**Target**:
The price at which the Trade Plan takes profit. Above Entry for long, below for short.
_Avoid_: Take-profit, objective, goal

**Reward Ratio**:
The distance from Entry to Target divided by the distance from Entry to Stop.
_Avoid_: Risk/reward, R-multiple, R:R

### Watchlist

**Watchlist**:
A named list of tickers the user saves to scan together, usually one market per list.
_Avoid_: Portfolio, universe, list

**Scan**:
A quick set of numbers for each ticker in a Watchlist or ad-hoc list, used to pick which ones deserve a Signal. Has no Verdict.
_Avoid_: Screen, screener, filter

**Gap**:
The percent move from the previous session's close to this session's open. Before the open, it uses the latest pre-market price where the market has one.
_Avoid_: Opening move, overnight change

**Relative Volume**:
This session's volume so far divided by the average volume up to the same time of day over the last 10 sessions.
_Avoid_: RVOL, volume ratio

**VWAP Distance**:
How far the Price is from this session's VWAP (volume-weighted average price), as a percent. Positive when the Price is above it.
_Avoid_: VWAP gap, VWAP deviation
