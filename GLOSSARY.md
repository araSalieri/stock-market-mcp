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

**Track Record**:
The Outcome counts and Hit Rates of the current prompt version's Signals, by Confidence band and Verdict, optionally for one ticker, one Interval, or both. Shows how Confidence compares with Hit Rate and why the Minimum Confidence sits where it does.
_Avoid_: Stats, performance, backtest, scorecard

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
_Avoid_: Universe, list

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

### Portfolio

**Portfolio**:
The Positions the user holds, set by the user through the agent. The server only records them; it never places orders.
_Avoid_: Holdings, account

**Position**:
One ticker in the Portfolio with a Direction, a quantity, and an average cost.
_Avoid_: Holding, lot

**Unrealised P&L**:
A Position's profit or loss if closed at a given price: the price minus the average cost, times the quantity, sign flipped for short. In the ticker's currency, also given as a percent of cost.
_Avoid_: Gain, return, open P&L

**Cut Loss**:
A Signal, made with the held Position in view, whose Verdict points against that Position (sell on a long, buy on a short) at or above the Minimum Confidence while the Position's Unrealised P&L at the Price is negative. Advisory only; the server never closes the Position.
_Avoid_: Stop out, exit signal, bail

**Cut Loss Hit Rate**:
The Hit Rate over every scored Signal that pointed against a losing Position, across all tickers and Intervals and at any Confidence. Measures whether cutting was right.
_Avoid_: Exit accuracy, cut rate
