# Stock Market MCP

A server that gives coding agents market data, indicators, and advisory trading guidance. It never places orders.

## Language

### Signal

**Signal**:
An advisory answer to "is now the time to buy or hold this stock, or to cut it if held?", made of a Verdict, a Confidence, a rationale, the data date, the Price, the Hit Rate for its ticker, Interval and Verdict (none until enough are scored), and a Trade Plan when the Verdict is buy.
_Avoid_: Recommendation, call, alert

**Verdict**:
Buy or hold on a ticker not held; buy (add), hold, or cut on a held Position. Cut means exit now, whatever the Unrealised P&L. There is no sell or short.
_Avoid_: Decision, action

**Confidence**:
The classifier's own 0–1 estimate that the Verdict is right. Self-reported, not a measured hit rate.
_Avoid_: Probability, certainty, score

**Minimum Confidence**:
The lowest Confidence at which a buy gets a Trade Plan or a cut counts. Set by the server from buy Hit Rates: 0.6 until enough Signals are scored, and none when enough are scored but no Confidence band beats break-even, so no buy gets a Trade Plan.
_Avoid_: Threshold, cut-off

**Price**:
The last close in the data the Signal was made from; during the session that is the close of the bar still forming. Every Signal has one; on buy it equals the Entry.
_Avoid_: Current price, last price, quote

**Outcome**:
What happened to a Signal within 20 bars of its Interval, scored even when it was below the Minimum Confidence. A buy is judged against the default Trade Plan: Target (reached first), Stop (reached first, or on the same bar as the Target), or Expired (neither). A cut is judged against its Buyback Price and Cut Stop: Target (Buyback Price reached first), Stop (Cut Stop reached first, or on the same bar), or Expired (neither); a cut missing either is never scored. A hold's Outcome is Quiet (price moved the Target distance neither up nor down) or Missed (it did).
_Avoid_: Result, win/loss

**Hit Rate**:
For buy and cut, Targets divided by Targets plus Stops over a set of scored Signals, Expired left out; for hold, Quiet divided by all scored holds. The measured counterpart to Confidence.
_Avoid_: Win rate, accuracy

**Track Record**:
The Outcome counts and Hit Rates of the current prompt version's Signals, by Confidence band and Verdict, optionally for one ticker, one Interval, or both. Shows how Confidence compares with Hit Rate and why the Minimum Confidence sits where it does.
_Avoid_: Stats, performance, backtest, scorecard

**Interval**:
The length of one bar in the price history a Signal is made from: 1m, 5m, 15m, 1h, or 1d.
_Avoid_: Timeframe, resolution, granularity

**Prompt Version**:
The number stored with each Signal that names what the classifier was shown and asked: the bars and indicators, the Position, the previous Signal, and the questions. It changes whenever any of that changes, so Hit Rates and the Minimum Confidence count only Signals asked the same way.
_Avoid_: Model version, schema version

### Trade Plan

**Trade Plan**:
The three prices for a long trade: Entry, Stop, and Target. Part of a buy Signal at or above the Minimum Confidence; every other Signal has none. Advisory only.
_Avoid_: Setup, strategy, levels

**Entry**:
The price at which the Trade Plan opens the position.
_Avoid_: Buy price, trigger

**Stop**:
The price at which the Trade Plan is wrong and the position should close at a loss. Below Entry.
_Avoid_: Stop-loss price, invalidation

**Target**:
The price at which the Trade Plan takes profit. Above Entry.
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
A long stake in one ticker in the Portfolio, with a quantity and an average cost. There are no short Positions.
_Avoid_: Holding, lot

**Unrealised P&L**:
A Position's profit or loss if closed at a given price: the price minus the average cost, times the quantity. In the ticker's currency, also given as a percent of cost.
_Avoid_: Gain, return, open P&L

**Cut Loss**:
A cut Verdict at or above the Minimum Confidence while the Position's Unrealised P&L at the Price is negative. Advisory only; the server never closes the Position.
_Avoid_: Stop out, exit signal, bail

**Cut Loss Hit Rate**:
The Hit Rate over every scored cut on a losing Position, across all tickers and Intervals and at any Confidence. Measures whether cutting was right. Reported only; it sets nothing.
_Avoid_: Exit accuracy, cut rate

**Buyback Price**:
The level below a cut's Price at which getting back in is cheaper, picked by the classifier from chart levels below the Price (the 20-bar and 50-bar lows, the lower Bollinger Band, the 10-bar EMA, the SMA, VWAP), or none when no level fits. Given on a cut at or above the Minimum Confidence, in that Signal. Reaching it first makes the cut's Outcome a Target; otherwise it changes nothing.
_Avoid_: Re-entry target, short target, buy-the-dip price

**Cut Stop**:
The level above a cut's Price at which the cut was wrong, picked by the classifier from the same kinds of chart levels above the Price (the 20-bar and 50-bar highs, the upper Bollinger Band, the 10-bar EMA, the SMA, VWAP), or none when no level fits. Given with the Buyback Price. Reaching it first makes the cut's Outcome a Stop. Not the Trade Plan's Stop.
_Avoid_: Invalidation, re-entry price

**Buyback Quantity**:
The whole number of shares a cut's proceeds buy at the Buyback Price: the Position's quantity times the cut's Price, divided by the Buyback Price, rounded down. Fees left out. Given with the Buyback Price.
_Avoid_: Re-entry size, position size
