# Stock Market MCP

A read-only server that gives coding agents market data, indicators, and advisory trading guidance. It never places orders.

## Language

### Signal

**Signal**:
An advisory answer to "is now the time to buy, sell, or hold this stock?", made of a Verdict, a confidence, a rationale, the data date, the Price, and a Trade Plan when the Verdict is buy or sell.
_Avoid_: Recommendation, call, alert

**Verdict**:
One of buy, sell, or hold – the direction the Signal leans.
_Avoid_: Decision, action

**Price**:
The last close in the data the Signal was made from. Every Signal has one; on buy or sell it equals the Entry.
_Avoid_: Current price, last price, quote

### Trade Plan

**Trade Plan**:
The three prices for one Direction: Entry, Stop, and Target. Part of a Signal; a hold Verdict has none. Advisory only.
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
