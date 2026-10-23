# Stock Market MCP

A read-only server that gives coding agents market data, indicators, and advisory trading guidance. It never places orders.

## Language

### Signal

**Signal**:
An advisory answer to "is now the time to buy, sell, or hold this stock?", made of a Verdict, a confidence, a rationale, and the data date.
_Avoid_: Recommendation, call, alert

**Verdict**:
One of buy, sell, or hold – the direction the Signal leans.
_Avoid_: Decision, action

### Trade Plan

**Trade Plan**:
The three prices for one Direction: Entry, Stop, and Target. Advisory only.
_Avoid_: Setup, strategy, levels

**Direction**:
Long (profit if price rises) or short (profit if price falls).
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
