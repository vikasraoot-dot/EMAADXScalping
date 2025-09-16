# EMAMerged (EMA Crossover + RSI + ADX/+DI + EMA Slope) — Live Paper

Live paper-trading strategy using Alpaca:
- **Entry**: EMA9 crosses above EMA21 **AND**
  - ADX ≥ 22 and +DI > −DI
  - RSI in [50, 80]
  - EMA9 slope (3-bar) ≥ 12 bps; EMA21 slope > 0
- **Exit**: (logged) EMA9 crosses below EMA21 (add sell logic if desired)

## Setup

1) Set env on your Windows self-hosted runner (as system env or repo secrets):
