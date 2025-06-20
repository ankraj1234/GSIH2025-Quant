# GSIH2025-Quant

This repository contains my solutions to the Goldman Sachs India Hackathon 2025 - Quant, where participants tackled real-world financial challenges involving portfolio hedging, adaptive market-making strategies, and exotic option pricing.

I secured Rank 8 overall.

---

## Problem Overview

### Problem 1: Optimal Hedging Strategy 
**Objective:** Hedge an unhedged portfolio using a set of equity stocks.  
**Goal:** Minimize Value at Risk (VaR) and hedging cost.

- Approach: Utilized LassoCV regression with customized enhancements to determine optimal hedging weights.
- This was the most approachable problem — many contestants achieved high scores during the contest.
- I was ranked 1st in this problem after post-contest evaluation, with a final score of 94.12.

---

### Problem 2: Automated Market Making
**Objective:** Build an adaptive quoting strategy using order book data, recent trades, and inventory levels.

- Inspired by the Avellaneda & Stoikov market-making framework.
- Method: Developed a dynamic market maker that adjusts bid/ask spreads in response to market volatility, order imbalance, and current inventory.
- Challenge: This was the most complex and demanding problem, requiring robust modeling and real-time decision-making.

---

### Problem 3: Exotic Option Pricing using Monte Carlo Simulation 
**Objective:** Price exotic European up-and-out basket options on three correlated assets using Monte Carlo simulation.  
**Bonus:** Calibrate local volatility surfaces using market data from vanilla call options.

- Constructed local volatility surfaces from vanilla option prices.
- Simulated correlated asset paths and implemented efficient Monte Carlo pricing with barrier condition checks.

---
