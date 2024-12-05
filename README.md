# OTC Accumulator Valuation Tool

## Overview

This repository contains a Python-based valuation tool for pricing an **OTC Accumulator** product. The tool implements Monte Carlo simulations to price the accumulator under different equity models (e.g., Dupire Local Volatility Model, Black-Scholes with constant volatility) and yield curve assumptions (e.g., derived from futures or constant rates). The tool allows for flexible configuration of key parameters such as Forward Price, Gearing, Knock-Out level, Guaranteed Period, and more.

The accumulator product is structured as follows:

### **Term Sheet**

- **Valuation Date / Maturity Date**: 3 months after the Trade Date (13 weeks).
- **Accumulation Periods**: Weekly (settlement price determined every weekend).
- **Shares per Day**: 1 share by default.
- **Gearing**: 2x (if the settlement price is below the FP, the accumulation will double to 2 shares per day).

#### **Daily Share Accumulation Rules**:
- If the settlement price is **greater than or equal to** the Forward Price (FP):  
  Daily share accumulation = 1 share.  
- If the settlement price is **lower than** the Forward Price (FP):  
  Daily share accumulation = 2 shares (gearing applied).

#### **Knock-Out (KO)**:
- KO Price: **110% of the initial spot price**.  
  - If the settlement price is **greater than or equal to the KO price**, the accumulator terminates immediately.  
  - Any shares accumulated **before the KO trigger** will still be delivered to the investor.  
- If KO is **NOT triggered**, the investor will continue accumulating shares until the maturity date.

#### **Guaranteed Period**:
- **Duration**: 3 weeks (21 days).  
  - If KO occurs during this period, the bank will sell the investor shares up to the Guaranteed Period in addition to the shares accumulated before the KO event.  
  - Outside the Guaranteed Period, the accumulation stops immediately upon KO.

---

## Features

### **1. Pricing Models**
The tool provides two pricing approaches:
- **Example 1**: Pricing with a yield curve derived from futures and the **Dupire Local Volatility Model**.
- **Example 2**: Pricing with a constant volatility and a constant risk-free rate under the **Black-Scholes Model**.

### **2. Simulation Parameters**
- **Monte Carlo Simulation**: The tool supports generating price paths using Monte Carlo simulations with support for variance reduction via antithetic variates.
- **Path Dependence**: Full support for path-dependent features such as weekly settlement prices and knock-out events.
- **Guaranteed Period Logic**: Incorporates the additional accumulation guarantee for the first 3 weeks (21 days).

---

## Code Structure

The implementation is split into modular components for flexibility and scalability:

### **Files**
1. **`main.py`**: The main script to run pricing examples and configure input settings for the accumulator.
2. **`Accumulator.py`**: Contains the `Accumulator_MC` class, which models the accumulator product and calculates its fair price.
3. **`EQModel.py`**: Implements equity models such as the **Dupire Local Volatility Model** and **Black-Scholes with constant volatility**.
4. **`YieldCurve.py`**: Defines yield curve classes:
   - **`YieldCurve_from_Futures`**: Interpolates the yield curve from futures prices.
   - **`YieldCurve_Constant`**: Uses a constant risk-free rate.
5. **`util.py`**: Utility functions, including a timing decorator to measure function execution time.

---

## Usage

### **Setup**
1. Clone the repository and install required dependencies:
    ```bash
    git clone <repo-url>
    cd <repo-directory>
    pip install -r requirements.txt 
    ```

2. Ensure the input data files are present in the `data/` directory:
   - **`futures_input.csv`**: Contains futures prices for yield curve construction.
   - **`vol_input.csv`**: Contains implied volatility surface data.

---

## Key Classes and Methods

### **`Accumulator_MC`**
- **Purpose**: Models the accumulator product and calculates the fair Forward Price (FP).
- **Key Methods**:
  - `get_fp()`: Calculates the fair FP using bisection method.
  - `get_pv(fp)`: Computes the present value of the accumulator given a forward price.

### **`EQModel`**
- **Purpose**: Abstract base class for equity models.
- **Implementations**:
  - `DupireModel`: Local volatility model with implied volatility surface fitting.
  - `BlackScholesConstantVolModel`: Simplified model with constant volatility.

### **`YieldCurve`**
- **Purpose**: Abstract base class for yield curves.
- **Implementations**:
  - `YieldCurve_from_Futures`: Interpolates rates from futures prices.
  - `YieldCurve_Constant`: Assumes a constant zero rate.

---

## Dependencies

- Python 3.8+
- Required libraries: `numpy`, `pandas`, `scipy`, `matplotlib`, `numdifftools`