# Schrödinger Bridge for ETFs

Entropic optimal transport and stochastic bridge matching between market regimes. Learns a path‑space coupling (Schrödinger bridge) that interpolates between start and end return distributions. The bridge can be used for generative simulation of market trajectories.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63, 252, 504, 1008, 2016, 4032, 4536 days)
- Sinkhorn algorithm with entropic regularization
- Per‑ETF score = total mass sent from that asset in the optimal coupling (source centrality)
- Best window selected automatically (largest absolute raw signal)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-schrodinger-bridge-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- The Schrödinger bridge gives the most likely stochastic evolution between the observed marginal distributions.
- ETFs that send large probability mass to others are "source hubs" – their relative return changes are strongly coupled to the rest.
- This signal can be used for regime detection, portfolio allocation, or generative scenario generation.

## Requirements

See `requirements.txt`.
