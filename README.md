# Evolutionary Federated Learning for Alzheimer’s Disease Diagnosis

## Overview
This repository contains the research code developed for my Master’s thesis titled  
**“An Evolutionary Federated Learning Approach to Diagnose Alzheimer’s Disease Under Uncertainty.”**

The project focuses on building an interpretable, uncertainty-aware machine learning pipeline
for medical imaging data, combining federated learning with belief rule-based expert systems
and evolutionary optimization.

## Problem Context
- Domain: Medical imaging and clinical decision support
- Challenge: Heterogeneous, imbalanced, and uncertain data from multiple sources
- Motivation: Enable collaborative learning without sharing sensitive medical data,
  while maintaining interpretability and robustness

## Methodology
The core components of the approach include:
- **Belief Rule-Based (BRB) Expert System** for knowledge-driven and interpretable reasoning
- **Federated Learning** to support privacy-preserving multi-client model training
- **Particle Swarm Optimization (PSO)** for tuning rule weights and parameters under uncertainty
- Explicit handling of **data imbalance and rare cases**, which are critical in disease diagnosis

## Code Structure
- `brb_PSO.py`  
  Implements PSO-based optimization for belief rule parameters

- `brb_fed.py`  
  Core federated learning logic integrating BRB models

- `brb_fl_client.py`  
  Client-side implementation for federated learning experiments

- `main.py`  
  Entry point for running experiments and evaluations

- `compile.py`  
  Utility functions for experiment setup and execution

## What This Project Demonstrates
- Design of end-to-end machine learning pipelines under uncertainty
- Integration of knowledge-based systems with data-driven learning
- Experience working with heterogeneous and imperfect real-world data
- Interpretable feature-level reasoning rather than black-box prediction
- Reproducible experimentation for applied research

## Relevance Beyond Healthcare
Although developed for medical imaging, the methods in this repository are transferable to
other imaging and sensing domains where uncertainty, limited labels, and multimodal data
are present, such as industrial X-ray inspection and computational imaging systems.

## Notes
This repository is research-oriented and prioritizes clarity and experimental transparency
over production-level optimization.
