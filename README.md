# TensRec
Tensorized Hypergraph Neural Network-based Approach for Third-Party Library (TPL) Recommendation

## Overview

**TensRec** is a novel Tensorized Hypergraph Neural Network (THNN)-based approach for recommending third-party libraries (TPLs) in Python software development. Unlike traditional pairwise models, TensRec captures complex higher-order interactions among projects and libraries through dual hypergraph structures and tensor operations. It is designed to provide accurate, diverse, and reliable recommendations while ensuring scalability for large-scale datasets.

## Features

- **Tensorized Hypergraph Modeling**: Captures multi-way interactions among projects and TPLs using dual hypergraphs.
- **Higher-Order Representation Learning**: Embeds nuanced co-usage patterns through tensor-based message passing and hyperedge aggregation.
- **Scalable and Sparse Tensor Operations**: Ensures low memory footprint and efficient computation even on large datasets.
- **Attention-Based Multi-Layer Aggregation**: Integrates local and global interaction information across layers to refine representations.
- **Hinge Loss Optimization**: Enhances the learning of ranking-based recommendations through a pairwise hinge loss function.
- **Empirical Evaluation**: Validated on a real-world dataset of 4,440 Python projects and 2,970 libraries, outperforming state-of-the-art models like PyRec and GRec.

## Getting Started

### Prerequisites

To run TensRec, install the following:

- Python 3.7+
- PyTorch 1.7+
- NumPy 1.18+
- SciPy 1.4+
- scikit-learn 0.22+
- tqdm
- pandas

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ABhiIITRoorkee/TensRec.git
   cd TensRec
