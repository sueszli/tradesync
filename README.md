```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣤⣤⣤⣤⣀⣀⣀⡀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⣠⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣄⠀████████╗██████╗  █████╗ ██████╗ ███████╗⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠙⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠀╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣿⣶⣤⣄⣉⣉⠙⠛⠛⠛⠛⠛⠛⠋⣉⣉⣠⣤⣶⣿⠀   ██║   ██████╔╝███████║██║  ██║█████╗  
⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  
⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀   ██║   ██║  ██║██║  ██║██████╔╝███████╗
⠀⠀⠀⠀⠀⣄⡉⠛⠻⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⢉⣠⠀   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝
⠀⠀⠀⠀⠀⣿⣿⣿⣶⣶⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣶⣶⣿⣿⣿⠀  ███████╗██╗   ██╗███╗   ██╗ ██████╗      
⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀  ██╔════╝╚██╗ ██╔╝████╗  ██║██╔════╝      
⠀⠀⠀⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀  ███████╗ ╚████╔╝ ██╔██╗ ██║██║           
⠀⠀⠀⠀⠀⣶⣤⣈⡉⠛⠛⠻⠿⠿⠿⠿⠿⠿⠟⠛⠛⢉⣁⣤⣶⠀  ╚════██║  ╚██╔╝  ██║╚██╗██║██║           
⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣶⣶⣶⣶⣾⣿⣿⣿⣿⣿⠀  ███████║   ██║   ██║ ╚████║╚██████╗      
⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝      
⠀⠀⠀⠀⠀⠙⠻⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠋⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠛⠛⠛⠛⠉⠉⠉⠁⠀⠀⠀⠀⠀
```

# Usage

```bash
make help

make docker
docker compose exec main pip install polars-lts-cpu
docker compose exec main python3.11 \
    ./src/sync.py \
    --transfers ./data/transfers.xml \
    --accounts ./data/accounts.json \
    --output ./data/output.xml \
    --pedantic

# 2025-08-23 18:48:49.537 | INFO     | validate:validate_xml_structure:38 - Validated XML structure for: data/transfers.xml
# 2025-08-23 18:48:49.911 | INFO     | validate:validate_json_structure:58 - Validated JSON structure for: data/accounts.json
# 2025-08-23 18:48:49.911 | INFO     | __main__:run:152 - Files loaded
# 2025-08-23 18:49:00.892 | INFO     | __main__:run:155 - Found 1162523 initial bank transfers
# 100%|███████████████████████████████████████████████████████████████████████████████████| 5206/5206 [00:06<00:00, 820.25it/s]
# 2025-08-23 18:49:07.292 | INFO     | __main__:run:158 - Optimized to 6621 bank transfers (Removed 99.43%)
# 2025-08-23 18:49:07.352 | INFO     | validate:validate_optimization:86 - Net account balances are preserved correctly!
```

# Design Rationale

In this transfer optimization problem you've got this set $T = \{(p,a,b,q) \mid p \in \text{Products}, a \to b \in \text{Accounts}, q \in \mathbb{R}^+\}$ and you want to shrink it to $T'$ while keeping all the account balances intact. For every product-account pair $(p,a)$, the total inflows $\sum_{(p,\_,a,q) \in T} q$ and outflows $\sum_{(p,a,\_,q) \in T} q$ have to match between $T$ and $T'$.

The solution uses three optimization passes. Flow aggregation merges identical transfers - multiple $(p,a,b,q_i)$ tuples with the same route collapse into $(p,a,b,\sum q_i)$. Bidirectional netting handles flows going both ways between accounts - replace $(p,a,b,q_{ab})$ and $(p,b,a,q_{ba})$ with $(p,a,b,\max(0,q_{ab}-q_{ba}))$. The interesting part is cycle elimination. For each product $p$, I build a directed graph $G_p = (V_p, E_p)$ where vertices are accounts and edges $(a,b,q)$ represent transfers. I hunt for cycles $C = [a_1 \to a_2 \to \ldots \to a_k \to a_1]$ and drain them by the bottleneck flow $q_{\text{min}} = \min_{i} q_{a_i \to a_{i+1}}$. Each edge in the cycle gets reduced by $q_{\text{min}}$, and zero-weight edges get pruned.

Since input files cap out around 16GB, everything fits in memory on consumer hardware. I used Polars for data wrangling. The lazy evaluation and columnar operations make aggregation really fast, compared to Pandas and I won't need to write custom CPython extensions. NetworkX handles graph operations. I bounded cycle search depth at 3 because transfer cycles rarely go beyond 4 hops, keeping me in polynomial territory while catching most optimization opportunities.

Apache Arrow integration through Polars was clutch for handling heterogeneous input formats. Zero-copy conversions between JSON, XML, and native Arrow formats meant I could prototype against different data sources without rewriting parsers.

I put a lot of time into testing because financial data is unforgiving. TDD, unit tests for individual passes, integration tests for the full pipeline, and regression tests for specific bug fixes. Property-based testing with Hypothesis generates random transfer sets and verifies conservation constraints $\sum \text{inflows} = \sum \text{outflows}$ hold across thousands of synthetic datasets.
