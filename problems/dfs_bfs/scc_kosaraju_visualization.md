# Kosaraju SCC Algorithm - Test Case Visualization

## Test Input
- **Vertices**: 5 (0-4)
- **Edges**: (1→3), (1→4), (2→1), (3→2), (4→5)
- **Expected SCC**: {1,2,3}, {4}, {5}

---

## 1. Original Graph

```mermaid
graph LR
    1 -->|edge| 3
    1 -->|edge| 4
    2 -->|edge| 1
    3 -->|edge| 2
    4 -->|edge| 5
    
    style 1 fill:#e1f5ff
    style 2 fill:#e1f5ff
    style 3 fill:#e1f5ff
    style 4 fill:#fff3e0
    style 5 fill:#f3e5f5
```

**Description**: The original directed graph with 5 vertices and edges forming a cycle between vertices 1, 2, 3.

---

## 2. Strongly Connected Components (SCCs)

```mermaid
graph LR
    subgraph SCC1["SCC 1: {1, 2, 3}"]
        1((1))
        2((2))
        3((3))
        1 -->|cycle| 3
        3 -->|cycle| 2
        2 -->|cycle| 1
    end
    
    subgraph SCC2["SCC 2: {4}"]
        4((4))
    end
    
    subgraph SCC3["SCC 3: {5}"]
        5((5))
    end
    
    SCC1 --> SCC2
    SCC2 --> SCC3
    
    style 1 fill:#b3e5fc
    style 2 fill:#b3e5fc
    style 3 fill:#b3e5fc
    style 4 fill:#ffe0b2
    style 5 fill:#f1c4f9
```

**Key Observations**:
- **SCC 1 {1, 2, 3}**: Forms a cycle - all three vertices are mutually reachable
- **SCC 2 {4}**: Single vertex, no cycle
- **SCC 3 {5}**: Single vertex, no cycle

---

## 3. SCC Condensation Graph (DAG)

```mermaid
graph LR
    A["SCC {1,2,3}<br/>SOURCE"]
    B["SCC {4}"]
    C["SCC {5}<br/>SINK"]
    
    A -->|from 1| B
    B -->|from 4| C
    
    style A fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style B fill:#ffecb3,stroke:#f57f17,stroke-width:2px
    style C fill:#f8bbd0,stroke:#c2185b,stroke-width:3px
```

**Graph Properties**:
- **Source SCC**: {1, 2, 3} - no incoming edges from other SCCs
- **Sink SCC**: {5} - no outgoing edges to other SCCs
- **Intermediate SCC**: {4} - has both incoming and outgoing edges to other SCCs
- **Structure**: Forms a linear DAG chain: SCC1 → SCC2 → SCC3

---

## 5. Algorithm Complexity

| Metric | Value |
|--------|-------|
| Time Complexity | O(V + E) |
| Space Complexity | O(V + E) |
| Number of DFS Passes | 2 |
| Number of SCCs | 3 |

---

## Summary

The Kosaraju algorithm successfully identifies:
- **3 Strongly Connected Components** from a graph with 5 vertices
- **Source**: Component {1,2,3} with no predecessors
- **Sink**: Component {5} with no successors
- **Cycle Detection**: Identifies the cycle 1→3→2→1 as a single connected component
