use crate::lattice::Lattice;
use serde_json::Value;

/// A graph definition as a node count + edge list.
pub struct GraphDef {
    pub n_nodes: usize,
    pub edges: Vec<(usize, usize)>,
}

impl GraphDef {
    /// Load from edge list CSV.
    ///
    /// Format (lines starting with # are ignored):
    ///   0,1
    ///   0,2
    ///   1,3
    pub fn from_edge_csv(content: &str) -> anyhow::Result<Self> {
        let mut edges = Vec::new();
        let mut max_node = 0usize;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split(',').collect();
            if parts.len() < 2 {
                anyhow::bail!("bad edge line: {trimmed}");
            }
            let i: usize = parts[0].trim().parse()?;
            let j: usize = parts[1].trim().parse()?;
            max_node = max_node.max(i).max(j);
            edges.push((i, j));
        }

        Ok(Self {
            n_nodes: max_node + 1,
            edges,
        })
    }

    /// Load from JSON adjacency list.
    ///
    /// Format: {"n_nodes": 1000, "edges": [[0,1],[0,2],[1,3],...]}
    pub fn from_json(content: &str) -> anyhow::Result<Self> {
        let root: Value = serde_json::from_str(content)?;
        let n_nodes = root
            .get("n_nodes")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("missing n_nodes"))? as usize;

        let edge_values = root
            .get("edges")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("missing edges"))?;

        let mut edges = Vec::with_capacity(edge_values.len());
        for edge in edge_values {
            let pair = edge
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("edge entry is not a 2-element array"))?;
            if pair.len() != 2 {
                anyhow::bail!("edge entry must have exactly 2 indices");
            }
            let i = pair[0]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("edge index is not an integer"))?
                as usize;
            let j = pair[1]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("edge index is not an integer"))?
                as usize;
            edges.push((i, j));
        }

        Ok(Self { n_nodes, edges })
    }

    /// Convert to Lattice for simulation.
    pub fn into_lattice(self) -> Lattice {
        Lattice::from_edges(self.n_nodes, &self.edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_edge_csv() {
        let csv = "# comment\n0,1\n1,2\n0,2\n";
        let g = GraphDef::from_edge_csv(csv).unwrap();
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.edges.len(), 3);
        assert!(g.edges.contains(&(0, 1)));
    }

    #[test]
    fn parse_edge_csv_skips_blanks() {
        let csv = "\n0,1\n\n1,2\n\n";
        let g = GraphDef::from_edge_csv(csv).unwrap();
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn parse_json_graph() {
        let json = r#"{"n_nodes": 4, "edges": [[0,1],[1,2],[2,3],[3,0]]}"#;
        let g = GraphDef::from_json(json).unwrap();
        assert_eq!(g.n_nodes, 4);
        assert_eq!(g.edges.len(), 4);
        assert!(g.edges.contains(&(2, 3)));
    }

    #[test]
    fn parse_json_graph_with_metadata_and_whitespace() {
        let json = r#"{
  "edges": [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
  ],
  "metadata": {
    "generator": "test",
    "realization": 0
  },
  "n_nodes": 4
}"#;
        let g = GraphDef::from_json(json).unwrap();
        assert_eq!(g.n_nodes, 4);
        assert_eq!(g.edges.len(), 4);
        assert!(g.edges.contains(&(3, 0)));
    }

    #[test]
    fn json_to_lattice_coordination() {
        // Complete graph K4: every node connected to every other
        let json = r#"{"n_nodes": 4, "edges": [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}"#;
        let g = GraphDef::from_json(json).unwrap();
        let lat = g.into_lattice();
        assert_eq!(lat.size(), 4);
        for nb in &lat.neighbours {
            assert_eq!(nb.len(), 3, "K4 should have z=3");
        }
    }

    #[test]
    fn csv_bad_line_errors() {
        let csv = "0\n"; // only one node per line
        assert!(GraphDef::from_edge_csv(csv).is_err());
    }
}
