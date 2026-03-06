use crate::lattice::Lattice;

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

        Ok(Self { n_nodes: max_node + 1, edges })
    }

    /// Load from JSON adjacency list.
    ///
    /// Format: {"n_nodes": 1000, "edges": [[0,1],[0,2],[1,3],...]}
    ///
    /// Uses hand-rolled parsing to avoid adding serde as a dependency.
    pub fn from_json(content: &str) -> anyhow::Result<Self> {
        let n_nodes: usize = {
            let key = "\"n_nodes\"";
            let pos = content.find(key).ok_or_else(|| anyhow::anyhow!("missing n_nodes"))?;
            let after = &content[pos + key.len()..];
            let colon = after.find(':').ok_or_else(|| anyhow::anyhow!("missing : after n_nodes"))?;
            let num_str = after[colon+1..].trim_start();
            let end = num_str.find(|c: char| !c.is_ascii_digit()).unwrap_or(num_str.len());
            num_str[..end].parse()?
        };

        let edges_key = "\"edges\"";
        let pos = content.find(edges_key).ok_or_else(|| anyhow::anyhow!("missing edges"))?;
        let after = &content[pos + edges_key.len()..];
        let bracket = after.find('[').ok_or_else(|| anyhow::anyhow!("missing [ after edges"))?;
        let array_str = &after[bracket..];
        let close = Self::find_matching_bracket(array_str)?;
        let inner = &array_str[1..close];

        let mut edges = Vec::new();
        let mut chars = inner.chars().peekable();
        loop {
            while chars.peek().map_or(false, |&c| c != '[') { chars.next(); }
            if chars.next().is_none() { break; }
            let i_str: String = chars.by_ref().take_while(|c| c.is_ascii_digit()).collect();
            if i_str.is_empty() { break; }
            while chars.peek().map_or(false, |&c| c != ',' && !c.is_ascii_digit()) { chars.next(); }
            if chars.peek() == Some(&',') { chars.next(); }
            let j_str: String = chars.by_ref()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if j_str.is_empty() { break; }
            edges.push((i_str.parse::<usize>()?, j_str.parse::<usize>()?));
        }

        Ok(Self { n_nodes, edges })
    }

    fn find_matching_bracket(s: &str) -> anyhow::Result<usize> {
        let mut depth = 0i32;
        for (i, c) in s.chars().enumerate() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 { return Ok(i); }
                }
                _ => {}
            }
        }
        anyhow::bail!("unmatched bracket")
    }

    /// Convert to Lattice for simulation.
    pub fn into_lattice(self) -> Lattice {
        Lattice::from_edges(self.n_nodes, &self.edges)
    }
}
