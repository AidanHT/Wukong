//! Linear-time dependency-cycle classification without recursive graph walks.

use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Symbol};

/// Return exactly the vertices lying on a directed cycle. A vertex that merely reaches a cycle
/// is excluded: the semantic diagnostics name definitions that depend on themselves.
///
/// Kosaraju's two passes visit each vertex and edge a bounded number of times. Explicit stacks
/// also keep a long chain of otherwise shallow declarations off the compiler's call stack.
/// Hash iteration affects only traversal order; callers emit diagnostics in source order.
pub(crate) fn cyclic_nodes(graph: &HashMap<Symbol, Vec<Symbol>>) -> HashSet<Symbol> {
    let mut reverse: HashMap<Symbol, Vec<Symbol>> = HashMap::default();
    for (&node, deps) in graph {
        reverse.entry(node).or_default();
        for &dep in deps {
            reverse.entry(dep).or_default().push(node);
        }
    }

    let mut visited = HashSet::default();
    let mut finished = Vec::with_capacity(reverse.len());
    let mut stack = Vec::new();
    for &root in graph.keys() {
        if !visited.insert(root) {
            continue;
        }
        stack.push((root, 0));
        while let Some((node, next)) = stack.last_mut() {
            let deps = graph.get(node).map_or(&[][..], Vec::as_slice);
            if let Some(&dep) = deps.get(*next) {
                *next += 1;
                if visited.insert(dep) {
                    stack.push((dep, 0));
                }
            } else {
                finished.push(*node);
                stack.pop();
            }
        }
    }

    visited.clear();
    let mut cyclic = HashSet::default();
    let mut pending = Vec::new();
    let mut component = Vec::new();
    for root in finished.into_iter().rev() {
        if !visited.insert(root) {
            continue;
        }
        pending.push(root);
        component.clear();
        while let Some(node) = pending.pop() {
            component.push(node);
            for &dep in &reverse[&node] {
                if visited.insert(dep) {
                    pending.push(dep);
                }
            }
        }
        if component.len() > 1 || graph.get(&root).is_some_and(|deps| deps.contains(&root)) {
            cyclic.extend(component.iter().copied());
        }
    }
    cyclic
}
