//! Checked concrete layouts. Unknown/generic/tensor layouts are distinct from overflow.
use super::{collect_value_structs, DefKind, DefMap, HashMap, HashSet, Symbol, Ty};

type Extent = Result<Option<(u64, u64)>, &'static str>;
#[derive(Default)]
pub(super) struct Cache(HashMap<Symbol, Extent>);

fn padded(size: u64, align: u64) -> Result<u64, &'static str> {
    size.div_ceil(align)
        .checked_mul(align)
        .ok_or("64-bit byte size")
}

impl Cache {
    /// Resolve named dependencies in postorder without growing the Rust call stack.
    /// Cycles retain an unknown entry; the existing E0402 pass diagnoses them.
    pub(super) fn prepare(&mut self, defs: &DefMap) {
        let mut graph = HashMap::default();
        for def in &defs.defs {
            let mut deps = Vec::new();
            match &def.kind {
                DefKind::Struct(fields) => {
                    for (_, ty) in fields {
                        collect_value_structs(ty, &mut deps);
                    }
                }
                DefKind::Enum(variants) => {
                    for variant in variants {
                        for ty in variant.payload.field_tys() {
                            collect_value_structs(&ty, &mut deps);
                        }
                    }
                }
                _ => continue,
            }
            graph.insert(def.name, deps);
            self.0.insert(def.name, Ok(None));
        }
        let mut visited = HashSet::default();
        let mut stack = Vec::new();
        for &root in graph.keys() {
            if !visited.insert(root) {
                continue;
            }
            stack.push((root, 0));
            while let Some((name, next)) = stack.last_mut() {
                let deps = graph.get(name).map_or(&[][..], Vec::as_slice);
                if let Some(&dep) = deps.get(*next) {
                    *next += 1;
                    if visited.insert(dep) {
                        stack.push((dep, 0));
                    }
                } else {
                    let name = *name;
                    stack.pop();
                    self.0.remove(&name);
                    let _ = self.extent(&Ty::Named(name), defs);
                }
            }
        }
    }

    pub(super) fn extent(&mut self, ty: &Ty, defs: &DefMap) -> Extent {
        match ty {
            Ty::Array { elem, len, .. } => {
                let Some((size, align)) = self.extent(elem, defs)? else {
                    return Ok(None);
                };
                Ok(Some((
                    size.checked_mul(*len).ok_or("64-bit byte size")?,
                    align,
                )))
            }
            Ty::Tuple(fields) => self.aggregate(fields.iter(), defs),
            Ty::Named(name) => {
                if let Some(result) = self.0.get(name) {
                    return *result;
                }
                // Recursive-by-value types already have E0402. Break cycles during recovery.
                self.0.insert(*name, Ok(None));
                let result = (|| match defs.lookup(*name).map(|d| &d.kind) {
                    Some(DefKind::Struct(fields)) => {
                        self.aggregate(fields.iter().map(|(_, ty)| ty), defs)
                    }
                    Some(DefKind::Enum(variants)) => {
                        let mut size = 0;
                        let mut align = 4;
                        for variant in variants {
                            let Some((sz, al)) =
                                self.aggregate(variant.payload.field_tys().iter(), defs)?
                            else {
                                return Ok(None);
                            };
                            size = size.max(sz);
                            align = align.max(al);
                        }
                        let size = padded(
                            padded(4, align)?
                                .checked_add(size)
                                .ok_or("64-bit byte size")?,
                            align,
                        )?;
                        Self::buffer(size, align)
                    }
                    _ => Ok(None),
                })();
                self.0.insert(*name, result);
                result
            }
            _ => Ok(ty.size_of().zip(ty.align_of())),
        }
    }

    fn aggregate<'a>(&mut self, fields: impl IntoIterator<Item = &'a Ty>, defs: &DefMap) -> Extent {
        let (mut size, mut align) = (0u64, 1u64);
        for ty in fields {
            let Some((sz, al)) = self.extent(ty, defs)? else {
                return Ok(None);
            };
            size = padded(size, al)?
                .checked_add(sz)
                .ok_or("64-bit byte size")?;
            align = align.max(al);
        }
        Self::buffer(padded(size, align)?, align)
    }

    fn buffer(size: u64, align: u64) -> Extent {
        // These aggregates lower to MIR [i8; size], whose element count is u32.
        if size > u32::MAX as u64 {
            return Err("32-bit aggregate byte-buffer size");
        }
        Ok(Some((size, align)))
    }
}
