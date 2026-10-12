//! Deterministic parallelism for the optimizer's per-function phases.
//!
//! Most of what [`crate::optimize`] does is per function: the fixpoint pipeline runs on one
//! [`Function`] at a time with nothing but that function and the call-effect table in hand, and
//! loop unrolling, straight-line SLP, their cleanup sweeps and switch formation each look at one
//! function only. Those phases run here across a private thread pool; inlining, call-graph
//! construction and the call-effect summaries stay on the calling thread, in program order.
//!
//! # Why the output cannot depend on the thread count
//!
//! A task is a pure function of its inputs: the function's own body and, for the fixpoint, the
//! call-effect table. The table is frozen while a batch runs, every task reads it through a shared
//! `Arc`, and a task touches no other function. A pass cannot even *name* a function its body did
//! not already call, because no pass holds the interner a new symbol would need. So the order in
//! which tasks run — and which thread runs them — cannot change what any one of them produces, and
//! results are put back in their sequential order before anything reads them. What differs between
//! thread counts is only *when* each function is optimized, never *what it becomes*: a batch's MIR
//! is byte-identical for every thread count, which `wukongc`'s determinism gate checks over the
//! whole corpus (one, the automatic count, and four threads).
//!
//! Failures keep their sequential identity too. Under the sequential schedule the first task to
//! fail (a verifier failure, or a panic) stops the batch; here every task runs, and the batch then
//! reports the failure — or resumes the panic — of the earliest task in sequential order, ignoring
//! whatever any later task did. (A later task's panic message may still reach stderr from its
//! worker; only the earliest one is propagated.)
//!
//! # The knob
//!
//! [`crate::OptimizeOptions::threads`] picks the thread count; `0`, its default, defers to the
//! `WUKONG_OPT_THREADS` environment variable, and an unset, `0` or unparseable variable means
//! *automatic*: the host's parallelism, capped at [`MAX_WORKERS`], used only for a batch with
//! enough spare work to pay for the hand-off ([`AUTO_MIN_SPARE_INSTS`]). An explicit count of `1`
//! is the sequential schedule exactly as it was before this module existed (same order of calls,
//! no pool). An explicit count above `1` takes the parallel path for every batch of two or more
//! functions, however small — what a test of the parallel path needs. The tracing switches
//! (`WUKONG_VEC_TRACE`, `WUKONG_SLP_TRACE`) force the sequential schedule, so a trace prints in
//! program order.
//!
//! # The pool
//!
//! One private `rayon` pool per process, [`MAX_WORKERS`] threads at most, built on first use and
//! never the global one: the runtime configures the *global* pool itself (16 MiB worker stacks for
//! `@parallel` region bodies) and must win that race, so the optimizer never touches it. Its
//! workers get the stack the compile thread runs on (`wukongc` spawns it with 256 MiB), so a
//! function deep enough to need that stack sequentially still fits on a worker. An explicit count
//! above the pool's width shares its threads. A pool that cannot be built leaves every batch on the
//! calling thread, which computes the same MIR.

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

use wukong_mir::Function;

/// The environment variable read when [`crate::OptimizeOptions::threads`] is `0`.
pub(crate) const THREADS_ENV: &str = "WUKONG_OPT_THREADS";

/// The most threads the automatic setting asks for, and the widest the pool is built. The
/// per-function phases rarely have more independent functions than this in one batch, and every
/// extra worker is another 256 MiB stack reservation.
pub(crate) const MAX_WORKERS: usize = 8;

/// Under the automatic setting, a batch goes to the pool only when the functions *other than its
/// largest* hold at least this many instructions between them — the work other threads can take
/// while the largest one runs. Below it the hand-off (and, the first time, building the pool) costs
/// more than it saves, and the batch runs on the calling thread. The value is set from debug-build
/// timings (the pipeline spends on the order of a hundred microseconds per instruction there, so
/// this much spare work is tens of milliseconds, against far less to hand a batch to the pool); a
/// release build is faster per instruction, and a release measurement is what should tune it.
pub(crate) const AUTO_MIN_SPARE_INSTS: usize = 256;

/// The stack each pool worker gets: the size `wukongc` gives its compile thread.
const WORKER_STACK: usize = 256 << 20;

/// How many threads one optimizer run may use, resolved once from the options and environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Parallelism {
    /// At most this many tasks run at once; `1` is the sequential schedule.
    workers: usize,
    /// An explicit count: every batch of two or more tasks goes to the pool, with no work estimate.
    forced: bool,
}

impl Parallelism {
    /// The schedule the optimizer had before it ran anything in parallel.
    pub(crate) const SEQUENTIAL: Parallelism = Parallelism {
        workers: 1,
        forced: true,
    };

    /// Resolve a requested thread count: `requested` when non-zero, else `WUKONG_OPT_THREADS`, else
    /// automatic (see the module note).
    pub(crate) fn resolve(requested: u32) -> Parallelism {
        if tracing() {
            return Parallelism::SEQUENTIAL;
        }
        let explicit = match requested {
            0 => std::env::var(THREADS_ENV)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0),
            n => Some(n as usize),
        };
        match explicit {
            Some(n) => Parallelism {
                workers: n,
                forced: true,
            },
            None => Parallelism {
                workers: host_parallelism().min(MAX_WORKERS),
                forced: false,
            },
        }
    }

    /// One thread: run the old schedule, in the old order.
    pub(crate) fn is_sequential(self) -> bool {
        self.workers <= 1
    }

    /// Should a batch whose tasks weigh `weights` go to the pool?
    fn engages(self, weights: &[usize]) -> bool {
        if self.workers <= 1 || weights.len() < 2 {
            return false;
        }
        if self.forced {
            return true;
        }
        let total: usize = weights.iter().sum();
        let largest = weights.iter().copied().max().unwrap_or(0);
        total - largest >= AUTO_MIN_SPARE_INSTS
    }
}

/// Whether a tracing switch that prints from inside a per-function phase is set: such a trace must
/// come out in program order, so it pins the sequential schedule.
fn tracing() -> bool {
    std::env::var_os("WUKONG_VEC_TRACE").is_some() || std::env::var_os("WUKONG_SLP_TRACE").is_some()
}

fn host_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// The private pool, built on first use (`None` when the host refuses the threads).
fn pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(host_parallelism().clamp(2, MAX_WORKERS))
            .stack_size(WORKER_STACK)
            .thread_name(|i| format!("wukong-opt-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// The size estimate a task is scheduled by: its instruction count.
fn weight(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

/// One task's state while a batch runs.
enum Slot<'a, T> {
    Pending(&'a mut Function),
    Running,
    Done(std::thread::Result<T>),
}

/// Apply `task` to each of `funcs` and return the results in `funcs` order, as the sequential
/// schedule would: up to and including the first `Err`, after which nothing more is reported (the
/// sequential schedule never ran the rest). With `par` engaged for this batch the calling thread and
/// up to `workers - 1` pool threads take tasks off one queue, largest first — the caller starts
/// before any worker has woken, so the largest task, the one that decides how long the batch takes,
/// usually stays on the thread that was already running; a panic in a task is resumed here, the
/// earliest in `funcs` order, after the whole batch has finished. Also returns how many threads the
/// batch was given (1: the caller's alone).
pub(crate) fn try_map<R, E, F>(
    par: Parallelism,
    funcs: Vec<&mut Function>,
    task: F,
) -> (Vec<Result<R, E>>, usize)
where
    R: Send,
    E: Send,
    F: Fn(&mut Function) -> Result<R, E> + Sync,
{
    let weights: Vec<usize> = funcs.iter().map(|f| weight(f)).collect();
    let pool = if par.engages(&weights) { pool() } else { None };
    let Some(pool) = pool else {
        let mut out = Vec::with_capacity(funcs.len());
        for f in funcs {
            let r = task(f);
            let failed = r.is_err();
            out.push(r);
            if failed {
                break;
            }
        }
        return (out, 1);
    };

    // Largest first, so the longest task is not the one left running alone at the end; ties keep
    // program order. Only the start order depends on this — every result lands in its own slot.
    let mut order: Vec<usize> = (0..funcs.len()).collect();
    order.sort_by(|&a, &b| weights[b].cmp(&weights[a]).then(a.cmp(&b)));
    let slots: Vec<Mutex<Slot<'_, Result<R, E>>>> = funcs
        .into_iter()
        .map(|f| Mutex::new(Slot::Pending(f)))
        .collect();
    let next = AtomicUsize::new(0);
    let jobs = par.workers.min(slots.len());
    let work = || loop {
        let k = next.fetch_add(1, Ordering::Relaxed);
        let Some(&i) = order.get(k) else {
            break;
        };
        let mut slot = slots[i].lock().unwrap_or_else(PoisonError::into_inner);
        let Slot::Pending(f) = std::mem::replace(&mut *slot, Slot::Running) else {
            unreachable!("each task is handed out once");
        };
        *slot = Slot::Done(panic::catch_unwind(AssertUnwindSafe(|| task(f))));
    };
    pool.in_place_scope(|scope| {
        for _ in 1..jobs {
            scope.spawn(|_| work());
        }
        work();
    });

    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        match slot.into_inner().unwrap_or_else(PoisonError::into_inner) {
            Slot::Done(Ok(r)) => {
                let failed = r.is_err();
                out.push(r);
                if failed {
                    break;
                }
            }
            Slot::Done(Err(payload)) => panic::resume_unwind(payload),
            Slot::Pending(_) | Slot::Running => unreachable!("the scope ran every task"),
        }
    }
    (out, jobs)
}

/// [`try_map`] for a task that cannot fail.
pub(crate) fn map<R, F>(par: Parallelism, funcs: Vec<&mut Function>, task: F) -> Vec<R>
where
    R: Send,
    F: Fn(&mut Function) -> R + Sync,
{
    try_map::<R, std::convert::Infallible, _>(par, funcs, |f| Ok(task(f)))
        .0
        .into_iter()
        .map(|r| r.unwrap_or_else(|never| match never {}))
        .collect()
}

/// Mutable references to `funcs[i]` for each `i` of `indices`, in that order. Each index may appear
/// once.
pub(crate) fn select_mut<'a>(
    funcs: &'a mut [Function],
    indices: &[usize],
) -> Vec<&'a mut Function> {
    let mut all: Vec<Option<&'a mut Function>> = funcs.iter_mut().map(Some).collect();
    indices
        .iter()
        .map(|&i| {
            all[i]
                .take()
                .expect("a function index appears once per batch")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{Builder, MirType, Op};
    use wukong_span::Interner;

    fn func(it: &mut Interner, name: &str, insts: usize) -> Function {
        let mut b = Builder::new(it.intern(name), MirType::I64);
        let mut v = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        for _ in 1..insts {
            v = b.build(MirType::I64, Op::Bin(wukong_mir::BinOp::Add, v, v));
        }
        b.ret(Some(v));
        b.finish()
    }

    #[test]
    fn explicit_counts_win_and_one_is_sequential() {
        assert!(Parallelism::resolve(1).is_sequential());
        let four = Parallelism::resolve(4);
        // A tracing switch in the environment pins the sequential schedule; otherwise four it is.
        if !tracing() {
            assert_eq!(
                four,
                Parallelism {
                    workers: 4,
                    forced: true
                }
            );
        }
        assert!(Parallelism::SEQUENTIAL.is_sequential());
    }

    #[test]
    fn the_automatic_setting_waits_for_spare_work() {
        let auto = Parallelism {
            workers: 4,
            forced: false,
        };
        assert!(
            !auto.engages(&[100_000]),
            "one task has no one to share with"
        );
        assert!(
            !auto.engages(&[100_000, AUTO_MIN_SPARE_INSTS - 1]),
            "the largest task is not spare work"
        );
        assert!(auto.engages(&[100_000, AUTO_MIN_SPARE_INSTS / 2, AUTO_MIN_SPARE_INSTS / 2]));
        let forced = Parallelism {
            workers: 4,
            forced: true,
        };
        assert!(forced.engages(&[1, 1]));
        assert!(!forced.engages(&[1]));
        assert!(!Parallelism::SEQUENTIAL.engages(&[100_000, 100_000]));
    }

    /// Results come back in input order whatever order the threads ran them in, the batch really
    /// was shared, and the first failure in input order ends the report.
    #[test]
    fn results_keep_their_order_and_the_first_failure_wins() {
        let mut it = Interner::new();
        let mut funcs: Vec<Function> = (0..12)
            .map(|i| func(&mut it, &format!("f{i}"), 1 + (i * 7) % 5))
            .collect();
        let par = Parallelism {
            workers: 4,
            forced: true,
        };
        let caller = std::thread::current().id();
        let all: Vec<usize> = (0..funcs.len()).collect();
        let names: Vec<_> = funcs.iter().map(|f| f.name).collect();
        // Every task waits (a few seconds at most) for a second one to start, so one thread cannot
        // drain the queue alone while the others are still waking.
        let started = AtomicUsize::new(0);
        let got = map(par, select_mut(&mut funcs, &all), |f| {
            started.fetch_add(1, Ordering::SeqCst);
            let t0 = std::time::Instant::now();
            while started.load(Ordering::SeqCst) < 2 && t0.elapsed().as_secs() < 5 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            (f.name, std::thread::current().id())
        });
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), names);
        if pool().is_some() {
            assert!(
                got.iter().any(|g| g.1 != caller),
                "an engaged batch shares its tasks with the pool"
            );
        }

        let order: Vec<usize> = vec![5, 2, 9, 0];
        let failing = [names[9], names[0]];
        let (got, _) = try_map(par, select_mut(&mut funcs, &order), |f| {
            if failing.contains(&f.name) {
                Err(f.name)
            } else {
                Ok(f.name)
            }
        });
        assert_eq!(got, vec![Ok(names[5]), Ok(names[2]), Err(names[9])]);
        let (sequential, threads) = try_map(
            Parallelism::SEQUENTIAL,
            select_mut(&mut funcs, &order),
            |f| {
                if failing.contains(&f.name) {
                    Err(f.name)
                } else {
                    Ok(f.name)
                }
            },
        );
        assert_eq!(got, sequential);
        assert_eq!(threads, 1);
    }

    /// A panicking task is resumed on the caller, and it is the earliest panic in input order.
    #[test]
    fn the_earliest_panic_is_the_one_resumed() {
        let mut it = Interner::new();
        let mut funcs: Vec<Function> = (0..6).map(|i| func(&mut it, &format!("p{i}"), 2)).collect();
        let names: Vec<_> = funcs.iter().map(|f| f.name).collect();
        let par = Parallelism {
            workers: 3,
            forced: true,
        };
        let all: Vec<usize> = (0..funcs.len()).collect();
        let caught = panic::catch_unwind(AssertUnwindSafe(|| {
            map(par, select_mut(&mut funcs, &all), |f| {
                let k = names.iter().position(|&n| n == f.name).unwrap();
                if k == 2 || k == 4 {
                    panic::panic_any(k);
                }
                k
            })
        }));
        let payload = caught.expect_err("a task panicked");
        assert_eq!(payload.downcast_ref::<usize>(), Some(&2));
    }
}
