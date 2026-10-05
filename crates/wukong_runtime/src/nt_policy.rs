//! Shared non-temporal-store threshold for CPU streaming kernels.
//!
//! `WUKONG_NT_MIN_BYTES` is an optional strictly positive ASCII decimal byte count.
//! Unset, non-Unicode, empty, signed, whitespace-containing, zero and overflowing
//! values fall back to the historical 10 MiB default. Leading zeroes are accepted.
//! The first caller caches the process-wide value; later environment changes have
//! no effect. Configure the variable before starting the process.
//!
//! This is an explicit scheduling override, not cache-size detection or a claim
//! that 10 MiB is optimal. Each family retains its own saturated traffic accounting.

use std::sync::OnceLock;

const DEFAULT_NT_MIN_BYTES: usize = 10 * 1024 * 1024;

fn parse_nt_min_bytes(value: Option<&str>) -> usize {
    value
        .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_NT_MIN_BYTES)
}

#[inline]
pub(crate) fn nt_min_bytes() -> usize {
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| parse_nt_min_bytes(std::env::var("WUKONG_NT_MIN_BYTES").ok().as_deref()))
}

// Only tests that deliberately allocate a whole NT-regime working set use this
// budget. Production accepts every positive usize, and ordinary small tests do
// not consult the budget. Reject oversized NT tests before allocating or rounding.
#[cfg(test)]
fn checked_test_bytes(bytes: usize) -> usize {
    assert!(
        bytes <= 64 * 1024 * 1024,
        "NT coverage would exceed its 64 MiB traffic budget; run these tests with WUKONG_NT_MIN_BYTES <= 67108864"
    );
    bytes
}

#[cfg(test)]
pub(crate) fn test_min_bytes() -> usize {
    checked_test_bytes(nt_min_bytes())
}

#[cfg(test)]
fn test_len_for(bytes: usize, bytes_per_element: usize, original: usize) -> usize {
    let bytes = checked_test_bytes(bytes);
    assert!(bytes_per_element > 0, "NT test traffic must be positive");
    // Keep every original element and its mod-32 tail residue. Rounding the policy
    // minimum upward also leaves full vector blocks after a scalar alignment peel.
    let rounded = bytes.div_ceil(bytes_per_element).div_ceil(32) * 32;
    original.max(rounded + original % 32)
}

#[cfg(test)]
pub(crate) fn test_len(bytes_per_element: usize, original: usize) -> usize {
    test_len_for(nt_min_bytes(), bytes_per_element, original)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_only_positive_ascii_decimal_bytes() {
        assert_eq!(DEFAULT_NT_MIN_BYTES, 10 * 1024 * 1024);
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("000"),
            Some("-1"),
            Some("+1"),
            Some(" 1"),
            Some("1 "),
            Some("1_024"),
            Some("1MiB"),
            Some("１２"),
        ] {
            assert_eq!(parse_nt_min_bytes(value), DEFAULT_NT_MIN_BYTES, "{value:?}");
        }
        for (value, expected) in [
            ("1", 1),
            ("0004096", 4096),
            ("10485760", DEFAULT_NT_MIN_BYTES),
        ] {
            assert_eq!(parse_nt_min_bytes(Some(value)), expected);
        }
        assert_eq!(
            parse_nt_min_bytes(Some(&usize::MAX.to_string())),
            usize::MAX
        );
        assert_eq!(
            parse_nt_min_bytes(Some(&format!("{}0", usize::MAX))),
            DEFAULT_NT_MIN_BYTES
        );
    }

    #[test]
    fn test_lengths_preserve_cases_and_tail_residues() {
        for (traffic, original) in [
            (8, 1_500_001),
            (8, 1_500_003),
            (8, 1_500_005),
            (5, 2_097_155),
            (6, 1_747_661),
            (6, 1_749_000),
            (6, 2_000_003),
            (8, 1400 * 1501),
        ] {
            assert_eq!(
                test_len_for(DEFAULT_NT_MIN_BYTES, traffic, original),
                original
            );
        }
        for bytes_per_element in [5, 6, 8, 12] {
            for original in [1_500_001usize, 1_500_003, 1_500_005, 2_097_155] {
                for bytes in [
                    1,
                    4096,
                    DEFAULT_NT_MIN_BYTES,
                    24 * 1024 * 1024,
                    64 * 1024 * 1024,
                ] {
                    let n = test_len_for(bytes, bytes_per_element, original);
                    assert!(n >= original);
                    assert_eq!(n % 32, original % 32);
                    assert!(n * bytes_per_element >= bytes);
                }
            }
        }
        assert!(std::panic::catch_unwind(|| test_len_for(usize::MAX, 8, 35)).is_err());
        assert!(std::panic::catch_unwind(|| test_len_for(4096, 0, 35)).is_err());
    }

    #[test]
    fn environment_override_is_cached_in_fresh_process() {
        const CHILD: &str = "WUKONG_NT_POLICY_TEST_CHILD";
        if let Ok(expected) = std::env::var(CHILD) {
            let expected: usize = expected.parse().unwrap();
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| assert_eq!(nt_min_bytes(), expected));
                }
            });
            assert_eq!(nt_min_bytes(), expected);
            println!("NT policy child: {expected} bytes");
            return;
        }
        let mut cases: Vec<(Option<std::ffi::OsString>, usize)> = vec![
            (None, DEFAULT_NT_MIN_BYTES),
            (Some("4096".into()), 4096),
            (Some("0".into()), DEFAULT_NT_MIN_BYTES),
            (Some("+4096".into()), DEFAULT_NT_MIN_BYTES),
            (Some(usize::MAX.to_string().into()), usize::MAX),
        ];
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            cases.push((
                Some(std::ffi::OsString::from_wide(&[0xd800])),
                DEFAULT_NT_MIN_BYTES,
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            cases.push((
                Some(std::ffi::OsString::from_vec(vec![0xff])),
                DEFAULT_NT_MIN_BYTES,
            ));
        }
        for (value, expected) in cases {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command.args([
                "--exact",
                "nt_policy::tests::environment_override_is_cached_in_fresh_process",
                "--nocapture",
            ]);
            command.env(CHILD, expected.to_string());
            match &value {
                Some(value) => {
                    command.env("WUKONG_NT_MIN_BYTES", value);
                }
                None => {
                    command.env_remove("WUKONG_NT_MIN_BYTES");
                }
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "child {value:?}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout)
                .contains(&format!("NT policy child: {expected} bytes")));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn override_reaches_all_streaming_families() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["nt_alignment", "--nocapture"])
            .env("WUKONG_NT_MIN_BYTES", "4096")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stdout}{stderr}");
        let passed = if cfg!(debug_assertions) { 6 } else { 5 };
        assert!(
            stdout.contains(&format!("{passed} passed; 0 failed")),
            "{stdout}{stderr}"
        );
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            assert_eq!(
                stderr.matches("checked stores").count(),
                9,
                "{stdout}{stderr}"
            );
        }
        println!("NT override child: {stdout}{stderr}");
    }
}
