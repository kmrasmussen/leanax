use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug)]
struct Case {
    name: String,
    expectation: Expectation,
}

#[derive(Debug)]
enum Expectation {
    Pass { output: String, golden: String },
    ValidationFail { expected_stderr: String },
}

fn usage() -> &'static str {
    "leanax-e2e-runner [--repo PATH] [--manifest PATH]"
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn is_repo_root(path: &Path) -> bool {
    path.join("lakefile.lean").is_file() && path.join("e2e/manifest.txt").is_file()
}

fn find_repo_root() -> Result<PathBuf, String> {
    let mut current = env::current_dir().map_err(|err| format!("failed to read cwd: {err}"))?;
    loop {
        if is_repo_root(&current) {
            return Ok(current);
        }
        if !current.pop() {
            return Err("could not find repo root with lakefile.lean and e2e/manifest.txt".into());
        }
    }
}

fn repo_root(args: &[String]) -> Result<PathBuf, String> {
    if let Some(path) = parse_flag(args, "--repo") {
        let root = PathBuf::from(path);
        if is_repo_root(&root) {
            Ok(root)
        } else {
            Err(format!(
                "--repo path '{}' is missing lakefile.lean or e2e/manifest.txt",
                root.display()
            ))
        }
    } else {
        find_repo_root()
    }
}

fn read_manifest(repo: &Path, manifest: &str) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(repo.join(manifest))
        .map_err(|err| format!("failed reading manifest {manifest}: {err}"))?;
    let mut cases = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.splitn(3, ' ');
        let outcome = parts.next().unwrap_or_default();
        let Some(name) = parts.next() else {
            return Err(format!(
                "manifest line {} is missing a case name",
                index + 1
            ));
        };
        let Some(rest) = parts.next() else {
            return Err(format!(
                "manifest line {} is missing expectation data",
                index + 1
            ));
        };

        let expectation = match outcome {
            "pass" => {
                let fields: Vec<_> = rest.split_whitespace().collect();
                if fields.len() != 2 {
                    return Err(format!(
                        "manifest line {} pass cases must use: pass case output golden",
                        index + 1
                    ));
                }
                Expectation::Pass {
                    output: fields[0].to_string(),
                    golden: fields[1].to_string(),
                }
            }
            "validation-fail" => Expectation::ValidationFail {
                expected_stderr: rest.to_string(),
            },
            other => {
                return Err(format!(
                    "manifest line {} has unknown outcome '{other}'",
                    index + 1
                ));
            }
        };

        if name.trim().is_empty() {
            return Err(format!(
                "manifest line {} has an empty case name",
                index + 1
            ));
        };
        cases.push(Case {
            name: name.to_string(),
            expectation,
        });
    }

    if cases.is_empty() {
        Err(format!("manifest {manifest} has no cases"))
    } else {
        Ok(cases)
    }
}

fn run(repo: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    eprintln!("running: {} {}", program, args.join(" "));
    let mut command = Command::new(program);
    command.args(args).current_dir(repo).stdin(Stdio::null());
    if program == "uv" {
        command.env("UV_CACHE_DIR", env::temp_dir().join("leanax-uv-cache"));
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to start {program}: {err}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} exited with {status}"))
    }
}

fn run_mlir_parse(repo: &Path, path: &str) -> Result<(), String> {
    eprintln!("running: mlir-opt --allow-unregistered-dialect {path}");
    let status = Command::new("mlir-opt")
        .arg("--allow-unregistered-dialect")
        .arg(path)
        .current_dir(repo)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .map_err(|err| format!("failed to start mlir-opt: {err}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("mlir-opt exited with {status}"))
    }
}

fn run_expect_failure(
    repo: &Path,
    program: &str,
    args: &[&str],
    expected_stderr: &str,
) -> Result<(), String> {
    eprintln!("running expected failure: {} {}", program, args.join(" "));
    let output = Command::new(program)
        .args(args)
        .current_dir(repo)
        .stdin(Stdio::null())
        .output()
        .map_err(|err| format!("failed to start {program}: {err}"))?;

    if output.status.success() {
        return Err(format!("{program} unexpectedly succeeded"));
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains(expected_stderr) {
        Ok(())
    } else {
        Err(format!(
            "stderr did not contain expected text '{expected_stderr}'. stderr was:\n{stderr}"
        ))
    }
}

fn compare(repo: &Path, actual: &str, expected: &str) -> Result<(), String> {
    let actual_text = fs::read_to_string(repo.join(actual))
        .map_err(|err| format!("failed reading {actual}: {err}"))?;
    let expected_text = fs::read_to_string(repo.join(expected))
        .map_err(|err| format!("failed reading {expected}: {err}"))?;

    if actual_text == expected_text {
        Ok(())
    } else {
        Err(format!("{actual} differs from {expected}"))
    }
}

fn run_pass_case(repo: &Path, case: &Case, output: &str, golden: &str) -> Result<(), String> {
    if let Some(parent) = repo.join(output).parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating output dir {}: {err}", parent.display()))?;
    }

    run(
        repo,
        "lake",
        &[
            "exe",
            "leanax",
            "emit-stablehlo",
            "--case",
            &case.name,
            "--out",
            output,
        ],
    )?;
    compare(repo, output, golden)?;
    run(
        repo,
        "uv",
        &[
            "run",
            "--no-managed-python",
            "--python",
            "python3",
            "--project",
            "e2e/python",
            "python",
            "e2e/python/verify_stablehlo_text.py",
            output,
        ],
    )?;
    run_mlir_parse(repo, output)?;
    Ok(())
}

fn run_validation_fail_case(repo: &Path, name: &str, expected_stderr: &str) -> Result<(), String> {
    let output = repo.join(format!("generated/invalid-{name}.mlir"));
    if output.exists() {
        fs::remove_file(&output)
            .map_err(|err| format!("failed removing stale {}: {err}", output.display()))?;
    }
    let output_arg = output
        .strip_prefix(repo)
        .map_err(|err| format!("failed making output path relative: {err}"))?
        .to_string_lossy()
        .into_owned();
    run_expect_failure(
        repo,
        "lake",
        &[
            "exe",
            "leanax",
            "emit-stablehlo",
            "--case",
            name,
            "--out",
            &output_arg,
        ],
        expected_stderr,
    )?;
    if output.exists() {
        Err(format!(
            "validation-fail case {name} left unexpected output {}",
            output.display()
        ))
    } else {
        Ok(())
    }
}

fn run_case(repo: &Path, case: &Case) -> Result<&'static str, String> {
    match &case.expectation {
        Expectation::Pass { output, golden } => {
            eprintln!("case pass: {}", case.name);
            run_pass_case(repo, case, output, golden)?;
            Ok("pass")
        }
        Expectation::ValidationFail { expected_stderr } => {
            eprintln!("case validation-fail: {}", case.name);
            run_validation_fail_case(repo, &case.name, expected_stderr)?;
            Ok("validation-fail")
        }
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", usage());
        return Ok(());
    }

    let repo = repo_root(&args)?;
    let manifest = parse_flag(&args, "--manifest").unwrap_or_else(|| "e2e/manifest.txt".into());
    let cases = read_manifest(&repo, &manifest)?;
    let mut pass_count = 0usize;
    let mut validation_fail_count = 0usize;

    run(&repo, "lake", &["build"])?;
    for case in &cases {
        match run_case(&repo, case)? {
            "pass" => pass_count += 1,
            "validation-fail" => validation_fail_count += 1,
            _ => unreachable!(),
        }
    }
    eprintln!(
        "e2e summary: {pass_count} pass, {validation_fail_count} expected validation-fail, 0 unexpected"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_repo(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "leanax-e2e-runner-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp test dir");
        dir
    }

    fn write_manifest(repo: &Path, text: &str) {
        fs::write(repo.join("manifest.txt"), text).expect("write manifest");
    }

    #[test]
    fn reads_pass_and_validation_fail_cases() {
        let repo = temp_repo("valid-manifest");
        write_manifest(
            &repo,
            "\
# outcome case output/golden-or-expected
pass affine generated/affine.mlir e2e/golden/affine.mlir
validation-fail bad-add-shape stablehlo.add operands: expected matching tensor types
",
        );

        let cases = read_manifest(&repo, "manifest.txt").expect("manifest should parse");
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].name, "affine");
        assert!(matches!(cases[0].expectation, Expectation::Pass { .. }));
        assert_eq!(cases[1].name, "bad-add-shape");
        assert!(matches!(
            cases[1].expectation,
            Expectation::ValidationFail { .. }
        ));
    }

    #[test]
    fn rejects_unknown_outcome() {
        let repo = temp_repo("unknown-outcome");
        write_manifest(
            &repo,
            "surprise affine generated/affine.mlir e2e/golden/affine.mlir\n",
        );

        let err = read_manifest(&repo, "manifest.txt").expect_err("manifest should fail");
        assert!(err.contains("unknown outcome 'surprise'"));
    }

    #[test]
    fn rejects_malformed_pass_case() {
        let repo = temp_repo("malformed-pass");
        write_manifest(&repo, "pass affine generated/affine.mlir\n");

        let err = read_manifest(&repo, "manifest.txt").expect_err("manifest should fail");
        assert!(err.contains("pass cases must use"));
    }

    #[test]
    fn rejects_empty_manifest() {
        let repo = temp_repo("empty-manifest");
        write_manifest(&repo, "# only comments\n\n");

        let err = read_manifest(&repo, "manifest.txt").expect_err("manifest should fail");
        assert!(err.contains("has no cases"));
    }
}
