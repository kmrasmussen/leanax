use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug)]
struct Case {
    name: String,
    output: String,
    golden: String,
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
        let fields: Vec<_> = trimmed.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(format!(
                "manifest line {} must have 3 fields: case output golden",
                index + 1
            ));
        }
        cases.push(Case {
            name: fields[0].to_string(),
            output: fields[1].to_string(),
            golden: fields[2].to_string(),
        });
    }

    if cases.is_empty() {
        Err(format!("manifest {manifest} has no cases"))
    } else {
        Ok(cases)
    }
}

fn read_invalid_manifest(repo: &Path, manifest: &str) -> Result<Vec<(String, String)>, String> {
    let text = fs::read_to_string(repo.join(manifest))
        .map_err(|err| format!("failed reading invalid manifest {manifest}: {err}"))?;
    let mut cases = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((name, expected)) = trimmed.split_once('|') else {
            return Err(format!(
                "invalid manifest line {} must use: case|expected stderr",
                index + 1
            ));
        };
        cases.push((name.to_string(), expected.to_string()));
    }

    Ok(cases)
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

fn run_case(repo: &Path, case: &Case) -> Result<(), String> {
    if let Some(parent) = repo.join(&case.output).parent() {
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
            &case.output,
        ],
    )?;
    compare(repo, &case.output, &case.golden)?;
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
            &case.output,
        ],
    )?;
    Ok(())
}

fn run_invalid_case(repo: &Path, name: &str, expected_stderr: &str) -> Result<(), String> {
    let output = format!("generated/invalid-{name}.mlir");
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
            &output,
        ],
        expected_stderr,
    )
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
    let invalid_cases = read_invalid_manifest(&repo, "e2e/invalid_manifest.txt")?;

    run(&repo, "lake", &["build"])?;
    for case in &cases {
        eprintln!("case: {}", case.name);
        run_case(&repo, case)?;
    }
    for (name, expected_stderr) in &invalid_cases {
        eprintln!("invalid case: {}", name);
        run_invalid_case(&repo, name, expected_stderr)?;
    }

    Ok(())
}
