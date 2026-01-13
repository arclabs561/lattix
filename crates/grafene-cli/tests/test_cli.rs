use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;

fn get_test_dir() -> PathBuf {
    let dir = PathBuf::from("target/tmp/tests");
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_cli_stats() -> Result<(), Box<dyn std::error::Error>> {
    let dir = get_test_dir();
    let file = dir.join("test_stats.nt");
    fs::write(&file, "<http://a> <http://b> <http://c> .")?;

    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("stats").arg(&file);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Entities:       2"))
        .stdout(predicate::str::contains("Triples:        1"));

    fs::remove_file(file)?;
    Ok(())
}

#[test]
fn test_cli_walks() -> Result<(), Box<dyn std::error::Error>> {
    let dir = get_test_dir();
    let input = dir.join("walks_input.nt");
    let output = dir.join("corpus.txt");

    // A -> B -> C
    let content = r#"
<http://A> <http://rel> <http://B> .
<http://B> <http://rel> <http://C> .
"#;
    fs::write(&input, content)?;

    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("walks")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .arg("--length")
        .arg("5")
        .arg("--num-walks")
        .arg("2")
        .arg("--seed")
        .arg("42");

    cmd.assert().success();

    let corpus = fs::read_to_string(&output)?;
    // Should have 3 nodes * 2 walks = 6 lines
    assert_eq!(corpus.lines().count(), 6);
    // Walks should contain A, B, C
    assert!(corpus.contains("http://A"));
    assert!(corpus.contains("http://B"));
    assert!(corpus.contains("http://C"));

    // Cleanup
    if input.exists() {
        fs::remove_file(input)?;
    }
    if output.exists() {
        fs::remove_file(output)?;
    }
    Ok(())
}

#[test]
fn test_full_pipeline_csv() -> Result<(), Box<dyn std::error::Error>> {
    let dir = get_test_dir();
    let input = dir.join("social.csv");
    let walks_out = dir.join("social_walks.txt");
    let binary_out = dir.join("social.bin");

    // Create a small social graph
    // Alice -> Bob
    // Bob -> Charlie
    // Charlie -> Alice
    // Dave -> Eve (disconnected component)
    let content = "Alice,knows,Bob\nBob,knows,Charlie\nCharlie,knows,Alice\nDave,knows,Eve";
    fs::write(&input, content)?;

    // 1. Stats
    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("stats").arg(&input);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Entities:       5"))
        .stdout(predicate::str::contains("Triples:        4"));

    // 2. Components (WCC - weakly connected, treating edges as undirected)
    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("components").arg(&input).arg("--verbose");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Number of components: 2")) // {A,B,C} and {Dave, Eve}
        .stdout(predicate::str::contains("3 nodes")); // The larger component

    // 3. PageRank
    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("pagerank").arg(&input).arg("--top").arg("5");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Alice"))
        .stdout(predicate::str::contains("Bob"));

    // 4. Walks
    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("walks")
        .arg(&input)
        .arg("-o")
        .arg(&walks_out)
        .arg("--length")
        .arg("10")
        .arg("--num-walks")
        .arg("5");
    cmd.assert().success();

    assert!(walks_out.exists());
    let walks = fs::read_to_string(&walks_out)?;
    assert!(walks.len() > 0);

    // 5. Save/Load Binary
    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("save").arg(&input).arg(&binary_out);
    cmd.assert().success();
    assert!(binary_out.exists());

    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("stats").arg(&binary_out);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Entities:       5"));

    // Cleanup
    if input.exists() {
        fs::remove_file(input)?;
    }
    if walks_out.exists() {
        fs::remove_file(walks_out)?;
    }
    if binary_out.exists() {
        fs::remove_file(binary_out)?;
    }

    Ok(())
}
