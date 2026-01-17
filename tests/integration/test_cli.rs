use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

#[test]
fn test_cli_stats() -> Result<(), Box<dyn std::error::Error>> {
    let file = "tests/integration/test.nt";
    fs::write(file, "<http://a> <http://b> <http://c> .")?;

    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("stats").arg(file);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Entities:       3"))
        .stdout(predicate::str::contains("Triples:        1"));

    fs::remove_file(file)?;
    Ok(())
}

#[test]
fn test_cli_walks() -> Result<(), Box<dyn std::error::Error>> {
    let input = "tests/integration/walks_input.nt";
    let output = "tests/integration/corpus.txt";
    
    // A -> B -> C
    let content = r#"
<http://A> <http://rel> <http://B> .
<http://B> <http://rel> <http://C> .
"#;
    fs::write(input, content)?;

    let mut cmd = Command::cargo_bin("lattice")?;
    cmd.arg("walks")
        .arg(input)
        .arg("-o")
        .arg(output)
        .arg("--length")
        .arg("5")
        .arg("--num-walks")
        .arg("2")
        .arg("--seed")
        .arg("42");

    cmd.assert().success();

    let corpus = fs::read_to_string(output)?;
    // Should have 3 nodes * 2 walks = 6 lines
    assert_eq!(corpus.lines().count(), 6);
    // Walks should contain A, B, C
    assert!(corpus.contains("http://A"));
    assert!(corpus.contains("http://B"));
    assert!(corpus.contains("http://C"));

    fs::remove_file(input)?;
    fs::remove_file(output)?;
    Ok(())
}
