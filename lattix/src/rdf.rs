//! Internal helpers for RDF term rendering.

use std::fmt::Write as _;

use crate::{EntityId, RelationType, Triple};

const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
const LOCAL_IRI_PREFIX: &str = "urn:lattix:local:";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LiteralParts {
    pub(crate) lexical: String,
    pub(crate) language: Option<String>,
    pub(crate) datatype: Option<String>,
}

pub(crate) fn subject_to_string(s: &oxrdf::NamedOrBlankNode) -> String {
    match s {
        oxrdf::NamedOrBlankNode::NamedNode(n) => iri_to_string(n.as_str()),
        oxrdf::NamedOrBlankNode::BlankNode(b) => format!("_:{}", b.as_str()),
    }
}

#[cfg(feature = "formats")]
pub(crate) fn graph_name_to_string(g: &oxrdf::GraphName) -> Option<String> {
    match g {
        oxrdf::GraphName::NamedNode(n) => Some(iri_to_string(n.as_str())),
        oxrdf::GraphName::BlankNode(b) => Some(format!("_:{}", b.as_str())),
        oxrdf::GraphName::DefaultGraph => None,
    }
}

pub(crate) fn term_to_string(t: &oxrdf::Term) -> String {
    match t {
        oxrdf::Term::NamedNode(n) => iri_to_string(n.as_str()),
        oxrdf::Term::BlankNode(b) => format!("_:{}", b.as_str()),
        oxrdf::Term::Literal(l) => {
            if let Some(lang) = l.language() {
                render_literal(l.value(), Some(lang), None)
            } else {
                let datatype = l.datatype().as_str();
                if datatype == XSD_STRING {
                    render_literal(l.value(), None, None)
                } else {
                    render_literal(l.value(), None, Some(datatype))
                }
            }
        }
    }
}

pub(crate) fn parse_ntriples_line(line: &str) -> crate::Result<Triple> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Err(crate::Error::ParseTriple("empty or comment line".into()));
    }

    let mut parsed = None;
    for result in oxttl::NTriplesParser::new().for_reader(std::io::Cursor::new(line.as_bytes())) {
        let triple =
            result.map_err(|e| crate::Error::InvalidNTriples(format!("{}: {}", line, e)))?;
        if parsed.is_some() {
            return Err(crate::Error::InvalidNTriples(format!(
                "expected one triple: {}",
                line
            )));
        }
        parsed = Some(Triple::new(
            subject_to_string(&triple.subject),
            iri_to_string(triple.predicate.as_str()),
            term_to_string(&triple.object),
        ));
    }

    parsed.ok_or_else(|| crate::Error::InvalidNTriples(line.to_string()))
}

pub(crate) fn render_iri_or_blank(s: &str) -> String {
    if s.starts_with("_:") || (s.starts_with('<') && s.ends_with('>')) {
        s.to_string()
    } else {
        format!("<{}>", iri_body_for(s))
    }
}

pub(crate) fn render_object(s: &str) -> String {
    if s.starts_with("_:") {
        s.to_string()
    } else if let Some(literal) = normalize_literal(s) {
        literal
    } else if s.starts_with('<') && s.ends_with('>') {
        s.to_string()
    } else {
        format!("<{}>", iri_body_for(s))
    }
}

pub(crate) fn render_literal(
    lexical: &str,
    language: Option<&str>,
    datatype: Option<&str>,
) -> String {
    let mut rendered = format!("\"{}\"", escape_literal_lexical(lexical));
    if let Some(language) = language {
        rendered.push('@');
        rendered.push_str(language);
    } else if let Some(datatype) = datatype {
        rendered.push_str("^^<");
        rendered.push_str(&iri_body_for(datatype));
        rendered.push('>');
    }
    rendered
}

pub(crate) fn normalize_literal(s: &str) -> Option<String> {
    let parts = parse_literal(s)?;
    Some(render_literal(
        &parts.lexical,
        parts.language.as_deref(),
        parts.datatype.as_deref(),
    ))
}

pub(crate) fn parse_literal(s: &str) -> Option<LiteralParts> {
    let rest = s.strip_prefix('"')?;
    let mut escaped = false;

    for (idx, ch) in rest.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }

        match ch {
            '\\' => escaped = true,
            '"' => {
                let suffix = &rest[idx + ch.len_utf8()..];
                if suffix.is_empty() || suffix.starts_with('@') || suffix.starts_with("^^<") {
                    let lexical = unescape_literal_lexical(&rest[..idx])?;
                    if suffix.is_empty() {
                        return Some(LiteralParts {
                            lexical,
                            language: None,
                            datatype: None,
                        });
                    }
                    if let Some(language) = suffix.strip_prefix('@') {
                        if language.is_empty() {
                            return None;
                        }
                        return Some(LiteralParts {
                            lexical,
                            language: Some(language.to_string()),
                            datatype: None,
                        });
                    }
                    if let Some(datatype) = suffix
                        .strip_prefix("^^<")
                        .and_then(|dt| dt.strip_suffix('>'))
                    {
                        if datatype.is_empty() {
                            return None;
                        }
                        return Some(LiteralParts {
                            lexical,
                            language: None,
                            datatype: Some(iri_to_string(datatype)),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    None
}

pub(crate) fn iri_to_string(iri: &str) -> String {
    decode_local_iri(iri).unwrap_or_else(|| iri.to_string())
}

pub(crate) fn iri_body_for(s: &str) -> String {
    if is_absolute_iri(s) && !s.starts_with(LOCAL_IRI_PREFIX) {
        escape_iri(s)
    } else {
        format!("{LOCAL_IRI_PREFIX}{}", percent_encode(s))
    }
}

fn is_absolute_iri(s: &str) -> bool {
    let Some((first, rest)) = s.split_once(':') else {
        return false;
    };
    let mut chars = first.chars();
    let Some(first_char) = chars.next() else {
        return false;
    };
    first_char.is_ascii_alphabetic()
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
        && !rest.is_empty()
}

fn decode_local_iri(iri: &str) -> Option<String> {
    let encoded = iri.strip_prefix(LOCAL_IRI_PREFIX)?;
    percent_decode(encoded)
}

fn percent_encode(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len());
    for byte in s.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            write!(encoded, "%{byte:02X}").expect("write to string");
        }
    }
    encoded
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            decoded.push(bytes[i]);
            i += 1;
            continue;
        }
        let hi = bytes.get(i + 1).copied()? as char;
        let lo = bytes.get(i + 2).copied()? as char;
        let value = (hi.to_digit(16)? << 4) | lo.to_digit(16)?;
        decoded.push(u8::try_from(value).ok()?);
        i += 3;
    }
    String::from_utf8(decoded).ok()
}

pub(crate) fn escape_literal_lexical(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

pub(crate) fn escape_iri(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len());
    for ch in s.chars() {
        if matches!(ch, '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' | '\\') || ch <= '\u{20}' {
            write_codepoint_escape(&mut escaped, ch);
        } else {
            escaped.push(ch);
        }
    }
    escaped
}

fn unescape_literal_lexical(s: &str) -> Option<String> {
    let mut unescaped = String::with_capacity(s.len());
    let mut chars = s.chars();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            unescaped.push(ch);
            continue;
        }

        let escaped = chars.next()?;
        match escaped {
            't' => unescaped.push('\t'),
            'b' => unescaped.push('\u{08}'),
            'n' => unescaped.push('\n'),
            'r' => unescaped.push('\r'),
            'f' => unescaped.push('\u{0C}'),
            '"' => unescaped.push('"'),
            '\'' => unescaped.push('\''),
            '\\' => unescaped.push('\\'),
            'u' => unescaped.push(read_hex_escape(&mut chars, 4)?),
            'U' => unescaped.push(read_hex_escape(&mut chars, 8)?),
            _ => return None,
        }
    }

    Some(unescaped)
}

fn read_hex_escape(chars: &mut impl Iterator<Item = char>, digits: usize) -> Option<char> {
    let mut value = 0_u32;
    for _ in 0..digits {
        let digit = chars.next()?.to_digit(16)?;
        value = (value << 4) | digit;
    }
    char::from_u32(value)
}

fn write_codepoint_escape(out: &mut String, ch: char) {
    let value = ch as u32;
    if value <= 0xFFFF {
        write!(out, "\\u{value:04X}").expect("write to string");
    } else {
        write!(out, "\\U{value:08X}").expect("write to string");
    }
}

pub(crate) fn statement_reification_triples(statement_id: &str, core: &Triple) -> [Triple; 3] {
    [
        Triple::new(
            EntityId::from(statement_id.to_string()),
            RelationType::from("rdf:subject"),
            core.subject().clone(),
        ),
        Triple::new(
            EntityId::from(statement_id.to_string()),
            RelationType::from("rdf:predicate"),
            EntityId::from(core.predicate().as_str()),
        ),
        Triple::new(
            EntityId::from(statement_id.to_string()),
            RelationType::from("rdf:object"),
            core.object().clone(),
        ),
    ]
}
