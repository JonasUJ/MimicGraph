use anyhow::Result;
use mimicgraph_core::labels::LabelSet;
use mimicgraph_core::mimicgraph::MimicGraphOptions;
use roargraph::RoarGraphOptions;
use sprs::CsMat;
use std::collections::HashMap;
use std::path::Path;

pub fn csmat_to_map(mat: CsMat<usize>, count: usize) -> Vec<LabelSet> {
    let mut map = Vec::with_capacity(mat.rows());
    let indices = mat.indices();

    for window in mat.proper_indptr().windows(2).take(count) {
        let mut set = LabelSet::new();

        for &idx in &indices[window[0]..window[1]] {
            set.insert(idx);
        }

        map.push(set);
    }

    map
}

pub fn dataset_file_name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("dataset path must include a valid UTF-8 file name"))
}

pub fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("artifact path must be valid UTF-8"))
}

pub fn require_labels(
    result: Result<Vec<LabelSet>, impl std::fmt::Display>,
    kind: &str,
) -> Result<Vec<LabelSet>> {
    result.map_err(|e| {
        anyhow::anyhow!("filtered mode requested, but `{kind}` could not be read: {e}")
    })
}

pub fn parse_search_params(input: &str) -> Result<Vec<(usize, usize)>> {
    let mut params = Vec::new();

    for entry in input.split(',').filter(|s| !s.trim().is_empty()) {
        let trimmed = entry.trim();
        let Some((k, ef)) = trimmed.split_once(':') else {
            anyhow::bail!("invalid --search-params entry, expected k:ef");
        };

        params.push((k.trim().parse()?, ef.trim().parse()?));
    }

    if params.is_empty() {
        anyhow::bail!("--search-params must not be empty");
    }

    Ok(params)
}

pub fn validate_build_percent(build_percent: f64) -> Result<()> {
    anyhow::ensure!(
        build_percent > 0.0 && build_percent <= 100.0,
        "--build-percent must be in (0, 100]"
    );

    Ok(())
}

pub fn build_count_from_percent(corpus_len: usize, build_percent: f64) -> usize {
    if corpus_len == 0 {
        0
    } else {
        let raw = ((corpus_len as f64) * (build_percent / 100.0)).round() as usize;

        raw.clamp(1, corpus_len)
    }
}

fn parse_kv(input: &str) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();

    for entry in input.split(',').filter(|s| !s.trim().is_empty()) {
        let trimmed = entry.trim();
        let Some((key, value)) = trimmed.split_once('=') else {
            anyhow::bail!("options must use key=value format");
        };

        map.insert(key.trim().to_string(), value.trim().to_string());
    }

    Ok(map)
}

fn ensure_empty(map: &HashMap<String, String>) -> Result<()> {
    if map.is_empty() {
        return Ok(());
    }

    let mut unknown = map.keys().cloned().collect::<Vec<_>>();
    unknown.sort();

    anyhow::bail!("unknown option keys: {}", unknown.join(","))
}

fn take_usize(map: &mut HashMap<String, String>, key: &str, default: usize) -> Result<usize> {
    map.remove(key)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .map_or(Ok(default), Ok)
}

fn take_f32(map: &mut HashMap<String, String>, key: &str, default: f32) -> Result<f32> {
    map.remove(key)
        .map(|v| v.parse::<f32>())
        .transpose()?
        .map_or(Ok(default), Ok)
}

fn take_bool(map: &mut HashMap<String, String>, key: &str, default: bool) -> Result<bool> {
    let Some(value) = map.remove(key) else {
        return Ok(default);
    };

    match value.as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => anyhow::bail!("invalid boolean for {key}: {value}"),
    }
}

pub struct ParsedFilteredMimicGraphOptions {
    pub base: MimicGraphOptions,
    pub threshold: usize,
}

pub struct ParsedFilteredVamanaOptions {
    pub alpha: f32,
    pub l: usize,
    pub r: usize,
    pub threshold: usize,
}

pub fn parse_mimicgraph_options(input: &str) -> Result<MimicGraphOptions> {
    let mut map = parse_kv(input)?;

    let options = MimicGraphOptions {
        m: take_usize(&mut map, "m", 32)?,
        l: take_usize(&mut map, "l", 500)?,
        p: take_usize(&mut map, "p", 100)?,
        e: take_usize(&mut map, "e", 16)?,
        qk: take_usize(&mut map, "qk", 0)?,
        qef: take_usize(&mut map, "qef", 100)?,
        con: take_bool(&mut map, "con", false)?,
        vis: take_bool(&mut map, "vis", true)?,
    };
    ensure_empty(&map)?;

    Ok(options)
}

pub fn parse_filtered_mimicgraph_options(input: &str) -> Result<ParsedFilteredMimicGraphOptions> {
    let mut map = parse_kv(input)?;

    let base = MimicGraphOptions {
        m: take_usize(&mut map, "m", 32)?,
        l: take_usize(&mut map, "l", 500)?,
        p: take_usize(&mut map, "p", 100)?,
        e: take_usize(&mut map, "e", 16)?,
        qk: take_usize(&mut map, "qk", 0)?,
        qef: take_usize(&mut map, "qef", 100)?,
        con: take_bool(&mut map, "con", false)?,
        vis: take_bool(&mut map, "vis", true)?,
    };
    let threshold = take_usize(&mut map, "threshold", 1000)?;
    ensure_empty(&map)?;

    Ok(ParsedFilteredMimicGraphOptions { base, threshold })
}

pub fn parse_hnsw_options(input: &str) -> Result<(usize, usize, usize)> {
    let mut map = parse_kv(input)?;

    let options = (
        take_usize(&mut map, "ef_construction", 400)?,
        take_usize(&mut map, "connections", 24)?,
        take_usize(&mut map, "max_connections", 64)?,
    );
    ensure_empty(&map)?;

    Ok(options)
}

pub fn parse_filtered_vamana_options(input: &str) -> Result<ParsedFilteredVamanaOptions> {
    let mut map = parse_kv(input)?;

    let options = ParsedFilteredVamanaOptions {
        alpha: take_f32(&mut map, "alpha", 1.2)?,
        l: take_usize(&mut map, "l", 90)?,
        r: take_usize(&mut map, "r", 96)?,
        threshold: take_usize(&mut map, "threshold", 1000)?,
    };
    ensure_empty(&map)?;

    Ok(options)
}

pub fn parse_roargraph_options(input: &str) -> Result<RoarGraphOptions> {
    let mut map = parse_kv(input)?;

    let options = RoarGraphOptions {
        m: take_usize(&mut map, "m", 32)?,
        l: take_usize(&mut map, "l", 500)?,
    };
    ensure_empty(&map)?;

    Ok(options)
}
