// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub offset_us: u64,
    pub data: Json,
}
#[derive(Debug, Clone, Deserialize)]
pub struct Call {
    pub seq: u64,
    pub key_hash: Option<String>,
    pub repeat_index: Option<u64>,
    pub stream: bool,
    pub outcome: String,
    pub ttft_us: Option<u64>,
    pub total_us: u64,
    pub chunks: Option<Vec<Chunk>>,
    pub response: Json,
}

pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
pub fn envelope(id: &str, mut value: Json) -> Json {
    value["v"] = json!(1);
    value["activation_id"] = json!(id);
    value["created_unix_ns"] = json!(now_ns());
    value
}
pub fn atomic_json(path: &Path, value: &Json) -> io::Result<()> {
    let temp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temp)?;
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temp, path)?;
    File::open(path.parent().unwrap_or(Path::new(".")))?.sync_all()
}
fn write_line(file: &mut impl Write, value: &Json) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    let written = file.write(&bytes)?;
    if written != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "short trace write",
        ));
    }
    Ok(())
}

pub fn append_output(path: &Path, value: &Json) -> io::Result<()> {
    write_line(
        &mut OpenOptions::new().append(true).create(true).open(path)?,
        value,
    )
}

pub struct Writer {
    files: HashMap<&'static str, File>,
    pub lines: BTreeMap<String, u64>,
    _lock: File,
}
impl Writer {
    #[cfg(all(test, unix))]
    pub fn fail_calls(&mut self) {
        self.files.insert("calls", File::open("/dev/null").unwrap());
    }
    pub fn create(
        dir: &Path,
        metadata: &Json,
        nonce: &str,
        id: &str,
    ) -> Result<Self, &'static str> {
        // Another process's live or unfinalized recording owns the folder.
        if dir.join("record.lock").exists() {
            return Err("trace_locked");
        }
        for name in ["trace.json", "calls.jsonl", "requests.jsonl", "tools.jsonl"] {
            if dir.join(name).exists() {
                return Err("trace_directory_not_empty");
            }
        }
        let mut lock = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("record.lock"))
            .map_err(|e| {
                if e.kind() == io::ErrorKind::AlreadyExists {
                    "trace_locked"
                } else {
                    "trace_unwritable"
                }
            })?;
        lock.try_lock().map_err(|_| "trace_locked")?;
        write_line(&mut lock,&json!({"nonce":nonce,"pid":std::process::id(),"activation_id":id,"created_unix_ns":now_ns()})).map_err(|_|"trace_unwritable")?;
        lock.sync_all().map_err(|_| "trace_unwritable")?;
        let mut metadata_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("trace.json"))
            .map_err(|_| "trace_unwritable")?;
        write_line(&mut metadata_file, metadata).map_err(|_| "trace_unwritable")?;
        metadata_file.sync_all().map_err(|_| "trace_unwritable")?;
        let mut files = HashMap::new();
        let mut lines = BTreeMap::new();
        for (kind, name) in [
            ("calls", "calls.jsonl"),
            ("requests", "requests.jsonl"),
            ("tools", "tools.jsonl"),
        ] {
            files.insert(
                kind,
                OpenOptions::new()
                    .append(true)
                    .create_new(true)
                    .open(dir.join(name))
                    .map_err(|_| "trace_unwritable")?,
            );
            lines.insert(kind.into(), 0);
        }
        Ok(Self {
            files,
            lines,
            _lock: lock,
        })
    }
    pub fn append(&mut self, kind: &'static str, value: &Json) -> io::Result<()> {
        write_line(self.files.get_mut(kind).expect("owned trace file"), value)?;
        *self.lines.get_mut(kind).expect("owned trace count") += 1;
        Ok(())
    }
    pub fn sync(&self) -> io::Result<()> {
        for file in self.files.values() {
            file.sync_all()?;
        }
        Ok(())
    }
}

pub struct Reader {
    pub calls: HashMap<(String, u64), Call>,
    pub tools: HashMap<String, Json>,
    pub directory: PathBuf,
}
pub fn tool_key(line: &Json) -> Option<String> {
    let parent = if line["parent_key_hash"].is_string() {
        json!([line["parent_key_hash"], line["parent_repeat_index"]])
    } else {
        json!(["keyless", line["parent_seq"]])
    };
    let tail = if line["match"] == "id" {
        json!(["id", line["tool_call_id"]])
    } else {
        json!(["fallback", line["tool_name"], line["ordinal"]])
    };
    crate::key::fingerprint(&json!([parent, tail]))
}
pub fn json_lines(path: &Path) -> Result<Vec<Json>, &'static str> {
    let bytes = fs::read(path).map_err(|_| "trace_unreadable")?;
    if !bytes.is_empty() && bytes.last() != Some(&b'\n') {
        return Err("trace_corrupt");
    }
    bytes
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| {
            let value: Json = serde_json::from_slice(line).map_err(|_| "trace_corrupt")?;
            if value["v"] != 1 {
                return Err("trace_corrupt");
            }
            Ok(value)
        })
        .collect()
}
impl Reader {
    pub fn load(dir: &Path, policy: &str) -> Result<Self, &'static str> {
        if dir.join("record.lock").exists() {
            return Err("trace_not_finalized");
        }
        let value: Json = serde_json::from_slice(
            &fs::read(dir.join("trace.json")).map_err(|_| "trace_unreadable")?,
        )
        .map_err(|_| "trace_corrupt")?;
        if value["format"]["major"] != 1 {
            return Err("trace_format_unsupported");
        }
        if value["key_policy_sha256"] != policy {
            return Err("key_policy_mismatch");
        }
        let mut calls = HashMap::new();
        let mut sequences = std::collections::HashSet::new();
        let mut repeats: HashMap<String, Vec<u64>> = HashMap::new();
        for value in json_lines(&dir.join("calls.jsonl"))? {
            let call: Call = serde_json::from_value(value).map_err(|_| "trace_corrupt")?;
            if !sequences.insert(call.seq) {
                return Err("trace_corrupt");
            }
            if !["ok", "provider_error", "stream_error", "consumer_dropped"]
                .contains(&call.outcome.as_str())
            {
                return Err("trace_corrupt");
            }
            if call.stream {
                let chunks = call.chunks.as_ref().ok_or("trace_corrupt")?;
                let mut previous = 0;
                for chunk in chunks {
                    if chunk.offset_us < previous || chunk.offset_us > call.total_us {
                        return Err("trace_corrupt");
                    }
                    previous = chunk.offset_us;
                }
                if call.ttft_us != chunks.first().map(|c| c.offset_us) {
                    return Err("trace_corrupt");
                }
            } else if call.chunks.is_some() || call.ttft_us.is_some() {
                return Err("trace_corrupt");
            }
            if let Some(key) = call.key_hash.clone() {
                let repeat = call.repeat_index.ok_or("trace_corrupt")?;
                repeats.entry(key.clone()).or_default().push(repeat);
                if calls.insert((key, repeat), call).is_some() {
                    return Err("trace_corrupt");
                }
            } else if call.repeat_index.is_some() {
                return Err("trace_corrupt");
            }
        }
        for repeat in repeats.values_mut() {
            repeat.sort_unstable();
            if repeat.iter().enumerate().any(|(i, r)| i as u64 != *r) {
                return Err("trace_corrupt");
            }
        }
        let mut tools = HashMap::new();
        for line in json_lines(&dir.join("tools.jsonl"))? {
            if line.get("tool_name").and_then(Json::as_str).is_none()
                || line.get("args_sha256").and_then(Json::as_str).is_none()
            {
                return Err("trace_corrupt");
            }
            let key = tool_key(&line).ok_or("trace_corrupt")?;
            if tools.insert(key, line).is_some() {
                return Err("trace_corrupt");
            }
        }
        Ok(Self {
            calls,
            tools,
            directory: dir.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_append_is_an_error_without_a_second_write() {
        struct ShortWriter(usize);
        impl Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0 += 1;
                Ok(bytes.len() - 1)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut writer = ShortWriter(0);
        assert_eq!(
            write_line(&mut writer, &json!({"v":1})).unwrap_err().kind(),
            io::ErrorKind::WriteZero
        );
        assert_eq!(writer.0, 1);
    }
}
