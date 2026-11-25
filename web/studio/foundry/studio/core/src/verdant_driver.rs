#![forbid(unsafe_code)]
//! In-memory virtual-FS effect driver for the Lean-certified `verdant_studio` binary.
//!
//! Runs `verdant_studio::run` deterministically inside `core.wasm` and natively in tests over
//! caller-supplied UTF-8 workspace files, refusing path traversal (`..`), paths outside
//! `/workspace`, and OS process spawning.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::verdant_studio::{
    run as engine_run, work as engine_work, Adt0 as Session, Adt2 as Event, Adt3 as Start,
    Adt4 as Failure, Adt5 as Kind, Adt6 as Entry, Adt7 as Shape, Adt9 as Request, Bytes, List,
};

const MAX_STEPS: usize = 250_000;
const MAX_ARGS: usize = 32;
const MAX_ARG_BYTES: usize = 512;
const MAX_PATH_BYTES: usize = 512;
const MAX_PATH_SEGMENTS: usize = 32;
const MAX_FILES: usize = 256;
const MAX_FILE_BYTES: usize = 64 * 1024;
const MAX_INLINE_INPUT_BYTES: usize = 64 * 1024;
const MAX_TOTAL_VFS_BYTES: usize = 512 * 1024;
const MAX_LINES: usize = 512;
const MAX_CSV_LINES: usize = 1024;
const MAX_NESTING_DEPTH: usize = 48;
const MAX_JSON_STRUCT_ITEMS: usize = 1_000;
const MAX_XML_TAGS: usize = 400;
const MAX_XML_ATTRS: usize = 256;
const PROVENANCE_RAW: &str = include_str!("verdant_provenance.json");

struct VirtualFs {
    files: BTreeMap<String, Vec<u8>>,
    dirs: BTreeSet<String>,
    initial_files: BTreeSet<String>,
    seeded_files: BTreeSet<String>,
    mutated_files: BTreeSet<String>,
    deleted_files: BTreeSet<String>,
    total_bytes: usize,
    virtual_input_path: Option<String>,
}

enum HandleOutcome {
    Continue(Event),
    Exit {
        status: u8,
        stdout: String,
        stderr: String,
    },
    DriverError(String),
}

impl VirtualFs {
    fn new(
        initial: &BTreeMap<String, String>,
        auto_init: bool,
        arg_paths: &BTreeSet<String>,
    ) -> Result<Self, String> {
        if initial.len() > MAX_FILES {
            return Err(format!("too many workspace files (max {MAX_FILES})"));
        }
        let mut vfs = Self {
            files: BTreeMap::new(),
            dirs: BTreeSet::new(),
            initial_files: BTreeSet::new(),
            seeded_files: BTreeSet::new(),
            mutated_files: BTreeSet::new(),
            deleted_files: BTreeSet::new(),
            total_bytes: 0,
            virtual_input_path: None,
        };
        vfs.dirs.insert(String::new());
        for (raw_path, content) in initial {
            let norm = normalize_path(raw_path).map_err(|msg| format!("{msg}: {raw_path}"))?;
            if norm.is_empty() {
                return Err(format!("invalid file path: {raw_path}"));
            }
            if content.len() > MAX_FILE_BYTES {
                return Err(format!(
                    "workspace file {norm} exceeds {MAX_FILE_BYTES} byte limit"
                ));
            }
            if should_validate_complexity(&norm, Some(arg_paths)) {
                validate_content_complexity(content.as_bytes(), Some(&norm))
                    .map_err(|msg| format!("workspace file {norm}: {msg}"))?;
            }
            vfs.insert_initial_file(&norm, content.as_bytes())?;
        }
        if auto_init {
            vfs.seed_minimal_registers()?;
        }
        Ok(vfs)
    }

    fn insert_initial_file(&mut self, norm: &str, bytes: &[u8]) -> Result<(), String> {
        if self.dirs.contains(norm) || self.has_child_entries(norm) {
            return Err(format!("path conflicts with existing directory: {norm}"));
        }
        if let Some(conflict) = self.ancestor_file(norm) {
            return Err(format!("ancestor path is a file: {conflict}"));
        }
        let next_total = self
            .total_bytes
            .saturating_sub(self.files.get(norm).map_or(0, Vec::len))
            .saturating_add(bytes.len());
        if next_total > MAX_TOTAL_VFS_BYTES {
            return Err(format!(
                "workspace exceeds {MAX_TOTAL_VFS_BYTES} total byte limit"
            ));
        }
        if self.files.len() >= MAX_FILES && !self.files.contains_key(norm) {
            return Err(format!("workspace exceeds {MAX_FILES} file limit"));
        }
        self.ensure_ancestors(norm);
        self.initial_files.insert(norm.to_owned());
        self.files.insert(norm.to_owned(), bytes.to_vec());
        self.total_bytes = next_total;
        Ok(())
    }

    fn seed_minimal_registers(&mut self) -> Result<(), String> {
        for dir in [
            "model",
            "model/bindings",
            "model/encodings",
            "model/mappings",
            "model/matrix",
            "model/queue",
            "catalog",
            "app",
            "app/src",
            "app/src/Catalog",
            "studio",
            "studio/src",
            "stdlib",
            "stdlib/src",
        ] {
            if self.files.contains_key(dir) || self.ancestor_file(dir).is_some() {
                continue;
            }
            self.ensure_ancestors(dir);
            self.dirs.insert(dir.to_owned());
        }
        let defaults: &[(&str, &str)] = &[
            ("model/requirements.json", "[]\n"),
            ("model/admissions.json", "[]\n"),
            ("model/owner-decisions.json", "[]\n"),
            ("model/strata.json", "[]\n"),
            ("model/register-lock.json", "[]\n"),
            ("model/authorities.toml", ""),
            ("model/ids.toml", "spec = \"template/2\"\n"),
            (
                "app/lexlean.toml",
                "spec = \"lexlean/project/1\"\nname = \"verdant-studio-core\"\nlanguage = \"1.2\"\nmodule_prefix = \"VerdantStudio\"\nsource_roots = [\"src\"]\nentrypoints = [\"src/Contract.lex.tex\"]\n",
            ),
            (
                "catalog/standards-index.csv",
                "id,domain,name,owner,phases,scope,adoption,adoption_code,source\n",
            ),
        ];
        for &(path, content) in defaults {
            if !self.files.contains_key(path)
                && !self.is_dir(path)
                && self.ancestor_file(path).is_none()
            {
                let next_total = self.total_bytes.saturating_add(content.len());
                if next_total > MAX_TOTAL_VFS_BYTES || self.files.len() >= MAX_FILES {
                    return Err("workspace limit exceeded during auto_init".to_owned());
                }
                self.ensure_ancestors(path);
                self.files
                    .insert(path.to_owned(), content.as_bytes().to_vec());
                self.seeded_files.insert(path.to_owned());
                self.total_bytes = next_total;
            }
        }
        Ok(())
    }

    fn ancestor_file(&self, norm: &str) -> Option<String> {
        let mut cursor = String::new();
        let parts: Vec<&str> = norm.split('/').collect();
        for part in parts.iter().take(parts.len().saturating_sub(1)) {
            if part.is_empty() {
                continue;
            }
            if !cursor.is_empty() {
                cursor.push('/');
            }
            cursor.push_str(part);
            if self.files.contains_key(&cursor) {
                return Some(cursor);
            }
        }
        None
    }

    fn has_child_entries(&self, norm: &str) -> bool {
        let prefix = format!("{norm}/");
        self.files.keys().any(|k| k.starts_with(&prefix))
            || self.dirs.iter().any(|d| d.starts_with(&prefix))
    }

    fn ensure_ancestors(&mut self, norm: &str) {
        let mut cursor = String::new();
        let parts: Vec<&str> = norm.split('/').collect();
        for part in parts.iter().take(parts.len().saturating_sub(1)) {
            if part.is_empty() {
                continue;
            }
            if !cursor.is_empty() {
                cursor.push('/');
            }
            cursor.push_str(part);
            self.dirs.insert(cursor.clone());
        }
    }

    fn is_dir(&self, norm: &str) -> bool {
        if norm.is_empty() {
            return true;
        }
        if self.files.contains_key(norm) {
            return false;
        }
        if self.dirs.contains(norm) {
            return true;
        }
        let prefix = format!("{norm}/");
        self.files.keys().any(|k| k.starts_with(&prefix))
            || self.dirs.iter().any(|d| d.starts_with(&prefix))
    }

    fn is_virtual_input(&self, norm: &str) -> bool {
        self.virtual_input_path
            .as_deref()
            .is_some_and(|p| p == norm)
    }

    fn record_write(&mut self, norm: &str) {
        if self.is_virtual_input(norm) {
            return;
        }
        self.deleted_files.remove(norm);
        self.mutated_files.insert(norm.to_owned());
    }

    fn record_delete(&mut self, norm: &str) {
        if self.is_virtual_input(norm) {
            return;
        }
        self.mutated_files.remove(norm);
        if self.initial_files.contains(norm) {
            self.deleted_files.insert(norm.to_owned());
        }
    }

    fn handle(&mut self, request: Request) -> HandleOutcome {
        match request {
            Request::C0(path) => self.files_read(path.text()),
            Request::C1(path, data) => self.files_write(path.text(), data.octets()),
            Request::C2(path) => self.files_list(path.text()),
            Request::C3(path) => self.files_stat(path.text()),
            Request::C4(path) => self.files_create_dir(path.text()),
            Request::C5(path) => self.files_remove_file(path.text()),
            Request::C6(path) => self.files_remove_tree(path.text()),
            Request::C7(from, to) => self.files_rename(from.text(), to.text()),
            Request::C8(path) => self.files_canonical(path.text()),
            Request::C9(path) => self.files_lock(path.text()),
            Request::C10(_) => HandleOutcome::Continue(Event::C6(Err(failure_permission(
                "process execution is disabled in browser WASM",
            )))),
            Request::C11(_, _) => HandleOutcome::Continue(Event::C7(Err(failure_permission(
                "process wait is disabled in browser WASM",
            )))),
            Request::C12(_) => HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "process signal is disabled in browser WASM",
            )))),
            Request::C13(status, stdout, stderr) => HandleOutcome::Exit {
                status,
                stdout: String::from_utf8_lossy(stdout.octets()).into_owned(),
                stderr: String::from_utf8_lossy(stderr.octets()).into_owned(),
            },
        }
    }

    fn files_read(&self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if norm.is_empty() || self.is_dir(&norm) {
            return HandleOutcome::Continue(Event::C1(Err(failure_permission("path is a directory"))));
        }
        match self.files.get(&norm) {
            Some(bytes) => HandleOutcome::Continue(Event::C1(Ok(Bytes::lit(bytes)))),
            None => HandleOutcome::Continue(Event::C1(Err(failure_not_found("file not found")))),
        }
    }

    fn files_write(&mut self, raw: &str, octets: &[u8]) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if norm.is_empty() {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "cannot overwrite workspace root",
            ))));
        }
        if self.is_dir(&norm) || self.ancestor_file(&norm).is_some() {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "path conflicts with directory or file ancestor",
            ))));
        }
        if octets.len() > MAX_FILE_BYTES {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "file exceeds virtual workspace size limit",
            ))));
        }
        if should_validate_complexity(&norm, None) {
            if let Err(msg) = validate_content_complexity(octets, Some(&norm)) {
                return HandleOutcome::DriverError(format!(
                    "written file {norm} exceeds reload limits: {msg}"
                ));
            }
        }
        let prev_len = self.files.get(&norm).map_or(0, Vec::len);
        if self.files.len() >= MAX_FILES && !self.files.contains_key(&norm) {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "virtual workspace file count limit exceeded",
            ))));
        }
        let next_total = self
            .total_bytes
            .saturating_sub(prev_len)
            .saturating_add(octets.len());
        if next_total > MAX_TOTAL_VFS_BYTES {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "virtual workspace total size limit exceeded",
            ))));
        }
        self.ensure_ancestors(&norm);
        self.files.insert(norm.clone(), octets.to_vec());
        self.total_bytes = next_total;
        self.record_write(&norm);
        HandleOutcome::Continue(Event::C2(Ok(())))
    }

    fn files_list(&self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if !self.is_dir(&norm) {
            return HandleOutcome::Continue(Event::C3(Err(failure_not_found(
                "directory not found",
            ))));
        }
        let prefix = if norm.is_empty() {
            String::new()
        } else {
            format!("{norm}/")
        };
        let mut children: BTreeSet<String> = BTreeSet::new();
        for path in self.files.keys() {
            if let Some(rest) = path.strip_prefix(&prefix) {
                if !rest.is_empty() {
                    let head = rest.split('/').next().unwrap_or(rest);
                    children.insert(head.to_owned());
                }
            }
        }
        for dir in &self.dirs {
            if let Some(rest) = dir.strip_prefix(&prefix) {
                if !rest.is_empty() {
                    let head = rest.split('/').next().unwrap_or(rest);
                    children.insert(head.to_owned());
                }
            }
        }
        let items: Vec<Vec<u8>> = children.into_iter().map(String::into_bytes).collect();
        HandleOutcome::Continue(Event::C3(Ok(bytes_list(items))))
    }

    fn files_stat(&self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if let Some(bytes) = self.files.get(&norm) {
            return HandleOutcome::Continue(Event::C4(Ok(Entry::C0(
                Shape::C0,
                bytes.len() as u64,
                Some((1, synthetic_inode(&norm))),
            ))));
        }
        if self.is_dir(&norm) {
            return HandleOutcome::Continue(Event::C4(Ok(Entry::C0(
                Shape::C1,
                0,
                Some((1, synthetic_inode(&norm))),
            ))));
        }
        HandleOutcome::Continue(Event::C4(Err(failure_not_found("path not found"))))
    }

    fn files_create_dir(&mut self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if self.files.contains_key(&norm) || self.ancestor_file(&norm).is_some() {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "cannot create directory over existing file",
            ))));
        }
        if !norm.is_empty() {
            self.ensure_ancestors(&norm);
            self.dirs.insert(norm);
        }
        HandleOutcome::Continue(Event::C2(Ok(())))
    }

    fn files_remove_file(&mut self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if let Some(bytes) = self.files.remove(&norm) {
            self.total_bytes = self.total_bytes.saturating_sub(bytes.len());
            self.record_delete(&norm);
            HandleOutcome::Continue(Event::C2(Ok(())))
        } else {
            HandleOutcome::Continue(Event::C2(Err(failure_not_found("file not found"))))
        }
    }

    fn files_remove_tree(&mut self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if norm.is_empty() {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "cannot remove workspace root",
            ))));
        }
        let prefix = format!("{norm}/");
        let to_remove: Vec<String> = self
            .files
            .keys()
            .filter(|k| *k == &norm || k.starts_with(&prefix))
            .cloned()
            .collect();
        let dir_remove: Vec<String> = self
            .dirs
            .iter()
            .filter(|d| *d == &norm || d.starts_with(&prefix))
            .cloned()
            .collect();
        if to_remove.is_empty() && dir_remove.is_empty() {
            return HandleOutcome::Continue(Event::C2(Err(failure_not_found("path not found"))));
        }
        for path in to_remove {
            if let Some(bytes) = self.files.remove(&path) {
                self.total_bytes = self.total_bytes.saturating_sub(bytes.len());
            }
            self.record_delete(&path);
        }
        for d in dir_remove {
            self.dirs.remove(&d);
        }
        HandleOutcome::Continue(Event::C2(Ok(())))
    }

    fn files_rename(&mut self, from_raw: &str, to_raw: &str) -> HandleOutcome {
        let from = match normalize_path(from_raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {from_raw}")),
        };
        let to = match normalize_path(to_raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {to_raw}")),
        };
        if from.is_empty() || to.is_empty() {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "cannot rename workspace root",
            ))));
        }
        if from == to {
            return if self.files.contains_key(&from) || self.is_dir(&from) {
                HandleOutcome::Continue(Event::C2(Ok(())))
            } else {
                HandleOutcome::Continue(Event::C2(Err(failure_not_found("source path not found"))))
            };
        }
        if let Some(bytes) = self.files.get(&from).cloned() {
            if self.is_dir(&to) || self.ancestor_file(&to).is_some() {
                return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                    "destination conflicts with directory or ancestor file",
                ))));
            }
            self.files.remove(&from);
            self.record_delete(&from);
            let displaced = self.files.insert(to.clone(), bytes).map_or(0, |b| b.len());
            self.total_bytes = self.total_bytes.saturating_sub(displaced);
            self.ensure_ancestors(&to);
            self.record_write(&to);
            return HandleOutcome::Continue(Event::C2(Ok(())));
        }
        if self.is_dir(&from) {
            let from_prefix = format!("{from}/");
            if to.starts_with(&from_prefix) {
                return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                    "cannot move directory into itself",
                ))));
            }
            if self.files.contains_key(&to) || self.ancestor_file(&to).is_some() {
                return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                    "destination conflicts with existing file",
                ))));
            }
            let moving_files: Vec<String> = self
                .files
                .keys()
                .filter(|k| k.starts_with(&from_prefix))
                .cloned()
                .collect();
            let moving_dirs: Vec<String> = self
                .dirs
                .iter()
                .filter(|d| *d == &from || d.starts_with(&from_prefix))
                .cloned()
                .collect();
            for old_dir in &moving_dirs {
                if old_dir != &from {
                    if let Some(suffix) = old_dir.strip_prefix(&from_prefix) {
                        let new_dir = format!("{to}/{suffix}");
                        if self.files.contains_key(&new_dir) || self.ancestor_file(&new_dir).is_some()
                        {
                            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                                "destination subdirectory conflicts with existing file",
                            ))));
                        }
                    }
                }
            }
            for old_path in &moving_files {
                if let Some(suffix) = old_path.strip_prefix(&from_prefix) {
                    let new_path = format!("{to}/{suffix}");
                    if self.is_dir(&new_path) || self.ancestor_file(&new_path).is_some() {
                        return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                            "destination file conflicts with existing directory",
                        ))));
                    }
                }
            }
            self.ensure_ancestors(&to);
            self.dirs.insert(to.clone());
            for old_dir in moving_dirs {
                self.dirs.remove(&old_dir);
                if old_dir != from {
                    if let Some(suffix) = old_dir.strip_prefix(&from_prefix) {
                        self.dirs.insert(format!("{to}/{suffix}"));
                    }
                }
            }
            for old_path in moving_files {
                if let Some(bytes) = self.files.remove(&old_path) {
                    let suffix = old_path.strip_prefix(&from_prefix).unwrap_or("");
                    let new_path = format!("{to}/{suffix}");
                    self.record_delete(&old_path);
                    self.ensure_ancestors(&new_path);
                    let displaced = self.files.insert(new_path.clone(), bytes).map_or(0, |b| b.len());
                    self.total_bytes = self.total_bytes.saturating_sub(displaced);
                    self.record_write(&new_path);
                }
            }
            return HandleOutcome::Continue(Event::C2(Ok(())));
        }
        HandleOutcome::Continue(Event::C2(Err(failure_not_found("source path not found"))))
    }

    fn files_canonical(&self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if self.files.contains_key(&norm) || self.is_dir(&norm) {
            let abs = if norm.is_empty() {
                "/workspace".to_owned()
            } else {
                format!("/workspace/{norm}")
            };
            HandleOutcome::Continue(Event::C5(Ok(Bytes::lit(abs.as_bytes()))))
        } else {
            HandleOutcome::Continue(Event::C5(Err(failure_not_found("path not found"))))
        }
    }

    fn files_lock(&mut self, raw: &str) -> HandleOutcome {
        let norm = match normalize_path(raw) {
            Ok(p) => p,
            Err(err) => return HandleOutcome::DriverError(format!("{err}: {raw}")),
        };
        if norm.is_empty() {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "cannot lock workspace root",
            ))));
        }
        if self.is_dir(&norm) || self.ancestor_file(&norm).is_some() {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "lock path conflicts with directory or ancestor file",
            ))));
        }
        if self.files.len() >= MAX_FILES && !self.files.contains_key(&norm) {
            return HandleOutcome::Continue(Event::C2(Err(failure_permission(
                "virtual workspace file count limit exceeded",
            ))));
        }
        self.ensure_ancestors(&norm);
        self.files.entry(norm).or_default();
        HandleOutcome::Continue(Event::C2(Ok(())))
    }
}

fn normalize_path(raw: &str) -> Result<String, &'static str> {
    if raw.len() > MAX_PATH_BYTES {
        return Err("path exceeds maximum length");
    }
    if raw.bytes().any(|b| b <= 0x1f || b == 0x7f || b == b':') {
        return Err("invalid character in workspace path");
    }
    let slash = raw.replace('\\', "/");
    let stripped = if slash == "/workspace" {
        ""
    } else if let Some(rest) = slash.strip_prefix("/workspace/") {
        rest
    } else if slash.starts_with('/') {
        return Err("absolute path outside /workspace is not permitted");
    } else {
        slash.as_str()
    };
    let mut parts: Vec<&str> = Vec::new();
    for seg in stripped.split('/') {
        match seg {
            "" | "." => {}
            ".." => return Err("path traversal is not permitted"),
            other => {
                if parts.len() >= MAX_PATH_SEGMENTS {
                    return Err("path exceeds maximum directory depth");
                }
                parts.push(other);
            }
        }
    }
    Ok(parts.join("/"))
}

fn should_validate_complexity(norm: &str, arg_paths: Option<&BTreeSet<String>>) -> bool {
    norm.ends_with(".json")
        || norm.ends_with(".jsonl")
        || norm.ends_with(".toml")
        || norm.ends_with(".csv")
        || norm.ends_with(".xml")
        || norm.ends_with(".xsd")
        || norm.ends_with(".lex.tex")
        || arg_paths.is_some_and(|set| set.contains(norm))
}

fn validate_content_complexity(raw_bytes: &[u8], path_hint: Option<&str>) -> Result<(), String> {
    let bytes = raw_bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(raw_bytes);
    let first_non_ws = bytes.iter().copied().find(|b| !b.is_ascii_whitespace());

    let is_csv = path_hint.is_some_and(|p| p.ends_with(".csv"));
    let is_toml = path_hint.is_some_and(|p| p.ends_with(".toml"));
    let is_xml_like = path_hint.is_some_and(|p| p.ends_with(".xml") || p.ends_with(".xsd"))
        || (!is_csv && !is_toml && first_non_ws == Some(b'<'));
    let is_jsonl = path_hint.is_some_and(|p| p.ends_with(".jsonl"))
        || (!is_csv && !is_toml && !is_xml_like && bytes.contains(&b'\n') && first_non_ws == Some(b'{'));
    let is_json_like = path_hint.is_some_and(|p| p.ends_with(".json") || p.ends_with(".jsonl"))
        || (!is_csv && !is_toml && !is_xml_like && matches!(first_non_ws, Some(b'{') | Some(b'[')));

    let line_limit = if is_csv { MAX_CSV_LINES } else { MAX_LINES };
    let line_count = bytes.iter().filter(|&&b| b == b'\n').count() + 1;
    if line_count > line_limit {
        return Err(format!("line count exceeds limit ({line_limit})"));
    }

    if is_json_like {
        let mut in_string = false;
        let mut escape = false;
        let mut json_depth: usize = 0;
        let mut line_struct_items: usize = 0;

        for &b in bytes {
            if in_string {
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_string = false;
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'{' | b'[' => {
                    json_depth += 1;
                    if json_depth > MAX_NESTING_DEPTH {
                        return Err(format!(
                            "nesting depth exceeds limit ({MAX_NESTING_DEPTH})"
                        ));
                    }
                    line_struct_items += 1;
                    if line_struct_items > MAX_JSON_STRUCT_ITEMS {
                        return Err(format!(
                            "structural item count exceeds limit ({MAX_JSON_STRUCT_ITEMS})"
                        ));
                    }
                }
                b'}' | b']' => {
                    json_depth = json_depth.saturating_sub(1);
                }
                b',' => {
                    line_struct_items += 1;
                    if line_struct_items > MAX_JSON_STRUCT_ITEMS {
                        return Err(format!(
                            "structural item count exceeds limit ({MAX_JSON_STRUCT_ITEMS})"
                        ));
                    }
                }
                b'\n' if is_jsonl && json_depth == 0 => {
                    line_struct_items = 0;
                }
                _ => {}
            }
        }
    }

    // Bound XML/XSD unclosed element nesting, tag count (`f232`), and attribute count (`f330`).
    if is_xml_like {
        let mut xml_depth: usize = 0;
        let mut xml_tags: usize = 0;
        let mut xml_attrs: usize = 0;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'<' {
                if bytes[i..].starts_with(b"<!--") {
                    if let Some(end) = find_subslice(&bytes[i + 4..], b"-->") {
                        i = i + 4 + end + 3;
                        continue;
                    }
                    break;
                }
                if bytes[i..].starts_with(b"<![CDATA[") {
                    if let Some(end) = find_subslice(&bytes[i + 9..], b"]]>") {
                        i = i + 9 + end + 3;
                        continue;
                    }
                    break;
                }
                xml_tags += 1;
                if xml_tags > MAX_XML_TAGS {
                    return Err(format!("XML tag count exceeds limit ({MAX_XML_TAGS})"));
                }
                if bytes.get(i + 1) == Some(&b'/') {
                    xml_depth = xml_depth.saturating_sub(1);
                } else if !matches!(bytes.get(i + 1), Some(b'?') | Some(b'!')) {
                    let mut j = i + 1;
                    let mut quote: Option<u8> = None;
                    let mut prev_non_ws = bytes[i];
                    while j < bytes.len() {
                        let b = bytes[j];
                        if let Some(q) = quote {
                            if b == q {
                                quote = None;
                            }
                        } else if b == b'"' || b == b'\'' {
                            quote = Some(b);
                        } else if b == b'=' {
                            xml_attrs += 1;
                            if xml_attrs > MAX_XML_ATTRS {
                                return Err(format!(
                                    "XML attribute count exceeds limit ({MAX_XML_ATTRS})"
                                ));
                            }
                        } else if b == b'>' {
                            break;
                        }
                        if !b.is_ascii_whitespace() {
                            prev_non_ws = b;
                        }
                        j += 1;
                    }
                    let self_closing = j < bytes.len() && j > i + 1 && prev_non_ws == b'/';
                    if !self_closing {
                        xml_depth += 1;
                        if xml_depth > MAX_NESTING_DEPTH {
                            return Err(format!(
                                "XML nesting depth exceeds limit ({MAX_NESTING_DEPTH})"
                            ));
                        }
                    }
                    i = j.min(bytes.len());
                }
            }
            i += 1;
        }
    }

    Ok(())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn synthetic_inode(norm: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in norm.as_bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash.max(1)
}

fn failure_permission(msg: &str) -> Failure {
    Failure::C0(Kind::C2, Bytes::lit(msg.as_bytes()))
}

fn failure_not_found(msg: &str) -> Failure {
    Failure::C0(Kind::C0, Bytes::lit(msg.as_bytes()))
}

fn bytes_list(mut items: Vec<Vec<u8>>) -> List<Bytes> {
    let mut out = List::nil();
    while let Some(last) = items.pop() {
        out = List::cons(Bytes::lit(&last), out);
    }
    out
}

fn pairs_list(mut items: Vec<(Vec<u8>, Vec<u8>)>) -> List<(Bytes, Bytes)> {
    let mut out = List::nil();
    while let Some((k, v)) = items.pop() {
        out = List::cons((Bytes::lit(&k), Bytes::lit(&v)), out);
    }
    out
}

fn build_start(args: &[String]) -> Event {
    let arg_bytes: Vec<Vec<u8>> = args.iter().map(|s| s.as_bytes().to_vec()).collect();
    let dir = Ok(Bytes::lit(b"/workspace"));
    let exe = Ok(Bytes::lit(b"/workspace/verdant_studio"));
    let env_pairs = Vec::new();
    Event::C0(Start::C0(
        bytes_list(arg_bytes),
        dir,
        exe,
        pairs_list(env_pairs),
        false,
    ))
}

fn virtual_input_path_for(op: Option<&str>) -> &'static str {
    match op.unwrap_or("") {
        "admit"
        | "record-decision"
        | "stage-bindings"
        | "stage-mappings"
        | "stage-matrix"
        | "stage-admissions"
        | "stage-decisions"
        | "stage-encodings" => ".verdant-input.jsonl",
        "propose-encoding" | "admit-encoding" | "merge-gate" => ".verdant-input.json",
        "ingest-index" => ".verdant-input.csv",
        "read-xml" => ".verdant-input.xml",
        "read-xsd" => ".verdant-input.xsd",
        _ => ".verdant-input",
    }
}

fn resolve_args(raw_args: &[String], virtual_input_path: Option<&str>) -> Vec<String> {
    let Some(vpath) = virtual_input_path else {
        return raw_args.to_vec();
    };
    let mut out: Vec<String> = raw_args
        .iter()
        .map(|a| {
            if a == "-" {
                vpath.to_owned()
            } else {
                a.clone()
            }
        })
        .collect();
    if !out.is_empty() {
        let op = out[0].as_str();
        let single_file_op = matches!(
            op,
            "admit"
                | "stage-bindings"
                | "stage-mappings"
                | "stage-matrix"
                | "stage-admissions"
                | "stage-decisions"
                | "stage-encodings"
                | "admit-encoding"
                | "record-decision"
                | "ingest-index"
                | "ingest-authority"
                | "merge-gate"
                | "read-xml"
                | "read-xsd"
        );
        if single_file_op && out.len() == 1 {
            out.push(vpath.to_owned());
        } else if op == "propose-encoding"
            && out.len() >= 3
            && out[1] != vpath
            && !out[1].ends_with(".json")
            && !out[1].contains('/')
        {
            out.insert(1, vpath.to_owned());
        }
    }
    out
}

fn provenance_json() -> Value {
    serde_json::from_str(PROVENANCE_RAW).unwrap_or_else(|_| json!({}))
}

fn error_result(msg: impl Into<String>) -> Value {
    let text = msg.into();
    json!({
        "ok": false,
        "status": 1,
        "error": text,
        "stdout": "",
        "stderr": text,
        "written_files": {},
        "deleted_files": [],
        "provenance": provenance_json(),
    })
}

/// Run a `verdant_studio` CLI operation against a virtual workspace described as JSON:
/// `{ args: [...], files?: { path: text }, inline_input?: string, auto_init?: bool }`.
pub fn run_from_json(value: &Value) -> Value {
    let raw_args: Vec<String> = match value["args"].as_array() {
        Some(arr) => {
            if arr.len() > MAX_ARGS {
                return error_result(format!("args length exceeds limit ({MAX_ARGS})"));
            }
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let Some(s) = item.as_str() else {
                    return error_result("args must contain only strings");
                };
                if s.len() > MAX_ARG_BYTES {
                    return error_result(format!(
                        "argument exceeds maximum length ({MAX_ARG_BYTES} bytes)"
                    ));
                }
                if s.bytes().any(|b| b == 0) {
                    return error_result("argument contains NUL byte");
                }
                out.push(s.to_owned());
            }
            out
        }
        None => return error_result("args must be an array of strings"),
    };

    let mut initial = BTreeMap::new();
    if !value["files"].is_null() {
        let Some(obj) = value["files"].as_object() else {
            return error_result("files must be an object mapping paths to strings");
        };
        if obj.len() > MAX_FILES {
            return error_result(format!("too many workspace files (max {MAX_FILES})"));
        }
        for (k, v) in obj {
            let Some(text) = v.as_str() else {
                return error_result(format!("file {k} content must be a string"));
            };
            initial.insert(k.clone(), text.to_owned());
        }
    }

    let auto_init = value["auto_init"].as_bool().unwrap_or(true);
    let inline_input = if value["inline_input"].is_null() {
        None
    } else if let Some(s) = value["inline_input"].as_str() {
        Some(s)
    } else {
        return error_result("inline_input must be a string");
    };

    let vpath = inline_input.map(|_| virtual_input_path_for(raw_args.first().map(String::as_str)));
    let args = resolve_args(&raw_args, vpath);
    let arg_paths: BTreeSet<String> = args
        .iter()
        .skip(1)
        .filter_map(|a| normalize_path(a).ok())
        .filter(|p| !p.is_empty())
        .collect();

    let mut vfs = match VirtualFs::new(&initial, auto_init, &arg_paths) {
        Ok(v) => v,
        Err(err) => return error_result(err),
    };
    if let (Some(input_text), Some(virtual_path)) = (inline_input, vpath) {
        if input_text.len() > MAX_INLINE_INPUT_BYTES {
            return error_result(format!(
                "inline_input exceeds {MAX_INLINE_INPUT_BYTES} byte limit"
            ));
        }
        if vfs.files.contains_key(virtual_path) || vfs.is_dir(virtual_path) {
            return error_result(format!(
                "files must not contain reserved inline_input path {virtual_path}"
            ));
        }
        if let Err(err) = validate_content_complexity(input_text.as_bytes(), Some(virtual_path)) {
            return error_result(format!("inline_input: {err}"));
        }
        let next_total = vfs.total_bytes.saturating_add(input_text.len());
        if next_total > MAX_TOTAL_VFS_BYTES || vfs.files.len() >= MAX_FILES {
            return error_result("inline_input exceeds virtual workspace limits");
        }
        vfs.virtual_input_path = Some(virtual_path.to_owned());
        vfs.files
            .insert(virtual_path.to_owned(), input_text.as_bytes().to_vec());
        vfs.total_bytes = next_total;
    }

    let work_before = engine_work();
    let mut state: Option<Session> = None;
    let mut event = build_start(&args);
    let mut steps = 0usize;

    let (status, stdout, stderr) = loop {
        if steps >= MAX_STEPS {
            return json!({
                "ok": false,
                "status": 124,
                "error": "verdant_studio exceeded virtual step budget",
                "stdout": "",
                "stderr": "verdant_studio exceeded virtual step budget",
                "written_files": {},
                "deleted_files": [],
                "steps": steps,
                "provenance": provenance_json(),
            });
        }
        steps += 1;
        match engine_run(state, event) {
            Ok((kept, request)) => {
                state = Some(kept);
                match vfs.handle(request) {
                    HandleOutcome::Continue(next_event) => event = next_event,
                    HandleOutcome::Exit {
                        status,
                        stdout,
                        stderr,
                    } => break (status, stdout, stderr),
                    HandleOutcome::DriverError(err) => break (1, String::new(), err),
                }
            }
            Err(_) => {
                break (
                    70,
                    String::new(),
                    "lexlean: arithmetic overflow in verdant_studio".to_owned(),
                );
            }
        }
    };

    let mut written_files: BTreeMap<String, String> = BTreeMap::new();
    let mut deleted_files: Vec<String> = Vec::new();
    if status == 0 {
        if !vfs.mutated_files.is_empty() {
            let mut export_paths: BTreeSet<String> = vfs.mutated_files.clone();
            export_paths.extend(vfs.seeded_files.iter().cloned());
            for path in export_paths {
                if let Some(bytes) = vfs.files.get(&path) {
                    written_files.insert(path, String::from_utf8_lossy(bytes).into_owned());
                }
            }
        }
        deleted_files.extend(vfs.deleted_files.into_iter());
    }

    json!({
        "ok": status == 0,
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "written_files": written_files,
        "deleted_files": deleted_files,
        "work_units": engine_work().saturating_sub(work_before),
        "steps": steps,
        "provenance": provenance_json(),
    })
}
