"""Download 16 ClimbMix train Parquets plus nanochat's held-out Parquet.

Adapted from Karpathy's fineweb.py approach: GPT-2 tokenization, one
<|endoftext|> before each document, uint16 tokens, and NumPy shards.
Each source Parquet keeps its own output shards, so validation cannot
be mixed into training. Output chunks contain at most 100M tokens.

Run: python climbmix.py --workers 15
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

REPO_ID = "karpathy/climbmix-400b-shuffle"
TRAIN_IDS = tuple(range(16))
VAL_ID = 6542
ENCODING = "gpt2"
_encoding = None


def init_tokenizer():
    global _encoding
    _encoding = tiktoken.get_encoding(ENCODING)


def tokenize(text):
    # encode_ordinary handles literal special-token strings as ordinary text.
    tokens = [_encoding.eot_token]
    tokens.extend(_encoding.encode_ordinary(text))
    array = np.asarray(tokens)
    if array.min() < 0 or array.max() >= 2**16:
        raise ValueError("GPT-2 token IDs must fit in uint16")
    return array.astype(np.uint16)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_datafile(path, tokens):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, tokens, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "filename": path.name,
        "tokens": int(tokens.size),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_tokens(output_dir, record, vocab_size):
    total_tokens = 0
    total_eot = 0
    for item in record["outputs"]:
        path = output_dir / item["filename"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Checksum or size mismatch: {path}")
        tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        if tokens.dtype != np.uint16 or tokens.shape != (item["tokens"],):
            raise ValueError(f"Invalid NumPy token shard: {path}")
        if tokens.size == 0 or int(tokens.max()) >= vocab_size:
            raise ValueError(f"Invalid GPT-2 token IDs: {path}")
        total_tokens += tokens.size
        total_eot += int(np.count_nonzero(tokens == _encoding.eot_token))
    if total_tokens != record["tokens"] or total_eot != record["documents"]:
        raise ValueError(f"Token or document-marker count mismatch: {record['source']}")


def prepare_source(source, source_info, output_dir, shard_size, pool):
    split = source_info["split"]
    source_id = source_info["index"]
    parquet = pq.ParquetFile(source)
    buffer = np.empty(shard_size, dtype=np.uint16)
    used = 0
    outputs = []
    documents = 0
    total_tokens = 0

    def flush_buffer():
        filename = f"climbmix_{split}_{source_id:05d}_{len(outputs):03d}.npy"
        outputs.append(write_datafile(output_dir / filename, buffer[:used]))

    with tqdm(total=parquet.metadata.num_rows, desc=f"{split} {source_id:05d}", unit="docs") as progress:
        # Bounded batches keep the multiprocessing input queue from loading
        # the entire corpus into memory. imap preserves document order.
        for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
            texts = batch.column(0).to_pylist()
            for tokens in pool.imap(tokenize, texts, chunksize=8):
                documents += 1
                total_tokens += tokens.size
                offset = 0
                # Unlike the lecture's single split, this also supports a
                # document that spans several output chunks.
                while offset < tokens.size:
                    take = min(shard_size - used, tokens.size - offset)
                    buffer[used:used + take] = tokens[offset:offset + take]
                    used += take
                    offset += take
                    if used == shard_size:
                        flush_buffer()
                        used = 0
                progress.update(1)
    if used:
        flush_buffer()
    if documents != parquet.metadata.num_rows:
        raise ValueError(f"Not all documents were tokenized: {source}")
    record = {
        "source": source.name,
        "source_sha256": source_info["sha256"],
        "split": split,
        "documents": documents,
        "tokens": int(total_tokens),
        "outputs": outputs,
    }
    verify_tokens(output_dir, record, _encoding.n_vocab)
    print(f"Prepared {source.name}: {documents:,} documents, {total_tokens:,} tokens", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "climbmix_small")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.workers < 1 or args.shard_size < 1:
        parser.error("workers and shard-size must be positive")
    root = args.output_dir.resolve()
    raw_dir, token_dir = root / "raw", root / "tokens"
    raw_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    # Generated data stays out of git without editing the repository .gitignore.
    (root / ".gitignore").write_text("*\n")
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / "manifest.json"
        config = {
            "repo_id": REPO_ID,
            "train_ids": list(TRAIN_IDS),
            "validation_id": VAL_ID,
            "tokenizer": ENCODING,
            "dtype": "uint16",
            "shard_size": args.shard_size,
            "document_separator": "<|endoftext|> before each document",
            "source_url": f"https://huggingface.co/datasets/{REPO_ID}",
            "original_source_url": "https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix",
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["config"] != config:
                raise ValueError("Existing manifest uses different settings; choose a new output directory")
            revision = manifest["revision"]
        else:
            revision = "main"
            manifest = {"config": config, "prepared": [], "started_utc": datetime.now(timezone.utc).isoformat()}
        info = HfApi().dataset_info(REPO_ID, revision=revision, files_metadata=True)
        manifest["revision"] = info.sha  # pin all 17 files to the same full commit
        files = [f"shard_{i:05d}.parquet" for i in (*TRAIN_IDS, VAL_ID)]
        remote = {item.rfilename: item for item in info.siblings}
        selected = []
        for index, filename in zip((*TRAIN_IDS, VAL_ID), files):
            item = remote[filename]
            if item.size is None or item.lfs is None:
                raise ValueError(f"Missing download size/checksum metadata: {filename}")
            selected.append({
                "filename": filename, "index": index,
                "split": "val" if index == VAL_ID else "train",
                "bytes": item.size, "sha256": item.lfs.sha256,
            })
        manifest["raw_files"] = selected
        manifest["status"] = "downloading"
        write_json(manifest_path, manifest)
        print(f"Downloading exactly {len(files)} files ({sum(f['bytes'] for f in selected) / 1e9:.2f} GB)", flush=True)
        print(f"Dataset revision: {info.sha}\nOutput: {root}", flush=True)
        snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", revision=info.sha,
            allow_patterns=files, local_dir=raw_dir, max_workers=4,
        )
        for item in selected:
            path = raw_dir / item["filename"]
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Downloaded file fails upstream checksum: {path}")
            parquet = pq.ParquetFile(path)
            if "text" not in parquet.schema_arrow.names or parquet.metadata.num_rows == 0:
                raise ValueError(f"Invalid text Parquet: {path}")
            item["documents"] = parquet.metadata.num_rows
        manifest["status"] = "tokenizing"
        write_json(manifest_path, manifest)
        init_tokenizer()
        manifest["eot_token_id"] = _encoding.eot_token
        completed = {record["source"]: record for record in manifest["prepared"]}
        with mp.get_context("spawn").Pool(args.workers, initializer=init_tokenizer) as pool:
            for item in selected:
                filename = item["filename"]
                if filename in completed:
                    record = completed[filename]
                    if record["source_sha256"] != item["sha256"]:
                        raise ValueError(f"Source revision changed: {filename}")
                    verify_tokens(token_dir, record, _encoding.n_vocab)
                    print(f"Verified existing token shards for {filename}", flush=True)
                    continue
                record = prepare_source(raw_dir / filename, item, token_dir, args.shard_size, pool)
                manifest["prepared"].append(record)
                write_json(manifest_path, manifest)
        manifest["train_tokens"] = sum(r["tokens"] for r in manifest["prepared"] if r["split"] == "train")
        manifest["validation_tokens"] = sum(r["tokens"] for r in manifest["prepared"] if r["split"] == "val")
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print(f"COMPLETE: {manifest['train_tokens']:,} training tokens + {manifest['validation_tokens']:,} validation tokens", flush=True)
        print(f"Verified manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
