#!/usr/bin/env python3
import csv
import os
import re
import shlex
import subprocess
import time
from statistics import mean, median
from pathlib import Path


def find_inputs(data_root: Path, species_list, length_list, reps):
    inputs = []
    for taxa in species_list:
        for seq_len in length_list:
            for rep in reps:
                path = data_root / str(taxa) / str(rep) / f"GTR_{seq_len}_MSA.phy"
                if path.exists():
                    inputs.append(path)
                else:
                    print(f"Missing: {path}")
    return inputs


def parse_elapsed(s: str) -> float:
    # formats: M:SS, H:MM:SS, or S.SS
    if ":" not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0
    parts = s.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def phylip_to_fasta(phylip_path: Path, fasta_path: Path):
    lines = phylip_path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty file: {phylip_path}")
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid PHYLIP header: {lines[0]}")
    n_seq = int(header[0])
    seq_len = int(header[1])
    idx = 1
    seqs = []

    while len(seqs) < n_seq and idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        parts = line.split()
        name = parts[0]
        seq = "".join(parts[1:])
        idx += 1
        while len(seq) < seq_len and idx < len(lines):
            seq += "".join(lines[idx].split())
            idx += 1
        seqs.append((name, seq))

    if len(seqs) != n_seq:
        raise ValueError(f"Expected {n_seq} sequences, got {len(seqs)} from {phylip_path}")

    with open(fasta_path, "w") as f:
        for name, seq in seqs:
            f.write(f">{name}\n")
            # wrap at 80 for readability
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def run_cmd(cmd, stdout_path=None, extra_env=None):
    time_bin = "/usr/bin/time"
    if not Path(time_bin).exists():
        time_bin = None
    if time_bin:
        full_cmd = [time_bin, "-v"] + cmd
    else:
        full_cmd = cmd

    stdout_fh = None
    if stdout_path:
        stdout_fh = open(stdout_path, "wb")

    start = time.time()
    try:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            full_cmd,
            stdout=stdout_fh or subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
        )
        end = time.time()
    finally:
        if stdout_fh:
            stdout_fh.close()

    wall_s = end - start
    max_rss_kb = None
    stderr_text = proc.stderr.decode("utf-8", errors="ignore")

    if time_bin:
        m = re.search(
            r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)",
            stderr_text,
        )
        if m:
            wall_s = parse_elapsed(m.group(1).strip())
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text)
        if m:
            max_rss_kb = int(m.group(1))

    return proc.returncode, wall_s, max_rss_kb, stderr_text


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def main():
    # === Edit only these lists for your benchmarks ===
    species_list = [24, 48, 96, 192]
    length_list = [10000000]
    reps = [0]

    data_root = Path("/mnt/c/Users/descfly/Desktop/publish_code/data")
    out_dir = Path("/mnt/c/Users/descfly/Desktop/publish_code/output/benchmark")
    fasttree_input_dir = out_dir / "fasttree_inputs"
    aster_bin = "/mnt/c/Users/descfly/Desktop/publish_code/software/ASTER/bin/caster-site"
    fasttree_bin = "/root/miniconda3/bin/FastTreeMP"
    if not Path(fasttree_bin).exists():
        fasttree_bin = "/mnt/c/Users/descfly/Desktop/publish_code/software/FastTree"
    threads = 32
    repeat = 1
    logs_dir = out_dir / "logs"
    aster_out_dir = out_dir / "aster"
    fasttree_out_dir = out_dir / "fasttree"
    ensure_dirs(logs_dir, aster_out_dir, fasttree_out_dir, fasttree_input_dir)

    inputs = find_inputs(data_root, species_list, length_list, reps)
    if not inputs:
        print(f"No inputs found under {data_root} for requested species/lengths")
        return 1

    csv_path = out_dir / "benchmark_results.csv"
    csv_exists = csv_path.exists()

    rows = []

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(
                [
                    "tool",
                    "taxa",
                    "rep",
                    "seq_len",
                    "input",
                    "output",
                    "threads",
                    "repeat",
                    "wall_time_s",
                    "max_rss_kb",
                    "exit_code",
                ]
            )

        for input_path in inputs:
            taxa = input_path.parent.parent.name
            rep = input_path.parent.name
            m = re.search(r"GTR_(\d+)_MSA\.phy", input_path.name)
            seq_len = m.group(1) if m else "unknown"
            base_name = f"taxa{taxa}_rep{rep}_len{seq_len}"

            for r in range(1, repeat + 1):
                # ASTER (caster-site)
                aster_out = aster_out_dir / f"{base_name}_r{r}.nwk"
                aster_log = logs_dir / f"aster_{base_name}_r{r}.log"
                aster_cmd = [
                    aster_bin,
                    "-i",
                    str(input_path),
                    "-o",
                    str(aster_out),
                    "-t",
                    str(threads),
                ]
                rc, wall_s, max_rss_kb, stderr_text = run_cmd(aster_cmd)
                with open(aster_log, "w") as lf:
                    lf.write("COMMAND: " + shlex.join(aster_cmd) + "\n")
                    lf.write(stderr_text)
                print(
                    f"[ASTER] taxa={taxa} len={seq_len} rep={rep} r={r} "
                    f"wall_time_s={wall_s:.3f} exit={rc}"
                )
                writer.writerow(
                    [
                        "ASTER(caster-site)",
                        taxa,
                        rep,
                        seq_len,
                        str(input_path),
                        str(aster_out),
                        threads,
                        r,
                        f"{wall_s:.3f}",
                        "" if max_rss_kb is None else str(max_rss_kb),
                        rc,
                    ]
                )
                rows.append(
                    [
                        "ASTER(caster-site)",
                        taxa,
                        rep,
                        seq_len,
                        str(input_path),
                        str(aster_out),
                        threads,
                        r,
                        wall_s,
                        max_rss_kb,
                        rc,
                    ]
                )

                # FastTree (FASTA input; convert from PHYLIP if needed)
                fasttree_out = fasttree_out_dir / f"{base_name}_r{r}.nwk"
                fasttree_log = logs_dir / f"fasttree_{base_name}_r{r}.log"
                fasttree_in = fasttree_input_dir / f"{base_name}.fasta"
                if not fasttree_in.exists():
                    phylip_to_fasta(input_path, fasttree_in)
                fasttree_cmd = [
                    fasttree_bin,
                    "-nt",
                    "-gtr",
                    str(fasttree_in),
                ]
                fasttree_env = {"OMP_NUM_THREADS": str(threads)}
                if "FastTreeMP" in os.path.basename(fasttree_bin):
                    fasttree_env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
                rc, wall_s, max_rss_kb, stderr_text = run_cmd(
                    fasttree_cmd, stdout_path=fasttree_out, extra_env=fasttree_env
                )
                with open(fasttree_log, "w") as lf:
                    lf.write("COMMAND: " + shlex.join(fasttree_cmd) + "\n")
                    lf.write(stderr_text)
                print(
                    f"[FastTree] taxa={taxa} len={seq_len} rep={rep} r={r} "
                    f"wall_time_s={wall_s:.3f} exit={rc}"
                )
                writer.writerow(
                    [
                        "FastTree",
                        taxa,
                        rep,
                        seq_len,
                        str(input_path),
                        str(fasttree_out),
                        threads if "FastTreeMP" in os.path.basename(fasttree_bin) else 1,
                        r,
                        f"{wall_s:.3f}",
                        "" if max_rss_kb is None else str(max_rss_kb),
                        rc,
                    ]
                )
                rows.append(
                    [
                        "FastTree",
                        taxa,
                        rep,
                        seq_len,
                        str(input_path),
                        str(fasttree_out),
                        threads if "FastTreeMP" in os.path.basename(fasttree_bin) else 1,
                        r,
                        wall_s,
                        max_rss_kb,
                        rc,
                    ]
                )

    # Summary by tool + taxa + seq_len
    summary_path = out_dir / "benchmark_summary.csv"
    summary = {}
    for tool, taxa, _rep, seq_len, _inp, _out, _thr, _r, wall_s, _rss, rc in rows:
        key = (tool, taxa, seq_len)
        if rc != 0:
            continue
        summary.setdefault(key, []).append(float(wall_s))

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tool",
                "taxa",
                "seq_len",
                "n_runs",
                "mean_s",
                "median_s",
                "min_s",
                "max_s",
            ]
        )
        for (tool, taxa, seq_len), times in sorted(summary.items()):
            writer.writerow(
                [
                    tool,
                    taxa,
                    seq_len,
                    len(times),
                    f"{mean(times):.3f}",
                    f"{median(times):.3f}",
                    f"{min(times):.3f}",
                    f"{max(times):.3f}",
                ]
            )

    print(f"Done. Results: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
