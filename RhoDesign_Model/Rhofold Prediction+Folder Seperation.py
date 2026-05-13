from pathlib import Path
import subprocess
import shutil
import os
import re


# settings

input_dir = Path("C:/Users/n764d743/RhoFold/Families")   # folder containing family fasta files
rhofold_dir = Path("C:/Users/n764d743/RhoFold")
ckpt = Path("C:/Users/n764d743/RhoFold/rhofold/pretrained/RhoFold_pretrained.pt")
out_root = Path("C:/Users/n764d743/RhoFold/rhofold_runs")
tmp_dir = Path("C:/Users/n764d743/RhoFold/rhofold_tmp_fastas")
device = "cuda:0"   # change to "cuda:0" or "cpu"
max_seconds_per_sequence = 120  # timeout in seconds

out_root.mkdir(exist_ok=True)
tmp_dir.mkdir(exist_ok=True)

assert rhofold_dir.exists(), f"Missing RhoFold dir: {rhofold_dir}"
assert ckpt.exists(), f"Missing checkpoint: {ckpt}"


# helpers

def safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:150] if len(name) > 150 else name


def parse_fasta(fasta_path: Path):
    records = []
    name = None
    seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(seq_lines).upper().replace("T", "U")
                    records.append((name, seq))
                name = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

    if name is not None:
        seq = "".join(seq_lines).upper().replace("T", "U")
        records.append((name, seq))

    return records


def sequence_already_done(unrelaxed_dir, ssct_dir, npz_dir, seq_name):
    unrelaxed_file = unrelaxed_dir / f"{seq_name}_unrelaxed_model.pdb"
    ssct_file = ssct_dir / f"{seq_name}.ss.ct"
    npz_file = npz_dir / f"{seq_name}.results.npz"

    return (
        unrelaxed_file.exists() and
        ssct_file.exists() and
        npz_file.exists()
    )


# finding all family fastas

fasta_files = sorted(input_dir.glob("*.fasta"))

if not fasta_files:
    print(f"No .fasta files found in {input_dir}")
else:
    print(f"Found {len(fasta_files)} fasta file(s):")
    for fp in fasta_files:
        print(" -", fp.name)


# main loop over families

for multi_fasta in fasta_files:
    family_name = safe_name(multi_fasta.stem)
    family_root = out_root / family_name

    relaxed_dir = family_root / "relaxed"
    unrelaxed_dir = family_root / "unrelaxed"
    logs_dir = family_root / "logs"
    ssct_dir = family_root / "ss_ct"
    npz_dir = family_root / "npz"

    relaxed_dir.mkdir(parents=True, exist_ok=True)
    unrelaxed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ssct_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Processing family fasta: {multi_fasta.name}")
    print(f"Family output folder: {family_root}")
    print("=" * 80)

    records = parse_fasta(multi_fasta)
    print(f"Found {len(records)} sequences in {multi_fasta.name}")

    if not records:
        print(f"Skipping {multi_fasta.name}: no valid FASTA records found")
        continue

    for idx, (name, seq) in enumerate(records, start=1):
        seq_name = safe_name(name) if name else f"seq_{idx}"

        # resume check after any stop/crsh
        if sequence_already_done(unrelaxed_dir, ssct_dir, npz_dir, seq_name):
            print("\n" + "-" * 80)
            print(f"Skipping already completed sequence {idx}/{len(records)}")
            print(f"Family: {family_name}")
            print(f"Sequence name: {seq_name}")
            print("-" * 80)
            continue

        tmp_fasta = tmp_dir / f"{family_name}__{seq_name}.fasta"
        seq_run_dir = family_root / f"work_{seq_name}"
        seq_run_dir.mkdir(exist_ok=True)

        with open(tmp_fasta, "w") as f:
            f.write(f">{seq_name}\n{seq}\n")

        log_path = logs_dir / f"{seq_name}.log"

        cmd = [
            "python", "inference.py",
            "--input_fas", str(tmp_fasta),
            "--output_dir", str(seq_run_dir),
            "--ckpt", str(ckpt),
            "--device", device,
            "--single_seq_pred", "True",
        ]

        print("\n" + "-" * 80)
        print(f"Running sequence {idx}/{len(records)}")
        print(f"Family: {family_name}")
        print(f"Sequence name: {seq_name}")
        print(f"Sequence length: {len(seq)}")
        print(f"Sequence: {seq}")
        print("Command:")
        print(" ".join(cmd))
        print("-" * 80)

        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(f"Family: {family_name}\n")
            log_f.write(f"Input fasta: {multi_fasta}\n")
            log_f.write(f"Sequence name: {seq_name}\n")
            log_f.write(f"Sequence length: {len(seq)}\n")
            log_f.write(f"Sequence: {seq}\n\n")
            log_f.write("Command:\n")
            log_f.write(" ".join(cmd) + "\n\n")
            log_f.write(" STDOUT / STDERR\n\n")

            process = subprocess.Popen(
                cmd,
                cwd=rhofold_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )

            try:
                stdout, _ = process.communicate(timeout=max_seconds_per_sequence)
                print(stdout, end="")
                log_f.write(stdout)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = process.communicate()
                print(stdout, end="")
                log_f.write(stdout)
                log_f.write(
                    f"\nTIMEOUT: sequence exceeded {max_seconds_per_sequence} seconds and was skipped.\n"
                )
                return_code = -1

            log_f.write(f"\n RETURN CODE: {return_code} =====\n")

        print(f"Return code: {return_code}")

        if return_code != 0:
            print(f"Partial/failed sequence: {seq_name}")
            print("Attempting to recover available outputs...")

        # Trying to collect outputs even if failed or timed out
        relaxed_src = seq_run_dir / "relaxed_1000_model.pdb"
        unrelaxed_src = seq_run_dir / "unrelaxed_model.pdb"
        ssct_src = seq_run_dir / "ss.ct"
        npz_src = seq_run_dir / "results.npz"

        if relaxed_src.exists():
            shutil.copy2(relaxed_src, relaxed_dir / f"{seq_name}_relaxed_1000_model.pdb")
        else:
            print(f"Warning: relaxed model missing for {seq_name}")

        if unrelaxed_src.exists():
            shutil.copy2(unrelaxed_src, unrelaxed_dir / f"{seq_name}_unrelaxed_model.pdb")
        else:
            print(f"Warning: unrelaxed model missing for {seq_name}")

        if ssct_src.exists():
            shutil.copy2(ssct_src, ssct_dir / f"{seq_name}.ss.ct")
        else:
            print(f"Warning: ss.ct missing for {seq_name}")

        if npz_src.exists():
            shutil.copy2(npz_src, npz_dir / f"{seq_name}.results.npz")
        else:
            print(f"Warning: results.npz missing for {seq_name}")

        # cleanup area for unnecessaries
        if tmp_fasta.exists():
            os.remove(tmp_fasta)

        shutil.rmtree(seq_run_dir, ignore_errors=True)

    print(f"\nFinished family: {family_name}")

print("\nAll done.")