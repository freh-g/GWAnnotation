#!/usr/bin/env python3
import argparse, sys, os, time, tempfile, glob, shutil
import pysam
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default = '../lifted_vcfs/')
    p.add_argument("--tabix", default = '/home/francesco.gualdi/scratch/Projects/AI_PRS_CADD/Efficient_querying/data/whole_genome_SNVs_inclAnno.tsv.gz')
    p.add_argument("--output-dir", default = '../annotated_vcfs/')
    p.add_argument("--tabix-header", default=None)
    p.add_argument("--chrom-col", type=int, default=0)
    p.add_argument("--pos-col",   type=int, default=1)
    p.add_argument("--ref-col",   type=int, default=2)
    p.add_argument("--alt-col",   type=int, default=3)
    p.add_argument("--threads",   type=int, default=cpu_count())
    return p.parse_args()


def get_tabix_header(tabix_path, tabix_header_path=None):
    if tabix_header_path:
        with open(tabix_header_path) as f:
            return f.readline().rstrip("\n").lstrip("#").split("\t")
    tb = pysam.TabixFile(tabix_path)
    if tb.header:
        line = list(tb.header)[-1]
        tb.close()
        return line.lstrip("#").split("\t")
    tb.close()
    plain = tabix_path[:-3] if tabix_path.endswith(".gz") else tabix_path
    with open(plain) as f:
        return f.readline().rstrip("\n").lstrip("#").split("\t")


def process_chunk(args):
    chunk_file, tabix_path, tabix_header_path, cols, out_file = args
    cc, pc, rc, ac = cols

    tb = pysam.TabixFile(tabix_path)
    tabix_cols = get_tabix_header(tabix_path, tabix_header_path)
    extra_tabix_cols = tabix_cols[4:]

    matched = 0
    total   = 0
    rows    = []

    with open(chunk_file) as fin:
        header = fin.readline().rstrip("\n").split("\t")
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            fields = line.split("\t")
            chrom = fields[cc]
            pos   = int(fields[pc])
            ref   = fields[rc]
            alt   = fields[ac]
            try:
                hits = tb.fetch(chrom, pos - 1, pos)
            except ValueError:
                continue
            for hit in hits:
                hf = hit.split("\t")
                if (hf[cc] == chrom and int(hf[pc]) == pos
                        and hf[rc] == ref and hf[ac] == alt):
                    rows.append("\t".join(fields + hf[4:]))
                    matched += 1
                    break

    tb.close()

    with open(out_file, "w") as fout:
        fout.write("\t".join(header + extra_tabix_cols) + "\n")
        fout.write("\n".join(rows))
        if rows:
            fout.write("\n")

    return matched, total, chrom


def annotate_file(in_file, out_file, args):
    """Annotate a single input file, writing to out_file."""
    cc, pc, rc, ac = args.chrom_col, args.pos_col, args.ref_col, args.alt_col

    tmpdir = tempfile.mkdtemp()
    chunks = defaultdict(list)

    with open(in_file) as fin:
        header = fin.readline()
        for line in fin:
            chrom = line.split("\t")[cc]
            chunks[chrom].append(line)

    chroms  = sorted(chunks.keys())
    nchroms = len(chroms)
    ncpus   = min(args.threads, nchroms)

    chunk_files = []
    for chrom in chroms:
        path     = os.path.join(tmpdir, f"chunk_{chrom}.tsv")
        out_path = os.path.join(tmpdir, f"chunk_{chrom}_merged.tsv")
        with open(path, "w") as f:
            f.write(header)
            f.writelines(chunks[chrom])
        chunk_files.append((path, args.tabix, args.tabix_header,
                            (cc, pc, rc, ac), out_path))

    del chunks

    total_matched = 0
    total_rows    = 0

    with Pool(processes=ncpus) as pool:
        for matched, total, chrom in tqdm(
            pool.imap_unordered(process_chunk, chunk_files),
            total=len(chroms), desc=f"  {os.path.basename(in_file)}", unit="chr"
        ):
            total_matched += matched
            total_rows    += total

    # Concatenate per-chromosome results
    first = True
    with open(out_file, "w") as fout:
        for chrom in chroms:
            out_path = os.path.join(tmpdir, f"chunk_{chrom}_merged.tsv")
            if not os.path.exists(out_path):
                continue
            with open(out_path) as fin:
                hdr = fin.readline()
                if first:
                    fout.write(hdr)
                    first = False
                for line in fin:
                    fout.write(line)

    shutil.rmtree(tmpdir)
    return total_matched, total_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*")))
    if not input_files:
        print("❌ No files found in input directory", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_files)} file(s) to annotate", file=sys.stderr)

    grand_matched = 0
    grand_total   = 0
    wall_start    = time.time()

    for in_file in input_files:
        base     = os.path.basename(in_file)
        out_file = os.path.join(args.output_dir, base + "_annotated.tsv")
        print(f"\n▶ {base}", file=sys.stderr)
        t0 = time.time()

        matched, total = annotate_file(in_file, out_file, args)

        elapsed = int(time.time() - t0)
        print(f"  {matched}/{total} matched in {elapsed}s → {out_file}", file=sys.stderr)
        grand_matched += matched
        grand_total   += total

    elapsed = int(time.time() - wall_start)
    h, r = divmod(elapsed, 3600)
    m, s = divmod(r, 60)
    print(f"\n✅ All done in {h}h {m}m {s}s. "
          f"{grand_matched}/{grand_total} total matched.", file=sys.stderr)


if __name__ == "__main__":
    main()