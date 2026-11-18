# third_party

Populate the tau2-bench checkout on a machine with network access:

    git clone https://github.com/sierra-research/tau2-bench
    cd tau2-bench && git rev-parse HEAD   # record this hash in data/raw/tau2/manifest.json

Then copy or symlink its data/tau2/domains/retail directory to
third_party/tau2-bench/data/tau2/domains/retail and run:

    python3.11 -m sdt.data.tau2_curate \
        --tau2-dir third_party/tau2-bench/data/tau2/domains/retail \
        --out-dir data/curated/tau2_retail

The cloned content is gitignored here; only the manifest and curation outputs
we generate ourselves are tracked.
