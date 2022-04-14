#!/usr/bin/python

"""
Runs coffea processor on the LPC via dask.

Author(s): Raghav Kansal
"""

import pickle
import os
import json
import argparse
import warnings

import numpy as np
import uproot

from coffea import nanoevents
from coffea import processor

from distributed.diagnostics.plugin import WorkerPlugin


def fxn():
    warnings.warn("userwarning", UserWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def add_bool_arg(parser, name, help, default=False, no_name=None):
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


# for running on condor
nanoevents.NanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
    "FatJetAK15_nConstituents",
    "JetPFCandsAK15",
)
nanoevents.NanoAODSchema.mixins["FatJetAK15"] = "FatJet"
nanoevents.NanoAODSchema.mixins["FatJetAK15SubJet"] = "FatJet"
nanoevents.NanoAODSchema.mixins["SubJet"] = "FatJet"
nanoevents.NanoAODSchema.mixins["PFCands"] = "PFCand"


# for Dask executor
class NanoeventsSchemaPlugin(WorkerPlugin):
    def __init__(self):
        pass

    def setup(self, worker):
        from coffea import nanoevents

        nanoevents.NanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
            "FatJetAK15_nConstituents",
            "JetPFCandsAK15",
        )
        nanoevents.NanoAODSchema.mixins["FatJetAK15"] = "FatJet"
        nanoevents.NanoAODSchema.mixins["FatJetAK15SubJet"] = "FatJet"
        nanoevents.NanoAODSchema.mixins["SubJet"] = "FatJet"
        nanoevents.NanoAODSchema.mixins["PFCands"] = "PFCand"


def get_fileset(year, samples, subsamples, starti, endi):
    with open(f"data/pfnanoindex_{year}.json", "r") as f:
        full_fileset = json.load(f)

    fileset = {}

    for sample in samples:
        sample_set = full_fileset[year][sample]
        set_subsamples = list(sample_set.keys())

        # check if any subsamples for this sample have been specified
        get_subsamples = set(set_subsamples).intersection(subsamples)

        # if so keep only that subset
        if len(get_subsamples):
            sample_set = {subsample: sample_set[subsample] for subsample in get_subsamples}

        sample_set = {
            f"{year}_{subsample}": [
                "root://cmsxrootd.fnal.gov//" + fname
                for fname in sample_set[subsample][starti:endi]
            ]
            for subsample in sample_set
        }

        fileset = {**fileset, **sample_set}

    return fileset


def main(args):
    # define processor
    from cms_jetnet.processors import DatasetProcessor

    p = DatasetProcessor(radius=args.label.split("_")[0])

    fileset = (
        get_fileset(args.year, args.samples, args.subsamples, args.starti, args.endi)
        if not len(args.files)
        else {f"{args.year}_files": args.files}
    )

    if args.test:
        fileset = {key: val[:10] for key, val in fileset.items()}

    print(fileset)

    import time
    from distributed import Client
    from lpcjobqueue import LPCCondorCluster
    import dask.dataframe as dd

    tic = time.time()
    cluster = LPCCondorCluster(
        ship_env=True,
        transfer_input_files="src/cms_jetnet",
    )
    client = Client(cluster)
    nanoevents_plugin = NanoeventsSchemaPlugin()
    client.register_worker_plugin(nanoevents_plugin)
    cluster.adapt(minimum=1, maximum=30)

    print("Waiting for at least one worker")
    client.wait_for_workers(1)

    # does treereduction help?
    executor = processor.DaskExecutor(client=client, use_dataframes=True)
    run = processor.Runner(
        executor=executor,
        # savemetrics=True,
        schema=nanoevents.NanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=args.maxchunks,
    )

    out = run(fileset, "Events", processor_instance=p)

    os.system(f"mkdir -p {args.tag}")
    dd.to_parquet(df=out, path=f"{args.tag}/")

    elapsed = time.time() - tic
    # print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")


if __name__ == "__main__":
    # e.g.
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="tag", type=str)
    parser.add_argument("--year", default="2017", help="year", type=str)
    parser.add_argument("--starti", default=0, help="start index of files", type=int)
    parser.add_argument("--endi", default=-1, help="end index of files", type=int)
    parser.add_argument(
        "--executor",
        type=str,
        default="iterative",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument("--samples", default=[], help="samples", nargs="*")
    parser.add_argument("--subsamples", default=[], help="subsamples", nargs="*")
    parser.add_argument(
        "--files", default=[], help="set of files to run on instead of samples", nargs="*"
    )
    parser.add_argument("--chunksize", type=int, default=10000, help="chunk size in processor")
    parser.add_argument("--label", default="AK8_QCD", help="label", type=str)
    parser.add_argument("--njets", default=2, help="njets", type=int)
    parser.add_argument("--maxchunks", default=0, help="max chunks", type=int)
    add_bool_arg(parser, "test", "use smaller fileset for testing")

    args = parser.parse_args()

    main(args)
