"""
Skimmer for ParticleNet tagger inputs.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray

from typing import Dict
import os

from .utils import add_selection_no_cutflow, pad_val

P4 = {
    "eta": "eta",
    "phi": "phi",
    "mass": "mass",
    "pt": "pt",
}


"""
Features
--------

Graph-level:
 - gen-jet p4
 - gen parton flavour

Node-level:
 - pfcand p4
 - pfcand pid
 - pfcand charge?

"""


class DatasetProcessor(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, radius="AK8", num_jets=1, num_particles=150):
        self.radius = radius

        if self.radius == "AK8":
            self.fatjet_label = "FatJet"
            self.subjet_label = "SubJet"
            self.pfcands_label = "FatJetPFCands"
            self.svs_label = "FatJetSVs"

        elif self.radius == "AK15":
            self.fatjet_label = "FatJetAK15"
            self.subjet_label = "FatJetAK15SubJet"
            self.pfcands_label = "FatJetAK15PFCands"
            self.svs_label = "JetSVsAK15"

        self.skim_vars = {
            "RecoJet": {
                **P4,
                "msoftdrop": "msd",
            },
            "GenJet": {**P4},
            "RecoPart": {
                **P4,
                "pdgId": "pid",
                "charge": "charge",
            },
        }

        self._columns = [
            f"{vartype}_{var}" for vartype in self.skim_vars for var in self.skim_vars[vartype]
        ]
        print(self._columns)

        self.num_jets = num_jets
        self.num_particles = num_particles

        # self._accumulator = dict_accumulator({})

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def dump_table(self, pddf: pd.DataFrame, fname: str) -> None:
        """
        Saves pandas dataframe events to './outparquet'
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        local_dir = os.path.abspath(os.path.join(".", "outparquet"))
        os.system(f"mkdir -p {local_dir}")

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, f"{local_dir}/{fname}")

    def to_pandas(self, events: Dict[str, np.array]) -> pd.DataFrame:
        """
        Convert our dictionary of numpy arrays into a pandas data frame.
        Uses multi-index columns for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        return pd.concat(
            [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in events.items()],
            axis=1,
            keys=list(events.keys()),
        )

    def get_genjet_vars(
        self,
        events: NanoEventsArray,
        fatjets: FatJetArray,
        ak15: bool = True,
        match_dR: float = 1.0,
    ):
        """Matched fat jet to gen-level jet and gets gen jet vars"""
        if ak15:
            gen_dr = fatjets.delta_r(events.GenJetAK15)
            matched_gen_jet = events.GenJetAK15[ak.argmin(gen_dr, axis=1, keepdims=True)]
            matched_gen_jet_mask = fatjets.delta_r(matched_gen_jet) < match_dR

            GenJetVars = {
                f"GenJet_{key}": matched_gen_jet.mass * ak.values_astype(matched_gen_jet_mask, int)
                + ak.values_astype(~matched_gen_jet_mask, int) * (-99999)
                for (var, key) in self.skim_vars["GenJet"].items()
            }
        else:
            # NanoAOD automatically matched ak8 fat jets
            # No soft dropped gen jets however

            GenJetVars = {
                f"GenJet_{key}": ak.fill_none(fatjets.matched_gen[var], -99999)
                for (var, key) in self.skim_vars["GenJet"].items()
            }

        return GenJetVars

    def get_recoparticles_vars(self, events: NanoEventsArray, fatjets: FatJetArray, jet_idx: int):
        jet_ak_pfcands = events[self.pfcands_label][events[self.pfcands_label].jetIdx == jet_idx]
        jet_pfcands = events.PFCands[jet_ak_pfcands.pFCandsIdx]

        feature_dict = {
            f"RecoPart_{key}": jet_pfcands[var] for (var, key) in self.skim_vars["RecoPart"].items()
        }

        feature_dict["RecoPart_mask"] = (
            ~(
                ak.pad_none(
                    # padding to have at least one pf candidate in the graph
                    pad_val(
                        feature_dict["RecoPart_eta"], 1, -1, axis=1, to_numpy=False, clip=False
                    ),
                    self.num_particles,
                    axis=1,
                    clip=True,
                )
                .to_numpy()
                .mask
            )
        ).astype(np.float32)

        # if no padding is needed, mask will = 1.0
        if isinstance(feature_dict["RecoPart_mask"], np.float32):
            feature_dict["RecoPart_mask"] = np.ones(
                (len(feature_dict["RecoPart_eta"]), self.num_particles)
            ).astype(np.float32)

        # convert to numpy arrays and normalize features
        for var in feature_dict:
            a = (
                ak.pad_none(feature_dict[var], self.num_particles, axis=1, clip=True)
                .to_numpy()
                .filled(fill_value=0)
            ).astype(np.float32)

            feature_dict[var] = a

        return feature_dict

    def process(self, events: ak.Array):

        jet_vars = []

        for jet_idx in range(self.num_jets):
            # objects
            fatjets = ak.pad_none(events[self.fatjet_label], self.num_jets, axis=1)[:, jet_idx]

            # selection
            selection = PackedSelection()
            preselection_cut = fatjets.pt > 200
            add_selection_no_cutflow("preselection", preselection_cut, selection)

            # variables
            RecoJetVars = {
                f"RecoJet_{key}": ak.fill_none(fatjets[var], -99999)
                for (var, key) in self.skim_vars["RecoJet"].items()
            }

            JetVars = {
                **RecoJetVars,
                **self.get_genjet_vars(events, fatjets, ak15=self.radius == "AK15"),
            }

            PartVars = self.get_recoparticles_vars(events, fatjets, jet_idx)

            if np.sum(selection.all(*selection.names)) == 0:
                print("No jets pass selections")
                continue

            skimmed_vars = {**JetVars, **PartVars}
            # apply selections
            skimmed_vars = {
                key: np.squeeze(np.array(value[selection.all(*selection.names)]))
                for (key, value) in skimmed_vars.items()
            }

            jet_vars.append(skimmed_vars)

        if len(jet_vars) > 1:
            # stack each set of jets
            jet_vars = {
                var: np.concatenate([jet_var[var] for jet_var in jet_vars], axis=0)
                for var in jet_vars[0]
            }
        elif len(jet_vars) == 1:
            jet_vars = jet_vars[0]
        else:
            print("No jets passed selection")
            return {}

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")

        # convert output to pandas
        df = self.to_pandas(jet_vars)

        # save to parquet
        self.dump_table(df, fname + ".parquet")

        return {}

    def postprocess(self, accumulator):
        return accumulator
