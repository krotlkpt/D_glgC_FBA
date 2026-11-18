"""
Reproducible FBA for the DglgC / ethylene manuscript (Frontiers MS 1989807).

Self-contained: downloads the base reconstruction iRH783 from its public
repository, so no local model path is required.

  pip install cobra pandas scipy matplotlib
  python dglgc_fba_reproducible.py

Outputs: table1.csv, table2.csv, sensitivity.csv, sensitivity.png
"""
import os, urllib.request
import cobra, pandas as pd, numpy as np
from scipy.optimize import root_scalar

MODEL_URL = ("https://raw.githubusercontent.com/krotlkpt/syn6803/master/"
             "Synechocystis_6803_constrained.xml")
MODEL_FILE = "Synechocystis_6803_constrained.xml"

# --- measured constraints -----------------------------------------------
GROWTH_WT   = 0.048681058969584315   # h-1, measured WT at 200 umol photons m-2 s-1
GLYCOGEN_WT = 13.7936                # % of dry weight
ETH_EFE     = 0.04083758588634332    # mmol gDW-1 h-1, measured DglgC+EFE ethylene
LGS_FRACTION = 0.5                   # LGS glycogen flux as fraction of WT (scenario)

# EFE lumped stoichiometry.  The two EFE reactions run in a fixed ratio:
#   n x (2-OG + O2 -> ethylene + 3 CO2)              ethylene-forming
#   1 x (2-OG + Arg + O2 -> succinate + guanidine + P5C + CO2)   Arg-hydroxylating
# EFE_RATIO = n.  n = 2 is the stoichiometry stated in the manuscript.
EFE_RATIO = 2


def load_model():
    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    m = cobra.io.read_sbml_model(MODEL_FILE)
    ethe = cobra.Metabolite("ethe_c", formula="C2H4", name="Ethylene",
                            compartment="cyt", charge=0)
    gua = cobra.Metabolite("M02035_c", formula="CH6N3", name="Guanidine",
                           compartment="cyt", charge=1)
    n = EFE_RATIO
    efe = cobra.Reaction("EFE", name="Ethylene-forming enzyme (lumped)")
    efe.add_metabolites({
        m.metabolites.akg_c:      -(n + 1),
        m.metabolites.arg__L_c:   -1,
        m.metabolites.o2_c:       -(n + 1),
        m.metabolites.h_c:        -(n + 1),
        m.metabolites.co2_c:       (3 * n + 1),
        ethe:                      n,
        m.metabolites.succ_c:      1,
        gua:                       1,
        m.metabolites.get_by_id("1pyr5c_c"): 1,
        m.metabolites.h2o_c:       (n + 1),
    })
    gdmh = cobra.Reaction("GDMH", name="Guanidine hydrolase (sll1077)")
    gdmh.add_metabolites({m.metabolites.h2o_c: -1, gua: -1,
                          m.metabolites.nh4_c: 1, m.metabolites.urea_c: 1})
    m.add_reactions([efe, gdmh])
    assert not efe.check_mass_balance(), efe.check_mass_balance()
    assert not gdmh.check_mass_balance(), gdmh.check_mass_balance()
    return m


def atp_flux(sol):
    return (sol.fluxes["ATPSu"] + sol.fluxes["ATPS4rpp_1"]) * 3


# --- Table 1: biosynthetic costs ----------------------------------------
def table1_row(model, name, moi, add_m=None):
    """ATP from the photon-optimal solution; NADPH (FNR) is NOT uniquely
    determined at that optimum, so its FVA minimum and maximum are both
    reported and the minimum is used for the ratio (same convention as
    Table 2)."""
    with model:
        metab = model.metabolites.get_by_id(moi)
        model.add_boundary(metab, type="demand")
        if add_m:
            model.add_boundary(model.metabolites.get_by_id(add_m), type="sink")
        model.objective = "EX_photon_e1_e"
        model.reactions.get_by_id(f"DM_{moi}").lower_bound = 1
        if moi == "bm_pro_c":
            model.reactions.get_by_id("DM_bm_pro_c").lower_bound = 1 / 9.156
        sol = model.optimize()
        model.reactions.EX_photon_e1_e.bounds = (sol.objective_value,
                                                 sol.objective_value)
        fva = cobra.flux_analysis.flux_variability_analysis(
            model, reaction_list=["FNOR"], loopless="cycleFreeFlux")
    atp = atp_flux(sol)
    lo, hi = fva.loc["FNOR", "minimum"], fva.loc["FNOR", "maximum"]
    return pd.Series({
        "Metabolite": name,
        "C atoms": metab.elements.get("C", 0),
        "ATP demand": atp,
        "NADPH demand (FVA min)": lo,
        "NADPH demand (FVA max)": hi,
        "ATP/NADPH (at FVA min)": atp / lo if lo else np.nan,
        "Photons": -sol.objective_value,
    })


# --- Table 2: export required to restore the WT ATP:NADPH balance -------
def get_values(model, moi=None):
    sol = model.optimize()
    fva = cobra.flux_analysis.flux_variability_analysis(
        model, reaction_list=["FNOR"], loopless="cycleFreeFlux")
    nadph = fva.loc["FNOR", "minimum"]
    atp = atp_flux(sol)
    c_loss = sol.fluxes["DM_" + moi.id] * moi.elements["C"] if moi else 0
    return (nadph, atp, sol.fluxes["EX_co2_e"], sol.fluxes["EX_no3_e"],
            sol.fluxes["EX_biomass_e"], c_loss)


def table2_row(strain, name, moi, photons, growth_wt, target, glycogen=0):
    with strain:
        metab = strain.metabolites.get_by_id(moi)
        strain.add_boundary(metab, type="demand")
        strain.reactions.EX_photon_e1_e.lower_bound = photons

        def f(export):
            with strain:
                strain.reactions.get_by_id(f"DM_{moi}").lower_bound = export
                nadph, atp = get_values(strain)[0:2]
            return atp / nadph - target

        x = root_scalar(f, x0=0.25, method="secant").root
        x = max(x, 0.0)
        strain.reactions.get_by_id(f"DM_{moi}").lower_bound = x
        nadph, atp, c_up, n_up, growth, c_loss = get_values(strain, metab)
        growth = growth / ((100 - glycogen) / 100)
        return pd.Series({
            "Metabolite": name, "Extracellular flux": x,
            "Growth rate (1/h)": growth,
            "Growth rate (% of modelled WT)": growth / growth_wt * 100,
            "C:N": c_up / n_up,
            "Carbon loss (mmol C gDW-1 h-1)": c_loss})


def add_bm10(model):
    """Move glycogen out of the lumped biomass reaction into its own
    component reaction BM0010 (present in recent versions of iRH783)."""
    if "BM0010" in [r.id for r in model.reactions]:
        return model
    bm_glyc = cobra.Metabolite("bm_glycogen_c", "R",
                               name="Glycogen component of biomass", compartment="cyt")
    model.add_metabolites([bm_glyc])
    bm10 = cobra.Reaction("BM0010", name="Glycogen component of biomass reaction",
                          subsystem="biomass", lower_bound=0, upper_bound=1000)
    bm10.add_metabolites({model.metabolites.glycogen_c: -0.210311 / .0345, bm_glyc: 1.0})
    model.add_reactions([bm10])
    bof = model.reactions.BM0009
    bof.subtract_metabolites({model.metabolites.glycogen_c: -0.21031})
    bof.add_metabolites({bm_glyc: -0.0345})
    return model


def set_glycogen(model, percent=0.0):
    """Set the glycogen share of biomass to `percent` (% of dry weight),
    rescaling the remaining biomass components."""
    model = add_bm10(model)
    bof = model.reactions.BM0009
    comp = model.metabolites.bm_glycogen_c
    extra = {"atp_c", "h2o_c", "bm_glycogen_c"}
    dif = (percent / 100 + bof.metabolites[comp]) * -1
    denom = sum(bof.get_coefficient(m) for m in bof.reactants if m.id not in extra)
    bof.subtract_metabolites({m: dif * bof.metabolites[m] / denom
                              for m in bof.reactants if m.id not in extra})
    bof.add_metabolites({comp: dif})
    return model


def lgs_model(glycogen_percent):
    """Strain with glycogen synthesis fixed to `glycogen_percent` % of dry weight."""
    s = load_model()
    if glycogen_percent > 0:
        s.add_boundary(s.metabolites.bm_glycogen_c, type="demand")
        c = s.problem.Constraint(
            ((100 - glycogen_percent) / glycogen_percent)
            * s.reactions.get_by_id("DM_bm_glycogen_c").flux_expression
            - s.reactions.EX_biomass_e.flux_expression,
            lb=0, ub=0, name="LGS_glycogen_constraint")
        s.add_cons_vars(c)
    return set_glycogen(s, 0)


def wt_reference():
    m = load_model()
    with m:
        m.objective = "EX_photon_e1_e"
        m.add_boundary(m.metabolites.bm_glycogen_c, type="demand")
        m.reactions.EX_biomass_e.lower_bound = GROWTH_WT * (100 - GLYCOGEN_WT) / 100
        m.reactions.get_by_id("DM_bm_glycogen_c").lower_bound = GROWTH_WT * GLYCOGEN_WT / 100
        m = set_glycogen(m, 0)
        sol = m.optimize()
        nadph, atp = get_values(m)[0:2]
    return sol.objective_value, atp / nadph


EXPORTS = {"Pyruvate": "pyr_c", "2-Oxoglutarate": "akg_c", "Fumarate": "fum_c",
           "Succinate": "succ_c", "Ribose-5-phosphate": "r5p_c", "Ethylene": "ethe_c"}

TABLE1_METS = {
    "Glycogen": ("glycogen_c", None), "Pyruvate": ("pyr_c", None),
    "Succinate": ("succ_c", None), "2-Oxoglutarate": ("akg_c", None),
    "Fumarate": ("fum_c", None), "Malate": ("mal__L_c", None),
    "Citrate": ("cit_c", None), "Lipid": ("palmACP_c", "ACP_c"),
    "Lipid elongation": ("palmACP_c", "myrsACP_c"),
    "Additional double bond": ("hdeceACP_c", "palmACP_c"),
    "Protein (per AA)": ("bm_pro_c", None), "Ethylene": ("ethe_c", None)}


def main():
    m = load_model()
    t1 = pd.DataFrame([table1_row(m, n, moi, add) for n, (moi, add) in TABLE1_METS.items()])
    t1.to_csv("table1.csv", index=False)
    print("\nTable 1\n", t1.round(3).to_string(index=False))

    photons_wt, target = wt_reference()
    print(f"\nWT photon uptake {photons_wt:.4f}; target ATP/NADPH {target:.4f}")

    strains = []
    d = set_glycogen(load_model(), 0)
    strains.append(("DglgC", d, 0, list(EXPORTS.items())))
    strains.append(("LGS", lgs_model(GLYCOGEN_WT * LGS_FRACTION),
                    GLYCOGEN_WT * LGS_FRACTION, list(EXPORTS.items())))
    e = load_model()
    e.add_boundary(e.metabolites.ethe_c, type="demand")
    e.reactions.get_by_id("DM_ethe_c").lower_bound = ETH_EFE
    strains.append(("DglgC+EFE", set_glycogen(e, 0), 0, list(EXPORTS.items())[:-1]))

    t2 = []
    for label, s, gly, keys in strains:
        for n, moi in keys:
            r = table2_row(s, n, moi, photons_wt, GROWTH_WT, target, gly)
            r["Strain"] = label
            t2.append(r)
    t2 = pd.DataFrame(t2)[["Strain", "Metabolite", "Extracellular flux", "Growth rate (1/h)",
                           "Growth rate (% of modelled WT)", "C:N",
                           "Carbon loss (mmol C gDW-1 h-1)"]]
    t2.to_csv("table2.csv", index=False)
    print("\nTable 2\n", t2.round(4).to_string(index=False))

    # --- sensitivity analysis on the LGS glycogen flux ------------------
    recs = []
    for x in np.arange(0.0, 1.01, 0.1):
        s = lgs_model(GLYCOGEN_WT * x)
        for n, moi in EXPORTS.items():
            r = table2_row(s, n, moi, photons_wt, GROWTH_WT, target, GLYCOGEN_WT * x)
            recs.append(dict(glycogen_pct_of_WT=round(x * 100),
                             metabolite=n, flux=r["Extracellular flux"],
                             growth=r["Growth rate (1/h)"],
                             growth_pct=r["Growth rate (% of modelled WT)"],
                             CN=r["C:N"], C_loss=r["Carbon loss (mmol C gDW-1 h-1)"]))
    sens = pd.DataFrame(recs)
    sens.to_csv("sensitivity.csv", index=False)
    print("\nSensitivity (required export flux)\n",
          sens.pivot(index="glycogen_pct_of_WT", columns="metabolite",
                     values="flux").round(4).to_string())

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        for n in EXPORTS:
            d = sens[sens.metabolite == n].sort_values("glycogen_pct_of_WT")
            ax[0].plot(d.glycogen_pct_of_WT, d.flux, marker="o", ms=3.5, label=n)
            ax[1].plot(d.glycogen_pct_of_WT, d.growth_pct, marker="o", ms=3.5, label=n)
        for a in ax:
            a.axvline(LGS_FRACTION * 100, ls="--", lw=1, color="0.5")
            a.set_xlabel("Glycogen synthesis flux (% of WT)")
            a.spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("Required export flux (mmol gDW$^{-1}$ h$^{-1}$)")
        ax[1].set_ylabel("Growth rate (% of modelled WT)")
        ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig("sensitivity.png", dpi=400)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
