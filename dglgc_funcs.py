import cobra
import pandas as pd
from scipy.optimize import root_scalar


def table1(model: cobra.Model, name: "str", moi: str, add_m: str|None = None) -> pd.Series:
    with model:
        metab = model.metabolites.get_by_id(moi)
        model.add_boundary(metab, type="demand")
        if add_m:
            model.add_boundary(model.metabolites.get_by_id(add_m), type="sink")
        model.objective = "EX_photon_e1_e"
        model.reactions.get_by_id(f"DM_{moi}").lower_bound = 1
        if moi == "bm_pro_c":
            model.reactions.get_by_id("DM_bm_pro_c").lower_bound = 1/9.156
        FNOR_var = cobra.flux_analysis.variability.flux_variability_analysis(model, reaction_list=["FNOR"], loopless="cycleFreeFlux")
        min_FNOR = FNOR_var.loc["FNOR", "minimum"]
        max_FNOR = FNOR_var.loc["FNOR", "maximum"]
        solution = model.optimize()

    atp_demand = ((solution.fluxes['ATPSu'] + solution.fluxes['ATPS4rpp_1'])*3)
    nadph_demand = solution.fluxes["FNOR"]

    met_row = pd.Series({
        "Metabolite": name,
        "C atoms": metab.elements.get("C", 0),
        "ATP demand": atp_demand,
        "NADPH demand": nadph_demand,
        "ATP/NADPH": atp_demand/nadph_demand,
        "Photons": -solution.objective_value
    })

    return met_row


def get_values(model: cobra.Model, moi: cobra.Metabolite = None):

    solution_moi = model.optimize()
    fva_moi = cobra.flux_analysis.variability.flux_variability_analysis(
        model, reaction_list=["FNOR"], loopless="cycleFreeFlux"
    )

    NAPDH_moi = fva_moi.loc["FNOR", "minimum"]
    atp_moi = ((solution_moi.fluxes['ATPSu'] + solution_moi.fluxes['ATPS4rpp_1'])*3)

    C_uptake = solution_moi.fluxes["EX_co2_e"]
    N_uptake = solution_moi.fluxes["EX_no3_e"]

    growth_moi = solution_moi.fluxes["EX_biomass_e"]

    if moi:
        C_loss = solution_moi.fluxes["DM_" + moi.id] * moi.elements["C"]
    else:
        C_loss = 0

    return NAPDH_moi, atp_moi, C_uptake, N_uptake, growth_moi, C_loss


def get_ATP_NADPH(model: cobra.Model, ex_rxn: str, AN_target: float):

    def objective(export):
            
        with model:
            model.reactions.get_by_id(ex_rxn).lower_bound = export
            nadph, atp = get_values(model)[0:2]
            AN_ratio = atp/nadph
        return AN_ratio - AN_target
    return objective


def table2(
    strain: cobra.Model,
    name: str,
    moi: str,
    photons: float,
    growth_WT: float,
    target_atp_nadph: float,
    glycogen: float = 0
) -> pd.Series:
    with strain:
        metab = strain.metabolites.get_by_id(moi)
        
        strain.add_boundary(metab, type="demand")
        
        strain.reactions.get_by_id("EX_photon_e1_e").lower_bound = photons

        x_export = root_scalar(get_ATP_NADPH(strain, f"DM_{moi}", target_atp_nadph), x0=0., method="secant").root
        
        strain.reactions.get_by_id(f"DM_{moi}").lower_bound = x_export
        nadph_export, atp_export, C_uptake_export, N_uptake_export, growth_export, C_loss = get_values(strain, metab)
        growth_export = growth_export/((100-glycogen)/100)

        met_row = pd.Series({
            "Metabolite": name,
            "Extracellular flux": x_export,
            "Growth rate (1/h)": growth_export,
            "Growth rate (% of WT)": growth_export/growth_WT*100,
            "C:N": C_uptake_export/N_uptake_export,
            "Carbon loss (mmol C gDW-1 h-1)": C_loss,
        })

        return met_row


def lgs_model(model: cobra.Model, glycogen_percent: float):
    """Strain with glycogen synthesis fixed to `glycogen_percent` % of dry weight."""
    s = model.copy()
    if glycogen_percent > 0:
        s.add_boundary(s.metabolites.bm_glycogen_c, type="demand")
        c = s.problem.Constraint(
            ((100 - glycogen_percent) / glycogen_percent)
            * s.reactions.get_by_id("DM_bm_glycogen_c").flux_expression
            - s.reactions.EX_biomass_e.flux_expression,
            lb=0, ub=0, name="LGS_glycogen_constraint")
        s.add_cons_vars(c)
    return update_glycogen(s, 0)
        

def add_BM10(model):
    try:
        glycogen = model.metabolites.C00182_cyt
        bigg = False
    except AttributeError:
        glycogen = model.metabolites.glycogen_c
        bigg = True
    if not bigg:
        bm_glyc = cobra.Metabolite("B10_cyt", "R")
    else:
        bm_glyc = cobra.Metabolite("bm_glycogen_c", "R")
    bm_glyc.name = 'Glycogen component of biomass_cyt'
    bm_glyc.compartment = "cyt"
    model.add_metabolites([bm_glyc])

    bm10 = cobra.Reaction('BM0010')
    bm10.name = 'Glycogen component of biomass reaction'
    bm10.subsystem = 'biomass'
    bm10.lower_bound = 0
    bm10.upper_bound = 1000

    if not bigg:
        bm10.add_metabolites({
            model.metabolites.get_by_id('C00182_cyt'): -0.210311/.0345,
            model.metabolites.get_by_id('B10_cyt'): 1.0,
        })
    else:
        bm10.add_metabolites({
            model.metabolites.get_by_id('glycogen_c'): -0.210311/.0345,
            model.metabolites.get_by_id('bm_glycogen_c'): 1.0,
        })

    model.add_reactions([bm10])

    # glycogen = model.metabolites.C00182_cyt
    bof = model.reactions.BM0009
    bof.subtract_metabolites({
        glycogen: -0.21031
    })
    bof.add_metabolites({
        bm_glyc: -0.0345
    })

    return model


def update_glycogen(model, percent=3.45):
    try:
        model.reactions.get_by_id("BM0010")
    except KeyError:
        model = add_BM10(model)
    bof = model.reactions.BM0009
    try:
        _ = model.metabolites.C00182_cyt
        bigg = False
    except AttributeError:
        _ = model.metabolites.glycogen_c
        bigg = True
    if not bigg:
        glyc_component = model.metabolites.get_by_id("B10_cyt")
        extra_mets = [
            "C00002_cyt", "C00001_cyt", "B10_cyt"
        ]
    else:
        glyc_component = model.metabolites.get_by_id("bm_glycogen_c")
        extra_mets = [
            "atp_c", "h2o_c", "bm_glycogen_c"
        ]

    dif = (percent/100 + bof.metabolites[glyc_component]) * -1
    bof.subtract_metabolites({
        met: dif*bof.metabolites[met]/sum(
            [
                bof.get_coefficient(met)
                for met in bof.reactants
                if met.id not in extra_mets
            ]
        )
        for met in bof.reactants
        if met not in [
            model.metabolites.get_by_id(metid)
            for metid in extra_mets
        ]
    })
    bof.add_metabolites({
        glyc_component: dif
    })

    return model
