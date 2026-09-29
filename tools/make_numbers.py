"""Write manuscript/numbers.tex — one macro per number used in the manuscript and the supplement, read from
results/*.json (checklist C6). Each macro carries its source key and numerator / denominator / unit in a comment.
tools/check_numbers.py re-derives every macro from the same json paths and must report 0 mismatches.
Usage: python tools/make_numbers.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "results"
OUT = ROOT / "manuscript" / "numbers.tex"


def J(n):
    return json.load(open(R / n, encoding="utf-8"))


def ucp(x, n, delta=0.05):
    return 1.0 if x >= n else float(sps.beta.ppf(1 - delta, x + 1, n - x))


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def f(x, d=2):
    return f"{x:.{d}f}"


def build():
    M = []   # (name, value, comment)

    def add(name, value, comment):
        assert name.isalpha(), name
        M.append((name, str(value), comment))

    c2 = J("c2_mc.json")
    prim = {r["delta_prime"]: r for r in c2["primary"]}
    add("Nzero", 59, "n0(5%,5%) = ceil(ln .05/ln .95); c2_mc.primary n0")
    assert prim[0.075]["n0"] == 59
    add("NzeroTwenty", 14, "n0(20%,5%); exchange_rate_v2.marked.OP*.n0")
    assert J("exchange_rate_v2.json")["marked"]["OP*"]["n0"] == 14
    for dp, tag in ((0.075, "A"), (0.10, "B"), (0.15, "C"), (0.20, "D")):
        r = prim[dp]
        add(f"cExact{tag}", r["c_integer_exact"], f"c2_mc.primary[delta'={dp}].c_integer_exact (events)")
        add(f"cStar{tag}", r["c_star"], f"c2_mc.primary[delta'={dp}].c_star (continuous lower bound)")
        add(f"cSave{tag}", f"{100 * r['c_integer_exact'] / 59:.0f}\\%", f"c / n0 = {r['c_integer_exact']}/59")
    add("CtwoNgrid", len(c2["grid"]), "c2_mc grid points")
    add("CtwoMaxDevPP", f(c2["max_abs_dev_pp_exact"]), "c2_mc.max_abs_dev_pp_exact (percentage points, 40-point grid, MC 1e5)")
    c3 = J("c3_numbers.json")["rows"]
    add("CthreeA", c3[0]["n_real"], "c3_numbers q_U=0"); add("CthreeB", c3[1]["n_real"], "q_U=1%"); add("CthreeC", c3[2]["n_real"], "q_U=2%")
    c4 = J("c4_power.json")
    add("PopScenesTen", c4["scenes_needed_c4a_pop"]["U<=0.1"], "c4_power.scenes_needed_c4a_pop U<=10% (J=0)")
    add("PopScenesFive", c4["scenes_needed_c4a_pop"]["U<=0.05"], "U<=5% (J=0)")
    add("LemmaMin", f(c4["lemma_mean_median"]["min_F_floor_Np"], 4), "c4_power.lemma_mean_median.min_F_floor_Np")

    man = J("corpus17_manifest.json")["counts"]
    add("CorpusT", f"{man['mode_T_total']:,}", "corpus17_manifest.counts.mode_T_total (events)")
    add("CorpusTvideos", man["mode_T_real_videos"], "mode_T_real_videos")
    add("MainScenes", man["calibration_scenes"]["main"], "calibration_scenes.main")
    add("NightScenes", man["calibration_scenes"]["night"], "calibration_scenes.night")
    add("TurbScenes", man["calibration_scenes"]["turbulence"], "calibration_scenes.turbulence")
    add("PTZScenes", man["calibration_scenes"]["excluded_PTZ"], "calibration_scenes.excluded_PTZ")
    bg = J("p1_background17.json")["by_tier"]
    add("MainCDnet", sum(1 for v in bg["main"] if not v[:2] in ("I_", "O_")), "p1_background17.by_tier.main CDnet")
    add("MainLASIESTA", sum(1 for v in bg["main"] if v[:2] in ("I_", "O_")), "by_tier.main LASIESTA")

    u = J("e0_units.json")["recomputed"]
    mm, nt, at = u["main"]["rule_2of3"], u["night"]["rule_2of3"], u["all_tiers"]["rule_2of3"]
    add("MainN", mm["N_events"], "e0_units main N (events, d>=32)")
    add("MainDw", mm["D_worst_phase"]["num"], "e0_units main D_worst_phase.num (events)")
    add("MainDwPct", pct(mm["D_worst_phase"]["value"], 2), f"{mm['D_worst_phase']['num']}/{mm['N_events']}")
    add("MainDexp", f(mm["D_old_variant_mean"]["num"], 4), "expected discordant events (mean over phases and 3 variants)")
    add("MainDexpPct", pct(mm["D_old_variant_mean"]["value"], 2), f"{mm['D_old_variant_mean']['num']:.4f}/{mm['N_events']}")
    add("MainUfleet", pct(mm["D_worst_phase"]["U_fleet"], 2), "U_CP(3,180)")
    add("MainJ", mm["J"]["num"], "scenes with a discordant event"); add("MainM", mm["J"]["den"], "scenes")
    add("MainUpop", pct(mm["J"]["U_pop"], 2), f"U_CP({mm['J']['num']},{mm['J']['den']})")
    add("NightN", nt["N_events"], "night events d>=32"); add("NightDw", nt["D_worst_phase"]["num"], "night worst-phase discordant")
    add("NightDwPct", pct(nt["D_worst_phase"]["value"], 0), f"{nt['D_worst_phase']['num']}/{nt['N_events']}")
    add("NightDexp", f(nt["D_old_variant_mean"]["num"], 2), "night expected discordant events (mean over phases and 3 variants)")
    add("NightDexpMaj", f(nt["D_expected"]["num"], 2), "night expected discordant events, majority rule")
    add("NightUfleet", pct(nt["D_worst_phase"]["U_fleet"], 1), "night U_fleet worst"); add("NightJ", nt["J"]["num"], "night J")
    add("NightUpop", pct(nt["J"]["U_pop"], 1), "night U_pop")
    tb = u["turbulence"]["rule_2of3"]
    add("TurbN", tb["N_events"], "turbulence events d>=32 (scenes run: 2)")
    add("AllN", at["N_events"], "all tiers N"); add("AllDw", at["D_worst_phase"]["num"], "all tiers worst-phase discordant")
    add("AllUfleet", pct(at["D_worst_phase"]["U_fleet"], 1), "all tiers U_fleet"); add("AllJ", at["J"]["num"], "all tiers J")
    add("AllM", at["J"]["den"], "all tiers scenes"); add("AllUpop", pct(at["J"]["U_pop"], 1), "all tiers U_pop")

    cf = J("e0_confusion.json")
    c0 = cf["by_op"]["OP*"]["main"]
    add("PmainOP", pct(c0["p_hat"], 2), f"{c0['real_miss_events_expected']:.4f}/{c0['N_events']} (phase-expected)")
    add("PmainOPshort", pct(c0["p_hat"], 1), "same, 1 decimal (abstract)")
    add("RealMissExpOP", f(c0["real_miss_events_expected"], 2), "expected real-miss events at OP* (main)")
    add("RealMissEvOP", c0["appearance"]["events_real_miss_any_phase"], "events with a real miss at >= 1 phase")
    add("RealMissScOP", c0["appearance"]["scenes_with_real_miss"], "scenes with a real miss")
    add("RecallOP", pct(c0["twin_recall_on_real_miss"], 0), f"{c0['cells']['real_miss & twin_miss']:.2f}/{c0['real_miss_events_expected']:.2f}")
    cn = cf["by_op"]["OP*"]["night"]
    add("RecallNight", pct(cn["twin_recall_on_real_miss"], 0), f"{cn['cells']['real_miss & twin_miss']:.2f}/{cn['real_miss_events_expected']:.2f}")
    ch = cf["challenged_along_dmin"]["main"]
    add("ChRealMissEv", ch["appearance"]["events_real_miss_any_phase"], "(16,16) real-miss events")
    add("ChTwinAlso", ch["appearance"]["events_twin_also_miss_at_some_missed_phase"], "(16,16) twin also misses")
    add("ChRecall", pct(ch["twin_recall_on_real_miss"], 0), "(16,16) recall expected")
    add("ChN", ch["N_events"], "(16,16) events")
    add("OldPhat", "11.3\\%", "operating_points_v2.final_operating_point.p_hat over 161 CDnet events (all tiers)")
    op = J("operating_points_v2.json")["final_operating_point"]
    assert abs(op["p_hat"] - 0.11335) < 1e-4 and op["n_events"] == 161 and op["n_required"] == 112
    add("OldN", op["n_events"], "events behind the pooled 11.3%"); add("OldNreq", op["n_required"], "direct n at 11.3%")

    va = J("validity_audit.json")
    lo, lc = va["points"]["OP*"]["loso"], va["points"]["challenge_16_16"]["loso"]
    add("LosoOP", lo["violation_rate_D"]["num"], "LOSO violations D_s > U_pop(-s), OP* (of 78)")
    add("LosoCh", lc["violation_rate_D"]["num"], "LOSO violations, (16,16) (of 78)")
    add("LosoChFC", lc["false_certificates"]["num"], "(16,16) LOSO false certificates")
    add("ChUpop", pct(lc["U_pop_all"], 1), f"U_CP({lc['J_all']},78)"); add("ChJ", lc["J_all"], "(16,16) J")
    add("McFC", pct(va["points"]["OP*"]["mc"]["false_cert_rate"], 1), "MC 10,000 false-cert rate OP*")
    add("McFCch", pct(va["points"]["challenge_16_16"]["mc"]["false_cert_rate"], 1), "MC false-cert rate (16,16)")
    mf = va["must_fail"]["d1_K48"]
    add("MustFailScenes", mf["n_scenes_p_gt_eps"], "K=48 scenes with p_s > eps")
    add("MustFailRefuse", pct(mf["mc_restricted_to_p_gt_eps"]["refuse_rate_given_p_gt_eps"], 0), "refusal rate")
    add("MustFailApp", mf["scenes_p_app_gt_eps"], "scenes with detector-miss p > eps")
    fl = [r for d in va["c4a_fleet"].values() for r in d.values() if "exact_fail_poisson_binomial" in r]
    add("FleetMeasPB", f(max(r["exact_fail_poisson_binomial"] for r in fl), 4), f"max exact PB failure over {len(fl)} cases")
    add("FleetMeasCases", len(fl), "cases")
    add("PopTightMax", f(va["pop_theorem_check"]["c4a_pop_max_fail"], 3), "C4a-pop tight-case max failure (MC 1e5)")

    m2 = J("must_fail_v2.json")["points"]
    cu, cc = m2["OP*"]["curve"], m2["challenge_16_16"]["curve"]
    add("LeakUpopZero", pct(cu[0]["U_pop_median"], 1), "k=0 U_pop OP*"); add("LeakUpopSix", pct(cu[6]["U_pop_median"], 1), "k=6 U_pop OP*")
    add("LeakUpopZeroCh", pct(cc[0]["U_pop_median"], 1), "k=0 U_pop (16,16)"); add("LeakUpopSixCh", pct(cc[6]["U_pop_median"], 1), "k=6 (16,16)")
    kw = min(c["k"] for c in cu if c["share_fleet_refused"] >= 1.0)
    add("LeakWithdrawK", kw, "smallest k with all subsets withdrawn (U > eps/2), OP*")
    kh = min(c["k"] for c in cu if c["share_fleet_refused"] > 0)
    add("LeakWithdrawKfirst", kh, "smallest k with some subsets withdrawn, OP*")
    add("NightFC", m2["OP*"]["night_false_cert_main_only"], "night cameras falsely certified, judged alone (OP*)")
    add("NightFCch", m2["challenge_16_16"]["night_false_cert_main_only"], "same at (16,16)")
    add("NightPgt", m2["OP*"]["night_p_gt_eps"], "night cameras with p_s > eps")
    add("MainFCleak", max(c["main_false_cert_max"] for c in cu + cc), "main cameras falsely certified, any k")

    ex = J("exchange_rate_v2.json")["marked"]
    s = ex["OP*"]
    add("SavedOP", f"{s['q_known']['events_saved_per_camera']['median']:.0f}", "median real events saved per camera, OP*, q known")
    add("SavedCamOP", s["q_known"]["cameras_saving"], "cameras saving (of 78)")
    add("SavedFinCamOP", s["q_finite"]["cameras_saving"], "cameras saving with the twin as built")
    add("SavedFinOP", f"{s['q_finite']['events_saved_per_camera']['median']:.0f}", "median saved, twin as built")
    ex1 = J("exchange_rate.json")["marked"]["OP*"]["q_known"]
    add("TwinRunsOP", f"{ex1['n_twin_events_needed_median']:.0f}", "exchange_rate.marked.OP*.q_known.n_twin_events_needed_median")
    add("CpuPerSavedOP", f(s["q_known"]["cpu_hours_per_saved_event"]["median"], 3), "CPU-hours per saved real event, OP* median")
    add("BreakEven", f"{1 / s['q_known']['cpu_hours_per_saved_event']['median']:.0f}", "1/c: events/hour (assumes 1 CPU-hour = 1 hour of waiting)")
    add("SavedCh", f"{ex['challenge_16_16']['q_known']['events_saved_per_camera']['median']:.0f}", "(16,16) saved per camera")
    add("SavedOPC", f"{ex['OP-C']['q_known']['events_saved_per_camera']['median']:.0f}", "OP-C saved per camera")
    tc = J("exchange_rate.json")["twin_cost"]
    add("TwinCostScene", f(tc["cpu_hours_per_scene_median"], 2), "CPU-hours per scene, median (C0 build)")
    add("TwinCostTotal", f(tc["cpu_hours_total"], 1), "CPU-hours, 78 scenes")
    g = [r for r in J("exchange_rate_v2.json")["grid"] if r["eps"] == 0.2]
    add("GridPts", len(g), "grid points eps=20%")
    add("GridFinMax", max(r["q_finite"]["cameras_saving"] for r in g), "max cameras saving with the twin as built, any grid point")

    ce = J("c2_empirical.json")["results"]["eps5_d32_K1"]["pools"]
    tw = ce["S1_twin_main"]
    add("CtwoTwinPass", pct(tw["P_sim_pass"], 1), "twin pass probability (MC 1e5)")
    for dp, tag in (("0.075", "A"), ("0.1", "B"), ("0.15", "C"), ("0.2", "D")):
        add(f"CtwoFC{tag}", pct(tw["by_delta_prime"][dp]["false_cert_at_p_eq_eps"], 1), f"false cert at p=eps, delta'={dp}")
    add("CtwoBMCPass", pct(ce["S0_BMC_Gstar"]["P_sim_pass"], 0), "BMC-synth pass probability")
    add("CtwoBMCN", ce["S0_BMC_Gstar"]["N_sim"], "BMC-synth events d>=32")

    iw = J("iw_ablation_v2.json")["summary_all_targets"]
    add("IWtemporal", f(iw["a_temporal|logistic"]["auc_median"]), "iw_ablation_v2 all targets, logistic, temporal")
    add("IWgeometry", f(iw["b_geometry|logistic"]["auc_median"]), "all targets, geometry")
    add("IWappearance", f(iw["e_appearance_only|logistic"]["auc_median"]), "all targets, appearance only")
    iwc = J("iw_ablation_v2.json")["summary_cdnet_targets"]
    add("IWtemporalCD", f(iwc["a_temporal|logistic"]["auc_median"]), "CDnet targets (supplement)")
    add("IWgeometryCD", f(iwc["b_geometry|logistic"]["auc_median"]), "CDnet targets")
    add("IWappearanceCD", f(iwc["e_appearance_only|logistic"]["auc_median"]), "CDnet targets")

    geo = J("e0_geometry_sensitivity.json")["frame_level"]
    add("ClampShare", pct(geo["clamp_share"], 0), "share of twin frames with a width clamp")
    add("ClampExtraDrop", f(geo["paired_within_event"]["extra_twin_drop_pts"], 1), "extra twin hit-rate drop (points) on clamped frames")
    add("ClampPairs", geo["paired_within_event"]["n_event_variants"], "event x variant pairs")
    c5 = J("c5a_monotone.json")["tables"]["d_min32"]["K16"]["q_by_x"]
    add("CfiveQzero", pct(c5["0.0"], 2), "q at x=0"); add("CfiveQfifty", pct(c5["0.5"], 2), "q at x=50%")
    conf = J("twin_discordance_c0_main.json")["confidence"]
    add("ConfTwin", f(conf["twin_score_median_pooled"]), "median detector confidence on twin hits")
    add("ConfReal", f(conf["real_score_median_pooled"]), "median detector confidence on real hits")
    # sensitivity: original event definition and OP-A / OP-C
    orig = J("twin_discordance_c0_main.json")["by_definition"]["orig"]["operating_point"]
    add("OrigN", orig["N"], "original definition events at OP*"); add("OrigJ", orig["J_possible"], "original definition J")
    for name, key in (("OPA", "OP-A"), ("OPC", "OP-C")):
        c = cf["by_op"][key]["main"]
        add(f"Pmain{name}", pct(c["p_hat"], 2), f"{key} main p-hat")
        add(f"Dexp{name}", f(c["cells"]["real_miss & twin_catch (D)"], 2), f"{key} expected discordant events")
    extra(add)
    return M


def main():
    M = build()
    names = [n for n, _, _ in M]
    assert len(names) == len(set(names)), "duplicate macro"
    lines = ["% generated by tools/make_numbers.py from results/*.json -- DO NOT EDIT BY HAND (checklist C6)"]
    for n, v, c in M:
        lines.append(f"\\newcommand{{\\n{n}}}{{{v}}} % {c}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {OUT} ({len(M)} macros)")



def extra(M_add):
    """Macros for Tables II-III (tier table, exchange-rate table)."""
    u = J("e0_units.json")["recomputed"]["turbulence"]["rule_2of3"]
    M_add("TurbDw", u["D_worst_phase"]["num"], "turbulence worst-phase discordant events")
    M_add("TurbUfleet", pct(u["D_worst_phase"]["U_fleet"], 1), "turbulence U_fleet")
    M_add("TurbJ", u["J"]["num"], "turbulence J"); M_add("TurbM", u["J"]["den"], "turbulence scenes run")
    M_add("TurbUpop", pct(u["J"]["U_pop"], 1), "turbulence U_pop")
    for j, w in ((0, "zero"), (1, "one"), (2, "two"), (3, "three")):
        M_add(f"UpopJ{w}", pct(ucp(j, 78), 1), f"U_CP({j},78), formula (worked example)")
    for j, w in ((0, "zero"), (1, "one"), (2, "two"), (3, "three")):
        M_add(f"UhalfJ{w}", pct(ucp(j, 78, 0.025), 1), f"U_CP({j},78) at delta/2 (decision level)")
    budget = 0.05 - ucp(0, 78, 0.025)
    M_add("FiveBudget", pct(budget, 1), "eps=5%: room left for q_U after U_{delta/2}(0,78)")
    M_add("FiveRuns", math.ceil(math.log(0.025) / math.log(1 - budget)), "eps=5%: miss-free twin runs needed for q_U <= budget")
    c0 = J("e0_confusion.json")["by_op"]["OP*"]["main"]
    M_add("CalibPerCam", f(c0["N_events"] / c0["m_scenes"], 1), "annotated calibration events at OP* per static camera (N/m)")
    M_add("DirectPooledU", pct(ucp(c0["real_miss_events_any_phase"], c0["N_events"]), 1),
          "direct CP on the calibrated cameras' real misses (events missed at some phase / N)")
    tw = J("c2_empirical.json")["results"]["eps5_d32_K1"]["pools"]["S1_twin_main"]["by_delta_prime"]
    for dp, tag in (("0.075", "A"), ("0.1", "B"), ("0.15", "C"), ("0.2", "D")):
        M_add(f"CtwoCF{tag}", pct(tw[dp]["closed_form"], 1), f"closed form P(pass)(1-eps)^n1+(1-P(pass))(1-eps)^n0, delta'={dp}")
    M_add("CtwoSE", pct(0.0010, 1), "MC standard error bound sqrt(0.1*0.9/1e5)=0.09 pts, rounded to 0.1")
    nt = J("twin_tiers.json")["night"]
    M_add("NightHeldD", pct(nt["held_out"]["median_scene_D"], 1), "night held-out median per-scene D (pre-registered criterion <= 5%)")
    M_add("NightHeldN", nt["held_out"]["n_events_OP"], "night held-out events at OP*")
    M_add("NightBaseD", pct(nt["base_tuning_D"], 1), "night tuning scenes, base twin, D-hat (phase-averaged)")
    M_add("NightProfD", pct([s for s in nt["steps_on_tuning"] if s["profile"] == nt["final_profile"]][0]["D_hat_op"], 1),
          "night tuning scenes, kept profile, D-hat")
    sd = J("sens_event_definition.json")["definitions"]
    M_add("DefAllA", sd["a256"]["n_events_all_durations"], "primary definition, events of all durations (static)")
    M_add("DefAllO", sd["orig"]["n_events_all_durations"], "original definition, events of all durations (static)")
    oc = sd["orig"]["points"]["challenge_16_16"]
    M_add("OrigChJ", oc["J"], "original definition (16,16) J"); M_add("OrigChUpop", pct(oc["U_pop"], 1), "original (16,16) U_pop")
    oo = sd["orig"]["points"]["OP*"]
    M_add("OrigOPUfleet", pct(oo["U_fleet"], 2), "original OP* U_fleet (worst phase)"); M_add("OrigOPDw", oo["disc_worst"], "original OP* disc")
    ex = J("exchange_rate_v2.json")["marked"]
    ex1 = J("exchange_rate.json")["marked"]
    for name, key in (("OPS", "OP*"), ("OPA", "OP-A"), ("OPC", "OP-C"), ("CH", "challenge_16_16")):
        s, s1 = ex[key], ex1[key]
        M_add(f"Tp{name}", pct(s["p_main"], 2), f"{key} static-domain p-hat")
        M_add(f"Tnd{name}", s["n_direct"], f"{key} n_direct (80% power at p-hat)")
        M_add(f"Tsv{name}", f"{s['q_known']['events_saved_per_camera']['median']:.0f}", f"{key} saved per camera (median)")
        M_add(f"Tcam{name}", s["q_known"]["cameras_saving"], f"{key} cameras saving (of 78)")
        M_add(f"Tcamfin{name}", s["q_finite"]["cameras_saving"], f"{key} cameras saving, twin as built")
        M_add(f"Truns{name}", f"{s1['q_known']['n_twin_events_needed_median']:.0f}", f"{key} twin runs needed per camera")
        M_add(f"Tcpu{name}", f(s["q_known"]["cpu_hours_per_saved_event"]["median"], 3), f"{key} CPU-h per saved event")


if __name__ == "__main__":
    main()
