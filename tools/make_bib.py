"""Build manuscript/refs.bib from Crossref metadata (every DOI checked: HTTP 200 + title printed for review).
Entries without a DOI (software, arXiv) are written by hand below. No unpublished manuscript is cited.
Log: manuscript/refs_check.json (doi, http status, container, title, date checked)."""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOIS = {
    "clopper1934": "10.1093/biomet/26.4.404",
    "hoeffding1956": "10.1214/aoms/1177728178",
    "greenberg2014": "10.1016/j.spl.2013.12.009",
    "koppschneider2020": "10.1002/bimj.201800395",
    "hanley1983": "10.1001/jama.1983.03330370053031",
    "meeker2017": "10.1002/9781118594841",
    "kalra2016": "10.1016/j.tra.2016.09.010",
    "zhao2017": "10.1109/TITS.2016.2582208",
    "angelopoulos2023ppi": "10.1126/science.adi6000",
    "bates2021rcps": "10.1145/3478535",
    "shimodaira2000": "10.1016/S0378-3758(00)00115-4",
    "wang2025clcp": "10.1109/TNNLS.2024.3356512",
    "wang2025pose": "10.1109/TNNLS.2025.3598481",
    "carlevaro2025": "10.1109/TNNLS.2025.3568174",
    "cai2023": "10.1109/TNNLS.2021.3128514",
    "fedeli2023": "10.1109/TNNLS.2023.3265524",
    "tobin2017": "10.1109/IROS.2017.8202133",
    "kadian2020": "10.1109/LRA.2020.3013848",
    "cdnet2014": "10.1109/CVPRW.2014.126",
    "lasiesta2016": "10.1016/j.cviu.2016.08.005",
    "bmc2012": "10.1007/978-3-642-37410-4_25",
    "redmon2016": "10.1109/CVPR.2016.91",
    "baze1979": "10.1080/00224065.1979.11980894",
    "riedmaier2020": "10.1109/ACCESS.2020.2993730",
    "koopman2017": "10.1109/MITS.2016.2583491",
    "koopman2016": "10.4271/2016-01-0128",
    "littlewood1993": "10.1145/163359.163373",
    "zhao2019cbi": "10.1109/ISSRE.2019.00012",
    "viele2014": "10.1002/pst.1589",
    "corso2021": "10.1613/jair.1.12716",
    "brown2001": "10.1214/ss/1009213286",
    "angelopoulos2025ltt": "10.1214/24-AOAS1998",
    "angelopoulos2026ppipp": "10.1214/26-AOAS2215",
    "wilson2021assurance": "10.1080/00401706.2020.1867646",
    "yoon2025bayesian": "10.1007/s12206-025-2409-1",
    "zheng2023gamma": "10.1016/j.ress.2023.109617",
    "zheng2023acceptance": "10.1016/j.ress.2022.108877",
    "wang2024monoprop": "10.1016/j.ress.2024.110173",
    "ding2024truncated": "10.1016/j.ress.2023.109782",
    "tan2025weibull": "10.1016/j.ress.2025.111074",
    "zhao2020operational": "10.1016/j.infsof.2020.106393",
    "zhao2018perfection": "10.1016/j.ress.2018.03.032",
    "littlewood2020replacement": "10.1016/j.ress.2019.106752",
    "bishop2022bootstrapping": "10.1109/ISSRE55969.2022.00020",
    "wang2026fusion": "10.1016/j.ress.2025.111844",
    "tao2024subsea": "10.1016/j.ress.2024.110153",
    "ye2023structuraldt": "10.1016/j.ress.2023.109543",
    "dwight2026dtmaint": "10.1016/j.ress.2025.111496",
    "riedmaier2021vvuq": "10.1007/s11831-020-09473-7",
    "saad2025dtperception": "10.1109/IV64158.2025.11097489",
    "li2026tugs": "10.1016/j.ress.2025.111975",
    "reway2020simgap": "10.1109/IV47402.2020.9304567",
    "dieter2023drone": "10.3390/electronics12102197",
    "amini2024translators": "10.1145/3691620.3695067",
    "zhao2025statfoundation": "10.1109/ITSC60802.2025.11423546",
    "barbier2019smc": "10.1109/IVS.2019.8813793",
    "fisch2024stratppi": "10.52202/079017-3541",
    "angelopoulos2023gentle": "10.1561/2200000101",
    "degrancey2022conformaldet": "10.1007/978-3-031-14862-0_23",
    "yang2023purse": "10.1109/CVPR52729.2023.00864",
    "mei2025pwc": "10.1177/02783649251378151",
    "yuan2026conformal": "10.1016/j.ress.2026.112417",
    "wen2025nversion": "10.1016/j.ress.2025.111016",
    "aghazadeh2026hipllm": "10.1016/j.ress.2026.112615",
    "paterson2025amlas": "10.1016/j.ress.2025.111311",
    "khastgir2021stpa": "10.1016/j.ress.2021.107610",
}
MANUAL = r"""
@misc{luo2025sim2val,
  author = {Rachel Luo and Heng Yang and Michael Watson and Apoorva Sharma and Sushant Veer and Edward Schmerling and Marco Pavone},
  title = {{Sim2Val}: Leveraging Correlation Across Test Platforms for Variance-Reduced Metric Estimation},
  year = {2025},
  howpublished = {Conference on Robot Learning (CoRL) 2025; preprint arXiv:2506.20553},
  note = {doi: \href{https://doi.org/10.48550/arXiv.2506.20553}{10.48550/arXiv.2506.20553}}
}
@misc{zhu2026scape,
  author = {Dijie Zhu and Seunghun Oh and Ruopeng Huang and Zhiyu Huang and Jiaqi Ma and Chen Tang},
  title = {{SCAPE}: Scenario-Conditioned Simulation-Augmented Policy Evaluation},
  year = {2026},
  howpublished = {Preprint arXiv:2608.19425 (not peer reviewed)},
  note = {doi: \href{https://doi.org/10.48550/arXiv.2608.19425}{10.48550/arXiv.2608.19425}}
}
@misc{cabon2020vkitti2,
  author = {Yohann Cabon and Naila Murray and Martin Humenberger},
  title = {Virtual {KITTI} 2},
  year = {2020},
  howpublished = {arXiv:2001.10773},
  note = {doi: \href{https://doi.org/10.48550/arXiv.2001.10773}{10.48550/arXiv.2001.10773}}
}
@misc{mpfb2,
  title = {{MPFB} 2: {MakeHuman} plugin for {Blender}, version 2.0.17},
  howpublished = {\url{https://extensions.blender.org/add-ons/mpfb/}},
  note = {Bundled assets CC0; accessed 2026-09-27}
}
@misc{ultralytics,
  author = {{Ultralytics}},
  title = {Ultralytics {YOLO} (yolo26s-seg)},
  howpublished = {\url{https://github.com/ultralytics/ultralytics}},
  note = {Accessed 2026-09-25}
}
"""


PROTECT = {"Bayesian": "{Bayesian}", "bayesian": "{Bayesian}", "Weibull": "{Weibull}", "weibull": "{Weibull}", "Type-I": "{Type-I}", "Wiener": "{Wiener}", "Gamma": "{Gamma}", "Monte Carlo": "{Monte Carlo}", "Cdnet": "{CDnet}", "CDnet": "{CDnet}", "Lasiesta": "{LASIESTA}", "LASIESTA": "{LASIESTA}", "IWDA": "{IWDA}",
           "Sim2Real": "{Sim2Real}", "JAMA": "{JAMA}"}


def protect(title):
    for k, v in PROTECT.items():
        title = title.replace(k, v)
    return title


def tex_escape(s):
    import html
    s = html.unescape(s).replace("Ⅰ", "I").replace("Ⅱ", "II")
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def fetch(doi):
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}", headers={"User-Agent": "p17-bib/1.0 (mailto:none)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.load(r)["message"]


def entry(key, doi, m):
    typ = m.get("type")
    def nm(x):
        return x.title() if x.isupper() else x
    authors = " and ".join(f"{nm(a.get('family', ''))}, {nm(a.get('given', ''))}".strip(", ") for a in m.get("author", [])) or None
    title = protect(tex_escape(m["title"][0]))
    cont = tex_escape(m.get("container-title", [""])[0]) if m.get("container-title") else ""
    yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    f = {"title": "{" + title + "}", "year": str(yr), "doi": doi,
         "note": "{doi: \\href{https://doi.org/" + doi + "}{" + doi.replace("_", "\\_") + "}}"}  # IEEEtran.bst does not print doi (C5)
    if authors:
        f["author"] = authors
    if m.get("volume"):
        f["volume"] = m["volume"]
    if m.get("issue"):
        f["number"] = m["issue"]
    if m.get("page"):
        f["pages"] = m["page"].replace("-", "--")
    if typ == "journal-article":
        bt, f["journal"] = "article", cont
    elif typ in ("proceedings-article", "book-chapter"):
        bt, f["booktitle"] = "inproceedings", cont
    elif typ in ("book", "monograph"):
        bt = "book"
        if m.get("publisher"):
            f["publisher"] = tex_escape(m["publisher"])
    else:
        bt, f["journal"] = "article", cont
    body = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}" for k, v in f.items())
    return f"@{bt}{{{key},\n{body}\n}}\n"


def main():
    out, log = [], []
    for key, doi in DOIS.items():
        try:
            st, m = fetch(doi)
            out.append(entry(key, doi, m))
            log.append(dict(key=key, doi=doi, http=st, type=m.get("type"), container=(m.get("container-title") or [""])[0],
                            title=m["title"][0], checked=str(date.today())))
            print("OK ", key, "|", (m.get("container-title") or [""])[0][:40], "|", m["title"][0][:70])
        except Exception as e:  # noqa: BLE001
            log.append(dict(key=key, doi=doi, http=str(e), checked=str(date.today())))
            print("ERR", key, doi, e)
        time.sleep(0.5)
    (ROOT / "manuscript" / "refs.bib").write_text("% generated by tools/make_bib.py (Crossref)\n" + "\n".join(out) + MANUAL,
                                                 encoding="utf-8")
    import re as _re
    full = "% generated by tools/make_bib.py (Crossref)\n" + "\n".join(out) + MANUAL
    ress = _re.sub(r"  note = \{doi: \\href\{https://doi.org/([^}]*)\}\{[^}]*\}\},?\n", lambda m: "  doi = {" + m.group(1) + "},\n", full)
    ress = _re.sub(r"(  doi = \{[^}]*\}),\n(  doi = \{[^}]*\},\n)", r"\1,\n", ress)   # keep one doi field
    (ROOT / "manuscript" / "refs_ress.bib").write_text(ress.replace("(Crossref)", "(Crossref) -- RESS variant: doi field only, no note"), encoding="utf-8")
    (ROOT / "manuscript" / "refs_check.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
