"""T_main (paper main comparison table) from results/baselines_{quality,seedtts}.json, quality_rescore.json (X4, ours)
and existing_seedtts.json (ours on Seed-TTS-eval).  Missing cells -> \\todo{}.  Pure stdlib + json; any env.

    python -m steer.paper.baselines_ext_table      (also: python -m steer.paper.baselines_ext_eval table)
"""
import json
from pathlib import Path

PAPER = Path("/home/karan/Desktop/whissle/marketing_post_agent/paper/steertts")
RES = PAPER / "results"

# (key, label, (acoustic M, vocoder M or None = end-to-end), controls); controls: P prompt, K numeric knobs,
# T style tags, E inline events, C zero-shot clone.  Counts from the loaded checkpoints (baselines_ext.SYSTEMS docs).
CLOSEST = [("fastpitch", "FastPitch~\\citep{lancucki2021fastpitch}", (45.8, 13.9), "K"),
           ("matcha", "Matcha-TTS~\\citep{mehta2024matcha}", (18.2, 13.9), "K"),
           ("piper", "Piper (VITS)~\\citep{kim2021vits,piper}", (15.8, None), "K"),
           ("kokoro", "Kokoro-82M~\\citep{kokoro82m}", (82.0, None), "K"),
           ("paradee", "Paradee-8M~\\citep{mahendrakar2026paradee}", (8.1, None), "K"),
           ("kitten", "Kitten TTS nano~\\citep{kittentts}", (15.0, None), "K"),
           ("styletts2", "StyleTTS\\,2~\\citep{li2023styletts2}", (135.8, None), "C"),
           ("parastyle", "ParaStyleTTS~\\citep{lou2025parastyletts}", (52.5, None), "P,K")]
ADVANCED = [("parler_mini", "Parler-TTS mini~\\citep{lyth2024parler}", (880.0, None), "P"),
            ("f5", "F5-TTS~\\citep{chen2024f5tts}", (337.1, 13.5), "K,C"),
            ("cosyvoice2", "CosyVoice\\,2~\\citep{du2024cosyvoice2}", (618.4, 20.8), "P,K,C"),
            ("xtts", "XTTS-v2~\\citep{casanova2024xtts}", (441.0, 25.9), "K,C")]
OURS = [("ours_base", "\\sys{} (L1)", (9.6, 13.5), "P,K,T,E,C", "spk_enc_v3@305k (L1)"),
        ("ours_flow", "\\sys{} (+flow)", (27.2, 13.5), "P,K,T,E,C", "spk_enc_v3@305k + flow_v3@100k")]
NOTE = {"parastyle": "\\textsuperscript{a}", "ours_base": "\\textsuperscript{c}", "ours_flow": "\\textsuperscript{c}"}


def params(key, pv):
    a, v = pv
    n = NOTE.get(key, "")
    if v is None:
        return f"{a:g}{n}"
    return f"{a:g}+{v:g}={a + v:.1f}{n}"


TODO = "\\todo{--}"


def load(name):
    f = RES / f"{name}.json"
    if not f.exists():
        return []
    d = json.load(open(f))
    return d["rows"] if isinstance(d, dict) else d


def pick(rows, metric, system, split=None, condition=None):
    for r in rows:
        if r.get("metric") == metric and r.get("system") == system and (split is None or r.get("split") == split) \
                and (condition is None or r.get("condition") == condition) and r.get("value") is not None:
            return r
    return None


def fmt(r, nd=2, scale=1.0):
    return TODO if r is None else f"{r['value'] * scale:.{nd}f}"


def main():
    q, s, x4, seed_old = load("baselines_quality"), load("baselines_seedtts"), load("quality_rescore"), load("existing_seedtts")
    L, CL = "libritts_r_test_clean_bench500", "libritts_r_test_clean_clone499"
    S = "seed_tts_eval_test_en_1088"

    def base_row(key, label, params, ctrl, rows_q, rows_s, sim_ok):
        wer = pick(rows_q, "wer", key, L)
        ut = pick(rows_q, "utmos", key, L)
        dn = pick(q, "dnsmos_ovrl", key, L)
        sim = pick(rows_q, "sim_wavlm_large", key, CL) if sim_ok else None
        sw = pick(rows_s, "wer", key, S) if sim_ok else None
        ss = pick(rows_s, "sim_wavlm_large", key, S) if sim_ok else None
        c = lambda r: fmt(r) if sim_ok else "--"
        return f"{label} & {params} & {ctrl} & {fmt(wer)} & {fmt(ut)} & {fmt(dn)} & {c(sim)} & {c(sw)} & {c(ss)} \\\\"

    lines = []
    lines.append("\\midrule\n\\multicolumn{9}{l}{\\textit{Closest methods (small, explicit-duration / non-autoregressive)}} \\\\")
    for k, lab, p, c in CLOSEST:
        lines.append(base_row(k, lab, params(k, p), c, q, s, "C" in c))
    lines.append("\\midrule\n\\multicolumn{9}{l}{\\textit{Widely used advanced systems}} \\\\")
    for k, lab, p, c in ADVANCED:
        lines.append(base_row(k, lab, params(k, p), c, q, s, "C" in c))
    lines.append("\\midrule\n\\multicolumn{9}{l}{\\textit{Ours}} \\\\")
    for k, lab, p, c, seedname in OURS:
        wer, ut = pick(x4, "wer", k, L), pick(x4, "utmos", k, L)
        if wer is not None and wer["value"] < 1:  # x4 stores WER as a fraction
            wer = dict(wer, value=wer["value"] * 100)
        dn = pick(q, "dnsmos_ovrl", k, L)
        sim = pick(x4, "sim_wavlm_large", "ours_flow_pflow" if k == "ours_flow" else k, CL)
        sw, ss = pick(seed_old, "WER", seedname), pick(seed_old, "SIM", seedname)
        lines.append(f"{lab} & {params(k, p)} & {c} & {fmt(wer)} & {fmt(ut)} & {fmt(dn)} & {fmt(sim)} & "
                     f"{fmt(sw)}\\textsuperscript{{b}} & {fmt(ss)}\\textsuperscript{{b}} \\\\")
    gt_w, gt_u = pick(x4, "wer", "ground_truth", L), pick(x4, "utmos", "ground_truth", L)
    if gt_w is not None and gt_w["value"] < 1:
        gt_w = dict(gt_w, value=gt_w["value"] * 100)
    gt_s = pick(x4, "sim_wavlm_large", "ground_truth", CL)
    lines.append("\\midrule\n\\multicolumn{9}{l}{\\textit{References}} \\\\")
    cw, cu, cd = pick(q, "wer", "vocos_copysyn", L), pick(q, "utmos", "vocos_copysyn", L), pick(q, "dnsmos_ovrl", "vocos_copysyn", L)
    lines.append(f"Vocos copy-synthesis~\\citep{{siuzdak2024vocos}} & 13.5 & -- & {fmt(cw)} & {fmt(cu)} & {fmt(cd)} & -- & -- & -- \\\\")
    lines.append(f"Ground truth & -- & -- & {fmt(gt_w)} & {fmt(gt_u)} & {fmt(pick(q, 'dnsmos_ovrl', 'ground_truth', L))} & "
                 f"{fmt(gt_s)} & -- & -- \\\\")
    body = "\n".join(lines)
    tex = f"""% generated by steer/paper/baselines_ext_table.py from results/baselines_*.json, quality_rescore.json and
% existing_seedtts.json -- do not edit by hand; \\todo{{--}} = result not produced yet
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lllcccccc}}
\\toprule
 & & & \\multicolumn{{3}}{{c}}{{LibriTTS-R test-clean (500)}} & Clone (499) & \\multicolumn{{2}}{{c}}{{Seed-TTS-eval en (1,088)}} \\\\
\\cmidrule(lr){{4-6}}\\cmidrule(lr){{7-7}}\\cmidrule(lr){{8-9}}
System & Params (M) & Control & WER$\\downarrow$ & UTMOS$\\uparrow$ & DNSMOS$\\uparrow$ & SIM$\\uparrow$ & WER$\\downarrow$ & SIM$\\uparrow$ \\\\
{body}
\\bottomrule
\\end{{tabular}}
\\par\\vspace{{2pt}}
\\begin{{minipage}}{{\\linewidth}}\\footnotesize
All systems run locally from released checkpoints and scored with one scorer set: WER (\\%) Whisper-large-v3 with the
Whisper English normaliser; UTMOS22-strong; DNSMOS P.835 OVRL~\\citep{{reddy2022dnsmosp835}}; SIM = seed-tts-eval
WavLM-large speaker verification against the reference clip. Seed-TTS-eval uses its official scripts. Params (M): acoustic
model + vocoder = total; a single number is an end-to-end model. Control: P free-text prompt, K numeric knobs (rate, and pitch for FastPitch), T style tags, E inline
events, C zero-shot cloning. Default-voice systems use a gender-matched built-in voice. Cloning systems are conditioned on
a different real clip of the same test speaker, and \\sys{{}} on its training-voice pool. 95\\% bootstrap CIs are in
\\texttt{{results/baselines\\_quality.json}}. \\textsuperscript{{a}}Plus a frozen 109M sentence encoder for the prompt.
\\textsuperscript{{c}}Plus frozen external encoders: MiniLM (22.7M) for prompts and WavLM-base-plus-sv (101M) for speaker
vectors. The copy-synthesis row (ground-truth mel through Vocos) is the vocoder upper bound for \\sys{{}}.
\\textsuperscript{{b}}305k base checkpoint, before the text-normaliser fix.
\\end{{minipage}}
\\caption{{Main comparison with the closest small TTS methods and widely used larger systems.}}
\\label{{tab:main}}
\\end{{table*}}
"""
    (PAPER / "tables" / "T_main.tex").write_text(tex)
    print("wrote", PAPER / "tables" / "T_main.tex", "todo cells:", tex.count("\\todo{--}"))


if __name__ == "__main__":
    main()
