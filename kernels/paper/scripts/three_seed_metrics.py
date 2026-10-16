from report_generic_heldout_correction_v2 import METHODS
METRICS=['Abs','1px','D1']

def tables(p,summary,ds,seeds):
    lookup={(r['model'],r['method']):r for r in summary};end=r' \\'
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Held-out adaptation on the same 172 pairs across '+str(len(seeds))+r' balanced assignments. Values are mean $\pm$ sample SD of six-corruption macro metrics; SD measures assignment variation. Source / Clean is not selectable; oracles use GT.}',r'\label{tab:generic-heldout}',r'\scriptsize',r'\setlength{\tabcolsep}{1.2pt}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{ll c cc cc cc}',r'\toprule',r'\multirow{2}{*}{Model} & \multirow{2}{*}{Metric $\downarrow$} & Reference & \multicolumn{2}{c}{Baselines} & \multicolumn{2}{c}{Adaptation} & \multicolumn{2}{c}{Oracles}'+end,r'\cmidrule(lr){3-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',r'& & Source / Clean & Identity & Random & Signal-Global & Signal-Pixel & Global & Pixel'+end,r'\midrule']
    for model in ['raft','acv']:
        label=r'\shortstack[l]{RAFT-Stereo\\\citep{lipson2021raft}}' if model=='raft' else r'\shortstack[l]{ACVNet\\\citep{xu2022attention}}'
        for j,metric in enumerate(METRICS):
            left=r'\multirow{3}{*}{'+label+'}' if j==0 else ''
            ml={'Abs':r'Abs (px) $\downarrow$','1px':r'1px (\%) $\downarrow$','D1':r'D1 (\%) $\downarrow$'}[metric]
            cells=[f"${lookup[model,m][metric]:.2f}\\pm{lookup[model,m][metric+'_seed_SD']:.2f}$" for m in METHODS]
            lines.append(' & '.join([left,ml]+cells)+end)
        if model=='raft':lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}}',r'\end{table*}'])
    (p/'tables/generic_heldout.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for model,name,cite in [('raft','RAFT-Stereo','lipson2021raft'),('acv','ACVNet','xu2022attention')]:
        lines += [r'\begin{table}[ht]',r'\centering\small',r'\caption{'+name+r'~\citep{'+cite+r'} D1 gains and headroom across '+str(len(seeds))+r' assignments. SD is across assignment seeds; paired sequence-cluster 95\% CIs condition on these assignments. Gains/headroom use pp; recovery is a fraction.}',r'\begin{tabular}{lrrl}',r'\toprule',r'Quantity & Mean & Seed SD & 95\% CI'+end,r'\midrule']
        for r in ds:
            if r['model']!=model:continue
            dec=3 if r['unit']=='fraction' else 2
            lines.append(f"{r['quantity']} & {r['mean']:.{dec}f} & {r['seed_SD']:.{dec}f} & $[{r['ci_low']:.{dec}f}, {r['ci_high']:.{dec}f}]$"+end)
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (p/'tables/generic_d1_acv.tex').write_text('\n'.join(lines)+'\n')
