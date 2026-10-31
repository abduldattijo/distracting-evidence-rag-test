import json, re, string, random, csv, collections
from pathlib import Path
from html import escape

BASE=Path(__file__).resolve().parent
INPUT=BASE/'generator_full.jsonl'
OUT=BASE
BOOTSTRAPS=10000
BOOT_SEED=20261029

def normalize(text):
    text=(text or '').lower()
    text=''.join(ch for ch in text if ch not in string.punctuation)
    text=re.sub(r'\b(a|an|the)\b',' ',text)
    return ' '.join(text.split())

def score(pred, refs):
    pred_tokens=normalize(pred).split()
    best_em=0.0; best_f1=0.0
    for ref in refs or []:
        ref_tokens=normalize(ref).split()
        em=float(pred_tokens==ref_tokens)
        overlap=sum((collections.Counter(pred_tokens)&collections.Counter(ref_tokens)).values())
        f1=(2*overlap/(len(pred_tokens)+len(ref_tokens))) if pred_tokens and ref_tokens and overlap else float(not pred_tokens and not ref_tokens)
        best_em=max(best_em,em); best_f1=max(best_f1,f1)
    return best_em,best_f1

rows=[json.loads(line) for line in INPUT.open(encoding='utf8')]
keys=[(r['qid'],r['seed'],r['condition'],r['mode']) for r in rows]
assert len(rows)==1200 and len(set(keys))==1200, (len(rows),len(set(keys)))
eligible=sorted({r['qid'] for r in rows if r.get('answer_refs')})
assert len(eligible)==70, len(eligible)
lookup={(r['qid'],r['seed'],r['condition'],r['mode']):r for r in rows}
metrics={}
for r in rows:
    r['em'],r['token_f1']=score(r.get('answer'),r.get('answer_refs'))
    r['qrel_in_context']=bool(set(r['gold_ids']) & set(r['context_docids']))
    quote=normalize(r.get('evidence_quote'))
    r['quote_verbatim_supported']=bool(quote) and any(quote in normalize(t) for t in r.get('context_texts',[]))

def estimate(mode,condition,metric):
    selected=[r for r in rows if r['mode']==mode and r['condition']==condition and r.get('answer_refs')]
    perq={q:sum(r[metric] for r in selected if r['qid']==q)/3 for q in eligible}
    return sum(perq.values())/len(perq),perq

def paired_delta(mode,metric):
    o=estimate(mode,'ordinary',metric)[1]
    h=estimate(mode,'hard',metric)[1]
    diffs=[h[q]-o[q] for q in eligible]
    point=sum(diffs)/len(diffs)
    rng=random.Random(BOOT_SEED)
    boots=[]
    for _ in range(BOOTSTRAPS):
        boots.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs))
    boots.sort()
    return point,boots[int(.025*BOOTSTRAPS)],boots[int(.975*BOOTSTRAPS)-1]

summary={
 'study':'post-hoc generator add-on to BEIR NQ same-page distractor retrieval arm',
 'model':'Qwen/Qwen3-4B', 'decoding':'greedy, do_sample=false',
 'hardware':'Colab Pro T4 then A100 high-RAM; sequential then batch-8; same checkpoint, prompt, tokenizer, and generation settings',
 'rows':len(rows),'unique_question_draw_condition_mode_keys':len(set(keys)),
 'questions_total':100,'answer_key_questions':len(eligible),'answer_key_source':'DPR Natural Questions test references, exact question match after case, Unicode, punctuation, and whitespace normalization',
 'draw_seeds':[11,23,37],'contexts':['end_to_end top 10 retrieved passages','gold_forced: one answer-bearing qrel passage placed first plus top 9 non-qrel retrieved passages'],
 'answer_metric':'SQuAD-style exact match and token F1 against maximum over DPR answer aliases; average 3 draws per question; paired bootstrap resamples question IDs, 10,000 replicates, seed 20261029',
 'parse_ok':sum(bool(r.get('parse_ok')) for r in rows),'parse_fallback':sum(r.get('parse_method')=='answer_regex_fallback' for r in rows),
 'ordinary_hard_retrieval_qrel_in_top10_all_100_questions':{},
 'quote_exact_substring_support':{},'scores':{},'paired_effects':{},'bootstrap':{'replicates':BOOTSTRAPS,'seed':BOOT_SEED}
}
for c in ('ordinary','hard'):
 z=[r for r in rows if r['mode']=='end_to_end' and r['condition']==c]
 summary['ordinary_hard_retrieval_qrel_in_top10_all_100_questions'][c]=sum(r['qrel_in_context'] for r in z)/len(z)
for mode in ('end_to_end','gold_forced'):
 summary['scores'][mode]={}
 for c in ('ordinary','hard'):
  summary['scores'][mode][c]={}
  for metric in ('em','token_f1'):
   mean,_=estimate(mode,c,metric)
   summary['scores'][mode][c][metric]=mean
  z=[r for r in rows if r['mode']==mode and r['condition']==c]
  quotes=[r for r in z if r.get('evidence_quote')]
  summary['quote_exact_substring_support'][f'{mode}_{c}']={'quotes':len(quotes),'supported':sum(r['quote_verbatim_supported'] for r in quotes),'fraction':sum(r['quote_verbatim_supported'] for r in quotes)/len(quotes) if quotes else None}
 summary['paired_effects'][mode]={}
 for metric in ('em','token_f1'):
  point,lo,hi=paired_delta(mode,metric)
  summary['paired_effects'][mode][metric]={'hard_minus_ordinary':point,'ci95_low':lo,'ci95_high':hi}

with (OUT/'generator_question_scores.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=['qid','question','mode','condition','em_mean_3_draws','token_f1_mean_3_draws','qrel_top10_rate','answer_refs'])
 w.writeheader()
 for mode in ('end_to_end','gold_forced'):
  for c in ('ordinary','hard'):
   for q in eligible:
    z=[r for r in rows if r['qid']==q and r['mode']==mode and r['condition']==c]
    w.writerow({'qid':q,'question':z[0]['question'],'mode':mode,'condition':c,'em_mean_3_draws':sum(r['em'] for r in z)/3,'token_f1_mean_3_draws':sum(r['token_f1'] for r in z)/3,'qrel_top10_rate':sum(r['qrel_in_context'] for r in z)/3,'answer_refs':' | '.join(z[0]['answer_refs'] or [])})

(OUT/'generator_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

# Compact vector figure; x-axis is paired hard-minus-ordinary percentage points.
W,H=980,370; left,right=390,920; xmin,xmax=-25,5
def xpos(v): return left+(v-xmin)/(xmax-xmin)*(right-left)
items=[
 ('Retrieved top 10 — exact match','end_to_end','em'),
 ('Retrieved top 10 — token F1','end_to_end','token_f1'),
 ('Answer passage first — exact match','gold_forced','em'),
 ('Answer passage first — token F1','gold_forced','token_f1')]
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     '<rect width="100%" height="100%" fill="white"/>',
     '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:13px}.label{font-size:15px}.title{font-size:19px;font-weight:bold}.grid{stroke:#ddd;stroke-width:1}.zero{stroke:#555;stroke-width:1.5;stroke-dasharray:5 4}.ci{stroke:#356b8c;stroke-width:4}.dot{fill:#b24a3a}</style>',
     '<text x="40" y="30" class="title">Generator effect of hard retrieval contexts</text>',
     '<text x="40" y="52" class="small">Hard minus ordinary; 70 questions; paired question-bootstrap 95% intervals</text>']
for tick in [-25,-20,-15,-10,-5,0,5]:
    x=xpos(tick); svg.append(f'<line x1="{x:.1f}" y1="76" x2="{x:.1f}" y2="310" class="{"zero" if tick==0 else "grid"}"/>')
    svg.append(f'<text x="{x:.1f}" y="332" class="small" text-anchor="middle">{tick}</text>')
for i,(label,mode,metric) in enumerate(items):
    y=105+i*58; e=summary['paired_effects'][mode][metric]
    lo=e['ci95_low']*100; hi=e['ci95_high']*100; point=e['hard_minus_ordinary']*100
    svg.append(f'<text x="40" y="{y+5}" class="label">{escape(label)}</text>')
    svg.append(f'<line x1="{xpos(lo):.1f}" y1="{y}" x2="{xpos(hi):.1f}" y2="{y}" class="ci"/>')
    svg.append(f'<circle cx="{xpos(point):.1f}" cy="{y}" r="6" class="dot"/>')
svg.append('<text x="655" y="356" class="small" text-anchor="middle">Percentage points</text></svg>')
(OUT/'figures'/'generator_paired_effect.svg').write_text('\n'.join(svg),encoding='utf8')
print(json.dumps(summary,indent=2))
