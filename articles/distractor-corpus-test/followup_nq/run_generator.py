import argparse, gzip, json, time, os, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-4B"
DATA = "/content/generator_inputs.jsonl.gz"
OUT = os.environ.get('GEN_OUTPUT','/content/generator_outputs_v2.jsonl')

ap=argparse.ArgumentParser()
ap.add_argument('--limit-qids',type=int,default=int(os.environ.get('LIMIT_QIDS','0')))
ap.add_argument('--output',default=OUT)
ap.add_argument('--batch-size',type=int,default=int(os.environ.get('BATCH_SIZE','1')))
args,_unknown=ap.parse_known_args()
print('GPU:',torch.cuda.get_device_name(0),'memory_GB:',round(torch.cuda.get_device_properties(0).total_memory/1e9,2),flush=True)
print('Loading',MODEL_ID,flush=True)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True)
model.eval()
print('Model ready',flush=True)
with gzip.open(DATA,'rt',encoding='utf8') as f: rows=[json.loads(x) for x in f]
if args.limit_qids:
    qids=[]
    for r in rows:
        if r['qid'] not in qids: qids.append(r['qid'])
    selected=set(qids[:args.limit_qids]); rows=[r for r in rows if r['qid'] in selected]
done=set()
if os.path.exists(args.output):
    for line in open(args.output,encoding='utf8'):
        try:
            z=json.loads(line); done.add((z['qid'],z['seed'],z['condition'],z['mode']))
        except Exception: pass
print('records',len(rows),'already_done',len(done),flush=True)

def render_context(ds):
    return '\n\n'.join(f"[{i}] {d.get('title','')}\n{d.get('text','')}" for i,d in enumerate(ds,1))
def make_prompt(question, docs):
    prompt=("Answer the question using the supplied passages. The answer field must be the shortest answer that resolves the question, usually one name, date, number, or noun phrase, and must be at most 8 words. Do not write a full sentence in the answer field. "
            "If the passages do not contain the answer, return an empty answer. Do not use outside knowledge. "
            "Return only JSON: {\"answer\": \"...\", \"evidence_quote\": \"an exact short quote from a passage, or empty\"}.\n\n"
            f"Question: {question}\n\nPassages:\n{render_context(docs)}\n\n/no_think")
    messages=[{'role':'user','content':prompt}]
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)

def ask_batch(jobs):
    prompts=[make_prompt(j['row']['question'],j['row']['contexts'][j['condition']][j['mode']]) for j in jobs]
    enc=tok(prompts,padding=True,truncation=True,max_length=3800,return_tensors='pt').to(model.device)
    ids=enc['input_ids']
    start=time.time()
    with torch.inference_mode():
        out=model.generate(**enc,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
    elapsed=time.time()-start
    results=[]
    for i in range(len(jobs)):
        raw=tok.decode(out[i,ids.shape[1]:],skip_special_tokens=True).strip()
        parse_method='json'
        try:
            txt=raw[raw.index('{'):raw.rindex('}')+1]
            parsed=json.loads(txt)
        except Exception:
            parse_method='answer_regex_fallback'
            ma=re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"',raw)
            mq=re.search(r'"evidence_quote"\s*:\s*"((?:\\.|[^"\\])*)"',raw)
            parsed={'answer':ma.group(1) if ma else None,'evidence_quote':mq.group(1) if mq else None}
        results.append((raw,parsed,elapsed/len(jobs),int(enc['attention_mask'][i].sum().item()),parse_method,elapsed))
    return results

os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
jobs=[]
for row in rows:
    for condition in ('ordinary','hard'):
        for mode in ('end_to_end','gold_forced'):
            key=(row['qid'],row['seed'],condition,mode)
            if key not in done: jobs.append({'row':row,'condition':condition,'mode':mode,'key':key})
print('pending jobs',len(jobs),'batch_size',args.batch_size,flush=True)
count=0; t0=time.time()
with open(args.output,'a',encoding='utf8') as out:
    for start_idx in range(0,len(jobs),args.batch_size):
        batch=jobs[start_idx:start_idx+args.batch_size]
        outputs=ask_batch(batch)
        for job,(raw,parsed,elapsed,ninput,parse_method,batch_elapsed) in zip(batch,outputs):
            row=job['row']; condition=job['condition']; mode=job['mode']
            rec={'qid':row['qid'],'seed':row['seed'],'condition':condition,'mode':mode,'question':row['question'],'answer_refs':row['answers'],'answer_source':row['answer_source'],'gold_ids':row['gold_ids'],'context_docids':[d['docid'] for d in row['contexts'][condition][mode]],'context_texts':[d.get('text','') for d in row['contexts'][condition][mode]],'gold_passage_text':row['gold_passage']['text'],'raw_output':raw,'answer':parsed.get('answer'),'evidence_quote':parsed.get('evidence_quote'),'parse_ok':bool(parsed.get('answer')),'parse_method':parse_method,'input_tokens':ninput,'generation_seconds':elapsed,'batch_seconds':batch_elapsed,'batch_size':len(batch),'model_id':MODEL_ID}
            out.write(json.dumps(rec,ensure_ascii=False)+'\n'); out.flush(); count+=1
        if count%10 < args.batch_size: print('generated',count,'seconds',round(time.time()-t0,1),'last',batch[-1]['key'],'batch_elapsed',round(batch_elapsed,2),'input_tokens',ninput,flush=True)
print('DONE generated',count,'seconds',round(time.time()-t0,1),'output',args.output,flush=True)
