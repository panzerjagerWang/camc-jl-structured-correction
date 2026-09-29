"""Offline 4B source-adapted correction with the frozen context-gated frontend."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from peft import PeftModel

HERE=Path(__file__).resolve().parent
SYSTEM='You correct Japanese grammar. Replace only the marked error span. Return only the replacement text, with no explanation.'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Corrector:
    def __init__(self,directory=HERE,seed=20260927,device='cuda'):
        directory=Path(directory);self.directory=directory;self.seed=seed;self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('This 4B package requires CUDA; see the separate CPU frontend.')
        dtype=torch.bfloat16;self.dtype=dtype
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.manual_seed(seed)
        manifest=json.loads((directory/'manifest.json').read_text())
        assert seed in manifest['seeds']
        for name,digest in manifest['files_sha256'].items():
            if name.startswith('adapters/') and not name.startswith(f'adapters/{seed}/'):continue
            assert sha(directory/name)==digest,name
        fusion_dir=directory/'frontend';sys.path.insert(0,str(fusion_dir))
        spec=importlib.util.spec_from_file_location('portable_context_predict',fusion_dir/'predict.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.frontend=module.Corrector(fusion_dir,seed)
        torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(directory/'base',local_files_only=True,trust_remote_code=False)
        base=AutoModelForCausalLM.from_pretrained(directory/'base',local_files_only=True,trust_remote_code=False,
            dtype=dtype,attn_implementation='sdpa').to(self.device)
        self.model=PeftModel.from_pretrained(base,directory/'adapters'/str(seed),is_trainable=False)
        assert sum(p.numel() for p in self.model.parameters())==4038983168
        assert sum(p.numel() for n,p in self.model.named_parameters() if 'lora_' in n)==16515072
        assert all(p.dtype==(torch.float32 if 'lora_' in n else torch.bfloat16) for n,p in self.model.named_parameters())
        self.model.config.use_cache=False;self.model.eval()

    def categories(self):return self.frontend.categories()

    def prompt(self,text,category,s,e):
        assert 0<=s<e<=len(text)
        content=f'Error type: {category}\nSentence: '+text[:s]+'<error>'+text[s:e]+'</error>'+text[e:]
        return self.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':content}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)

    @torch.inference_mode()
    def generate(self,requests):
        tok=self.tokenizer;groups={};results=[None]*len(requests)
        for i,(text,category,s,e) in enumerate(requests):groups.setdefault(min(128,max(16,4*(e-s)+8)),[]).append(i)
        for limit,indices in sorted(groups.items()):
            for first in range(0,len(indices),4):
                selected=indices[first:first+4];prompts=[self.prompt(*requests[i]) for i in selected]
                width=max(map(len,prompts));ids=torch.full((len(prompts),width),tok.pad_token_id,dtype=torch.long,device=self.device)
                mask=torch.zeros_like(ids)
                for j,p in enumerate(prompts):ids[j,-len(p):]=torch.tensor(p,device=self.device);mask[j,-len(p):]=1
                config=GenerationConfig(do_sample=False,num_beams=4,length_penalty=.6,early_stopping=True,
                    max_new_tokens=limit,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
                overrides={'do_sample':False,'temperature':None,'top_k':None,'top_p':None}
                resolved,_=self.model._prepare_generation_config(config,use_model_defaults=False,**overrides)
                assert resolved.get_generation_mode().value=='beam_search' and resolved.num_beams==4 and resolved.length_penalty==.6
                assert resolved.early_stopping is True and resolved.max_new_tokens==limit
                output=self.model.generate(input_ids=ids,attention_mask=mask,generation_config=config,
                    use_model_defaults=False,**overrides)[:,width:].cpu().tolist()
                for i,tokens in zip(selected,output):
                    completed=tok.eos_token_id in tokens
                    if completed:tokens=tokens[:tokens.index(tok.eos_token_id)]
                    raw=tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    candidate=raw.strip();text,category,s,e=requests[i];source=text[s:e]
                    valid=completed and bool(candidate) and candidate!=source and '\n' not in candidate and '\r' not in candidate and len(candidate)<=max(44,4*len(source))
                    results[i]={'raw':raw,'candidate':candidate if valid else '', 'completed':completed,'generated_tokens':len(tokens)}
        return results

    @torch.inference_mode()
    def scores(self,requests):
        tok=self.tokenizer;result=[]
        for first in range(0,len(requests),4):
            encoded=[]
            for text,category,s,e,revision in requests[first:first+4]:
                prompt=self.prompt(text,category,s,e);response=tok.encode(revision,add_special_tokens=False)
                assert tok.decode(response,skip_special_tokens=False,clean_up_tokenization_spaces=False)==revision
                assert tok.eos_token_id not in response;response.append(tok.eos_token_id)
                encoded.append((prompt+response,[-100]*len(prompt)+response))
            width=max(len(x[0]) for x in encoded)
            ids=torch.full((len(encoded),width),tok.pad_token_id,dtype=torch.long,device=self.device)
            labels=torch.full_like(ids,-100);attention=torch.zeros_like(ids)
            for i,(a,b) in enumerate(encoded):
                ids[i,:len(a)]=torch.tensor(a,device=self.device);labels[i,:len(a)]=torch.tensor(b,device=self.device);attention[i,:len(a)]=1
            logits=self.model(input_ids=ids,attention_mask=attention,use_cache=False).logits[:,:-1].float()
            targets=labels[:,1:];mask=targets.ne(-100)
            loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),targets,ignore_index=-100,reduction='none')
            result.extend((-loss.sum(1)/mask.sum(1).float().pow(.6)).cpu().tolist())
        return result

    def repair_many(self,inputs):
        requests=[(r['text'],r['error_type'],r['selected_start'],r['selected_end']) for r in inputs]
        generated=self.generate(requests);scoring=[]
        for r,request,g in zip(inputs,requests,generated):
            text,category,s,e=request;scoring.extend([(*request,g['candidate'] or text[s:e]),(*request,r['pred_revision'])])
        probabilities=self.scores(scoring);outputs=[]
        assert all(math.isfinite(p) for p in probabilities)
        for i,(r,g) in enumerate(zip(inputs,generated)):
            use=bool(g['candidate']) and probabilities[2*i]>probabilities[2*i+1]
            revision=g['candidate'] if use else r['pred_revision'];s,e=r['selected_start'],r['selected_end']
            span=r['text'][s:e] if use else r['pred_span']
            outputs.append(dict(g,pred_span=span,pred_revision=revision,selected_start=s,selected_end=e,
                neural_score=probabilities[2*i],incumbent_score=probabilities[2*i+1],
                score_margin=probabilities[2*i]-probabilities[2*i+1],fallback_used=int(not use),
                corrected_text=r['text'][:s]+revision+r['text'][e:] if revision else r['text']))
        return outputs

    def frontend_many(self,examples):
        # Preserve the frontend's verified CPU arithmetic and the decoder's four threads.
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            inputs=[]
            for example in examples:
                text=example['text'];category=example['error_type']
                inputs.append({'text':text,'error_type':category,**self.frontend.predict(text,category)})
            return inputs
        finally:torch.set_num_threads(previous)

    def predict_many(self,examples):
        return self.repair_many(self.frontend_many(examples))

    def predict(self,text,category):return self.predict_many([{'text':text,'error_type':category}])[0]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--text');p.add_argument('--error-type')
    p.add_argument('--seed',type=int,choices=[20260927,20260928,20260929],default=20260927)
    p.add_argument('--device',choices=['cuda'],default='cuda');p.add_argument('--list-error-types',action='store_true')
    args=p.parse_args()
    if args.list_error_types:
        sys.path.insert(0,str(HERE/'frontend'))
        spec=importlib.util.spec_from_file_location('portable_context_categories',HERE/'frontend/predict.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        print(json.dumps(module.Corrector(HERE/'frontend',args.seed).categories()));return
    if not args.text or not args.error_type:p.error('--text and --error-type are required')
    model=Corrector(seed=args.seed,device=args.device)
    print(json.dumps(model.predict(args.text,args.error_type),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
