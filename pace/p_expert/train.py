#!/usr/bin/env python3
"""Single-stage joint training for scratch evo/site-local p+c fusion."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file
import argparse,json,random,time,subprocess,gc,os
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from functools import partial
from scipy.stats import pearsonr,spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import CachedMutationDataset,collate,fullram_peak_bytes,linux_memory_info
from sampler import WTAwareLengthBatchSampler
from model import ScratchEvoLoRAPcFusion
ROOT=Path(__file__).parent
def met(p,y): return {'spearman':float(spearmanr(y,p).statistic),'pearson':float(pearsonr(y,p)[0]),'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(abs(p-y)))}
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def loader(dataset,split,cfg,seed_value):
 d=cfg['data']; kw={'num_workers':int(d['num_workers']),'pin_memory':bool(d['pin_memory']),'collate_fn':partial(collate,deduplicate_wt=bool(d['wt_aware_batching']))}
 if kw['num_workers']:kw.update(persistent_workers=bool(d['persistent_workers']),prefetch_factor=int(d['prefetch_factor']))
 if split=='train' and d['wt_aware_batching']:
  sampler=WTAwareLengthBatchSampler(dataset.frame,cfg['training']['batch_size'],seed_value,int(d['length_bucket_width']));return DataLoader(dataset,batch_sampler=sampler,**kw),sampler
 return DataLoader(dataset,batch_size=cfg['training']['batch_size'],shuffle=split=='train',**kw),None
def call(m,b,d,all=False): return m(b['wt'],b['mut'],b['position0'].to(d,non_blocking=True),b['site_p'].to(d,non_blocking=True),b['site_c'].to(d,non_blocking=True),b['delta_pc'].to(d,non_blocking=True),b['wt_aa'].to(d,non_blocking=True),b['mut_aa'].to(d,non_blocking=True),b['wt_inverse_index'].to(d,non_blocking=True),all)
def ram_stats():
 values={}
 for line in Path('/proc/meminfo').read_text().splitlines(): name,value,*_=line.replace(':','').split();values[name]=int(value)*1024
 return values['MemTotal']-values['MemAvailable'],values['MemAvailable']
def process_rss_bytes(): return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
def gpu_utilization(d):
 try:return float(subprocess.check_output(['nvidia-smi',f'--id={d.index}','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
 except Exception:return None
def _parameters(module): return list(module.parameters())
def _fingerprint(module):
    values=torch.cat([p.detach().float().cpu().flatten()[:8] for p in module.parameters()])
    return [float(x) for x in values[:16]]
def assert_scratch_configuration(m,opt,cfg,out=None):
    """Lightweight guards against accidentally restoring the old frozen evo branch."""
    forbidden=('residue_evo_mutation_centered_pooling','frozen_evo_checkpoint','site_local_gated')
    if any(token in json.dumps(cfg,sort_keys=True) for token in forbidden):
        raise RuntimeError('old evo/site checkpoint dependency found in configuration')
    groups={'LoRA':[p for n,p in m.esm.named_parameters() if 'lora_' in n], 'evo_encoder':_parameters(m.evo_encoder), 'site_gate':_parameters(m.site_gate), 'evo_projection':_parameters(m.evo_projection), 'struct_projection':_parameters(m.struct_projection), 'fusion_gate':_parameters(m.gate), 'prediction_head':_parameters(m.regressor)}
    if any(p.requires_grad for n,p in m.esm.named_parameters() if 'lora_' not in n): raise RuntimeError('ESM base must be frozen')
    if not all(p.requires_grad for p in groups['LoRA']): raise RuntimeError('LoRA must be trainable')
    if any(p.requires_grad for p in m.pythia.parameters()): raise RuntimeError('Pythia must be frozen')
    if any(not p.requires_grad for name in groups for p in groups[name]): raise RuntimeError('a task-specific module is frozen')
    optimizer_ids={id(p) for group in opt.param_groups for p in group['params']}
    missing=[name for name,params in groups.items() if not {id(p) for p in params}.issubset(optimizer_ids)]
    if missing: raise RuntimeError(f'trainable parameters missing from optimizer: {missing}')
    fp={'seed':cfg['seed'],'evo_encoder':_fingerprint(m.evo_encoder),'site_gate':_fingerprint(m.site_gate)}
    # One registry shared by all seed output directories is required to catch an
    # accidental cross-seed reuse; same-seed reruns are permitted to replace only
    # their own fingerprint entry.
    registry=ROOT/'results'/'initialization_fingerprints.json'; registry.parent.mkdir(parents=True,exist_ok=True)
    prior=resolve_paths(json.loads(registry.read_text())) if registry.exists() else {}
    for previous_seed,previous in prior.items():
        if int(previous_seed) != cfg['seed'] and (previous['evo_encoder']==fp['evo_encoder'] or previous['site_gate']==fp['site_gate']):
            raise RuntimeError(f'scratch initialization unexpectedly matches seed {previous_seed}')
    prior[str(cfg['seed'])]=fp; registry.write_text(json.dumps(prior,indent=2)+'\n')
    return groups
def gradient_statistics(m,groups):
    """Return compact per-module norms and first non-finite-gradient details."""
    names={id(p):name for name,p in m.named_parameters()}
    module_stats={}; failures=[]
    for module,params in groups.items():
        seen=[]; square_sum=0.0; maximum=0.0
        for p in params:
            grad=p.grad
            if grad is None: continue
            seen.append(grad)
            finite=torch.isfinite(grad); finite_values=grad[finite]
            if finite_values.numel():
                maximum=max(maximum,float(finite_values.abs().max()))
                square_sum+=float(finite_values.float().square().sum())
            nan_count=int(torch.isnan(grad).sum()); inf_count=int(torch.isinf(grad).sum())
            if nan_count or inf_count:
                failures.append({'module':module,'parameter':names[id(p)],'nan_count':nan_count,'inf_count':inf_count,'finite_abs_max':maximum if finite_values.numel() else None})
        module_stats[module]={'grad_present':bool(seen),'grad_norm':square_sum**0.5,'max_grad_abs':maximum}
        if not seen: failures.append({'module':module,'parameter':None,'error':'no_gradient'})
    return module_stats,failures
def check_gradients_finite(m,groups,context):
    stats,failures=gradient_statistics(m,groups)
    if any(p.grad is not None for n,p in m.esm.named_parameters() if 'lora_' not in n): raise RuntimeError('ESM base received a gradient')
    if any(p.grad is not None for p in m.pythia.parameters()): raise RuntimeError('Pythia received a gradient')
    if failures:
        context={**context,'non_finite_modules':sorted({x['module'] for x in failures}),'first_non_finite_gradient':failures[0],'module_gradient_statistics':stats}
        raise FloatingPointError('unscaled gradient check failed: '+json.dumps(context,sort_keys=True))
    return stats
@torch.no_grad()
def predict(m,l,d):
 m.eval();p=[];y=[];g=[]
 for b in l:
  q,x,*_=call(m,b,d,True);p+=q.float().cpu().tolist();y+=b['target'].tolist();g+=x.float().cpu().tolist()
 return np.asarray(p),np.asarray(y),np.asarray(g)
def snapshot(m,opt,scaler,epoch,val,best,best_epoch,cfg):
 state={k:x.detach().cpu().clone() for k,x in m.state_dict().items()}
 return {'model_state_dict':state,'lora_state_dict':{k:v for k,v in state.items() if 'lora_' in k},'structural_fusion_head_state_dict':{k:v for k,v in state.items() if not k.startswith('esm.')},'optimizer_state_dict':opt.state_dict(),'scaler_state_dict':scaler.state_dict() if scaler else None,'epoch':epoch,'best_epoch':best_epoch,'validation_metrics':val,'best_val_spearman':best,'seed':cfg['seed'],'config':cfg,'pythia_checkpoint':m.pythia_checkpoint,'selection_metric':'validation_spearman','model_report':m.report()}
def evaluate_checkpoint(m,path,cfg,device,checkpoint_type):
 x=torch.load(path,map_location='cpu');m.load_state_dict(x['model_state_dict']);out=Path(cfg.get('run_directory',ROOT/'results'/f"seed_{cfg['seed']}"))/checkpoint_type;out.mkdir(parents=True,exist_ok=True)
 for split,name in [('test','megascale_test'),('s669','s669')]:
  dataset=CachedMutationDataset(split,cfg['paths'],cfg['cache']['root'],cfg['data']);load,_=loader(dataset,split,cfg,cfg['seed']);p,y,g=predict(m,load,device);result=met(p,y)
  result_name='s669_test_results.json' if split=='s669' else 'megascale_test_results.json'
  (out/result_name).write_text(json.dumps({'seed':cfg['seed'],'checkpoint_type':checkpoint_type,'checkpoint_epoch':x['epoch'],'best_epoch':x['best_epoch'],'dataset':'MegaScale test' if split=='test' else 'S669',**result},indent=2)+'\n')
  frame=dataset.frame.copy();frame['prediction']=p;frame['gate_value']=g;frame.to_csv(out/f'predictions_{name}.csv',index=False)
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,default=ROOT/'config.yaml');a.add_argument('--device',default='cuda:1');a.add_argument('--seed',type=int,required=True);a.add_argument('--run-name');a.add_argument('--startup-only',action='store_true');precision=a.add_mutually_exclusive_group();precision.add_argument('--amp',dest='use_amp',action='store_true');precision.add_argument('--no-amp',dest='use_amp',action='store_false');a.set_defaults(use_amp=None);args=a.parse_args();c=resolve_paths(yaml.safe_load(args.config.read_text()));c['seed']=args.seed;t=c['training'];use_amp=bool(t['use_amp']) if args.use_amp is None else args.use_amp;t['use_amp']=use_amp;d=torch.device(args.device);seed(c['seed']);run_name=args.run_name or f"seed_{c['seed']}"
 if Path(run_name).name != run_name: raise ValueError('--run-name must be a single directory name')
 out=ROOT/'results'/run_name;log=out;best_path=out/'checkpoints'/f"best_model_seed{c['seed']}.pt";epoch20_path=out/'checkpoints'/f"epoch20_model_seed{c['seed']}.pt";c['run_name']=run_name;c['run_directory']=str(out)
 if out.exists() or best_path.exists() or epoch20_path.exists() or (log/f"seed{c['seed']}_progress.jsonl").exists() or (log/f"seed{c['seed']}_epochs.jsonl").exists(): raise FileExistsError(f'refusing to overwrite existing artifacts for seed {c["seed"]}; create a new experiment directory')
 out.mkdir(parents=True);log.mkdir(exist_ok=True);best_path.parent.mkdir(exist_ok=True)
 metadata={'run_name':run_name,'seed':c['seed'],'precision':'AMP' if use_amp else 'FP32','device':str(d),'start_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'batch_size':t['batch_size'],'lora_lr':t['lora_learning_rate'],'task_lr':t['fusion_learning_rate'],'cache_mode':c['data']['cache_mode'],'pythia_checkpoint':c['pythia_checkpoints'][c['seed']],'old_evo_checkpoint_loaded':False,'startup_only':args.startup_only}
 (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 pythia=Path(c['pythia_checkpoints'][c['seed']]);
 if not pythia.is_file(): raise FileNotFoundError(pythia)
 if c['data']['cache_mode']=='ram':
  memory,source=linux_memory_info();items,required=fullram_peak_bytes(c['cache']['root'],('train','val'));safe=memory['MemAvailable']*(1-float(c['data'].get('ram_safety_fraction',.2)))
  print(json.dumps({'memory_source':source,'memory_diagnostics':memory,'fullram_cache_items':items,'fullram_required_peak_bytes':required,'fullram_safe_limit_bytes':safe,'fullram_guard_decision':'PASS' if required<=safe else 'FAIL'}),flush=True)
  if required>safe: raise RuntimeError('combined train+val full-RAM cache safety margin would be violated')
 if c['data']['cache_mode']=='ram': print(json.dumps({'event':'before_train_cache_load','process_rss_bytes':process_rss_bytes(),'mem_available_bytes':linux_memory_info()[0]['MemAvailable']}),flush=True)
 data={'train':CachedMutationDataset('train',c['paths'],c['cache']['root'],c['data'])}
 if c['data']['cache_mode']=='ram': print(json.dumps({'event':'after_train_cache_load','process_rss_bytes':process_rss_bytes(),'mem_available_bytes':linux_memory_info()[0]['MemAvailable']}),flush=True)
 data['val']=CachedMutationDataset('val',c['paths'],c['cache']['root'],c['data'])
 if c['data']['cache_mode']=='ram':
  current_rss=process_rss_bytes();print(json.dumps({'event':'after_val_cache_load','process_rss_bytes':current_rss,'mem_available_bytes':linux_memory_info()[0]['MemAvailable']}),flush=True)
  if current_rss>50*2**30: raise RuntimeError('full-RAM cache RSS exceeded 50 GiB before model initialization')
 load={};samplers={}
 for s,x in data.items():load[s],samplers[s]=loader(x,s,c,c['seed'])
 m=ScratchEvoLoRAPcFusion(c,pythia);m.initialize_scratch();m.to(d);lora=[p for n,p in m.named_parameters() if 'lora_' in n];other=[p for n,p in m.named_parameters() if p.requires_grad and 'lora_' not in n]
 opt=torch.optim.Adam([{'params':lora,'lr':t['lora_learning_rate']},{'params':other,'lr':t['fusion_learning_rate']}],weight_decay=t['weight_decay']);scaler=torch.cuda.amp.GradScaler(enabled=True) if use_amp else None;best=-np.inf;best_epoch=0;stale=0
 groups=assert_scratch_configuration(m,opt,c,out)
 print(json.dumps({'experiment':'ESM-LoRA + scratch evo/site-local + frozen Pythia p+c','initialization':{'ESM backbone':'pretrained frozen','LoRA':'random','evo_encoder':'random','site_gate':'random','Pythia':'pretrained frozen','fusion/head':'random'},'parameter_report':m.report(),'optimizer_param_groups':[{'name':'lora','lr':t['lora_learning_rate'],'parameters':sum(p.numel() for p in lora)},{'name':'task_specific','lr':t['fusion_learning_rate'],'parameters':sum(p.numel() for p in other)}]},indent=2),flush=True)
 if args.startup_only:
  (out/'startup_verified.json').write_text(json.dumps({'status':'passed',**metadata},indent=2)+'\n');return
 for epoch in range(1,t['max_epochs']+1):
  samplers['train'].set_epoch(epoch);m.train();opt.zero_grad(set_to_none=True);losses=[];started=time.perf_counter();last=started
  for step,b in enumerate(load['train'],1):
   target=b['target'].to(d,non_blocking=True)
   with torch.cuda.amp.autocast(enabled=use_amp): prediction=call(m,b,d);loss=F.mse_loss(prediction,target)/t['gradient_accumulation_steps']
   if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
   if scaler: scaler.scale(loss).backward()
   else: loss.backward()
   losses.append(float(loss)*t['gradient_accumulation_steps'])
   if step%t['gradient_accumulation_steps']==0 or step==len(load['train']):
    if scaler: scaler.unscale_(opt)
    context={'seed':c['seed'],'epoch':epoch,'batch':step,'loss':float(loss)*t['gradient_accumulation_steps'],'grad_scaler_scale':float(scaler.get_scale()) if scaler else None,'prediction_all_finite':bool(torch.isfinite(prediction).all()),'target_all_finite':bool(torch.isfinite(target).all()),'optimizer_group_lrs':[group['lr'] for group in opt.param_groups]}
    check_gradients_finite(m,groups,context)
    torch.nn.utils.clip_grad_norm_(lora+other,1.0)
    if scaler: scaler.step(opt);scaler.update()
    else: opt.step()
    opt.zero_grad(set_to_none=True)
   if step==1: print(json.dumps({'seed':c['seed'],'run_name':run_name,'precision':'AMP' if use_amp else 'FP32','event':'first_optimizer_step_completed'}),flush=True)
   if step%t['log_every_batches']==0:
    torch.cuda.synchronize(d);now=time.perf_counter();sec=(now-last)/t['log_every_batches'];last=now;used,available=ram_stats();row={'seed':c['seed'],'epoch':epoch,'batch':step,'running_loss':float(np.mean(losses)),'sec_per_batch':sec,'samples_per_sec':len(b['target'])/sec,'unique_wt_per_batch':b['unique_wt_count'],'last2_sequences_per_batch':b['last2_sequence_count'],'gpu_memory_allocated':torch.cuda.memory_allocated(d),'gpu_memory_reserved':torch.cuda.memory_reserved(d),'gpu_utilization_pct':gpu_utilization(d),'ram_used':used,'ram_available':available,'eta_sec':sec*(len(load['train'])-step)};print(json.dumps(row),flush=True);open(log/f"seed{c['seed']}_progress.jsonl",'a').write(json.dumps(row)+'\n')
  p,y,_=predict(m,load['val'],d);v=met(p,y);improved=v['spearman']>best
  if improved:best=v['spearman'];best_epoch=epoch;stale=0;torch.save(snapshot(m,opt,scaler,epoch,v,best,best_epoch,c),best_path)
  else:stale+=1
  saved20=False
  if epoch==20: torch.save(snapshot(m,opt,scaler,epoch,v,best,best_epoch,c),epoch20_path);saved20=True
  pending=epoch<t['minimum_epochs'] and stale>=t['patience'];should_stop=epoch>=t['minimum_epochs'] and stale>=t['patience']
  row={'seed':c['seed'],'epoch':epoch,'train_loss':float(np.mean(losses)),**{'val_'+k:x for k,x in v.items()},'best_spearman':best,'best_epoch':best_epoch,'epochs_since_improvement':stale,'epoch20_checkpoint_saved':saved20,'pending_stop_at_epoch20':pending,'early_stop':should_stop,'epoch_seconds':time.perf_counter()-started,'sampler_audit':samplers['train'].last_audit};open(log/f"seed{c['seed']}_epochs.jsonl",'a').write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
  if should_stop:break
 if not epoch20_path.exists():raise RuntimeError('epoch20 checkpoint was not created')
 # Full-RAM training caches must be released before independently loading test/S669.
 # This preserves cache_mode=ram and its safety margin instead of relaxing the guard.
 del load, data, samplers
 gc.collect()
 # A prior asynchronous CUDA OOM can surface during cache cleanup even after
 # both required checkpoints have been written.  Cleanup is best-effort; do
 # not turn a successful training run into a failed serial job for this.
 if torch.cuda.is_available():
  try: torch.cuda.empty_cache()
  except RuntimeError as error: print(json.dumps({'event':'cuda_cache_cleanup_warning','error':str(error)}),flush=True)
 evaluate_checkpoint(m,best_path,c,d,'best');evaluate_checkpoint(m,epoch20_path,c,d,'epoch20')
 (out/'training_complete.json').write_text(json.dumps({'seed':c['seed'],'status':'complete','best_checkpoint':str(best_path),'epoch20_checkpoint':str(epoch20_path)},indent=2)+'\n')
if __name__=='__main__':main()
