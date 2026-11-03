from __future__ import annotations
import csv, html, json, math, re
from pathlib import Path
from statistics import median
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager
import fitz
from xls_biff_reader import read_xls
                                                                                  
                                                                             
                                                                                   
                                                        
FONT_TMP=Path('/tmp/NotoSansCJK-SC-DTW.otf')
if not FONT_TMP.exists():
    from fontTools.ttLib import TTCollection
    FONT_TMP.parent.mkdir(parents=True,exist_ok=True)
    TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').fonts[2].save(str(FONT_TMP))
font_manager.fontManager.addfont(str(FONT_TMP))

ROOT=Path(__file__).resolve().parents[1] / 'results'
ROOT.mkdir(exist_ok=True)
ALL_METRICS=['predator_diversity','trophic_link_count','node_count','edge_count']
FIGURE_STAGES=['#A5C9EF','#F7CF87','#A2D8B4','#D6C8EB','#D9D9D9','#91C4BC']
PALETTE=['#3B64A1','#CB424A','#21986A','#9A57BE','#E78632','#1695B3','#7366A8','#929B2B','#7E5450','#A86A21','#6F90B3','#4A785A']
plt.rcParams.update({
 'font.family':'Noto Sans CJK SC','font.sans-serif':['Noto Sans CJK SC','DejaVu Sans'],
 'axes.unicode_minus':False,'font.size':10.5,'axes.titlesize':11,
 'axes.labelsize':10,'legend.fontsize':8.5,'pdf.fonttype':3,
 'pdf.use14corefonts':False,'svg.fonttype':'none','figure.facecolor':'white',
 'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
})


def csvwrite(path,head,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(head)
  for row in rows:w.writerow([_csv_val(v) for v in row])

def _csv_val(v):
 if v is None:return ''
 if isinstance(v,(float,np.floating)) and not np.isfinite(v):return ''
 if isinstance(v,(float,np.floating)):return float(v)
 if isinstance(v,(int,np.integer)):return int(v)
 return v

def dtw(a,b):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 assert len(a) and len(b)
 n=len(a);m=len(b)
 D=np.full((n+1,m+1),np.inf,dtype=float)
 D[0,0]=0.0
 pr=np.zeros((n+1,m+1),dtype=np.uint8)
 for i in range(1,n+1):
  ai=a[i-1]
  for j in range(1,m+1):
   diag,up,left=D[i-1,j-1],D[i-1,j],D[i,j-1]
   if diag<=up and diag<=left:
    d=diag;pr[i,j]=0
   elif up<=left:
    d=up;pr[i,j]=1
   else:
    d=left;pr[i,j]=2
   D[i,j]=abs(ai-b[j-1])+d
 i=n;j=m;path=[]
 while i>0 or j>0:
  if not (i>0 and j>0):raise AssertionError('Path left fixed-start boundaries')
  path.append((i-1,j-1))
  p=pr[i,j]
  if p==0:i-=1;j-=1
  elif p==1:i-=1
  else:j-=1
 path.reverse()
 return float(D[n,m]),path

def interp(x,L):
 x=np.asarray(x,dtype=float)
 if len(x)==L:return x.copy()
 if len(x)==1:return np.repeat(x,L).astype(float)
 return np.interp(np.linspace(0,1,L), np.linspace(0,1,len(x)),x)

def make_dba(seqs,iterations=8,L=None):
 if L is None:L=max(1,int(math.floor(np.median([len(s) for s in seqs])+0.5)))
 centroid=np.mean(np.vstack([interp(s,L) for s in seqs]),axis=0)
 for _ in range(iterations):
  groups=[[] for _ in range(L)]
  for seq in seqs:
   _,p=dtw(seq,centroid)
   for i,j in p:groups[j].append(float(seq[i]))
  centroid=np.array([np.mean(g) if g else centroid[j] for j,g in enumerate(groups)],dtype=float)
 return centroid

def stnum(s):
 m=re.search(r'\d+',str(s));return int(m.group()) if m else 999

def load_book(path):
 source=read_xls(path)
 basins=OrderedDict();raw_sheet_notes=[]
 for sh,rows in source.items():
  if not rows:continue
  hdr=[str(v or '').strip() for v in rows[0]]
  normalized={h.lower():j for j,h in enumerate(hdr)}
  stageidx=normalized.get('reference',normalized.get('references'))
  seqidx=normalized.get('layer_seq')
  sampleidx=normalized.get('no.')
  if stageidx is None:raise ValueError(f'Missing Reference in {sh}: {hdr}')
  assert seqidx is not None
  indices={m:normalized[m] for m in ALL_METRICS if m in normalized}
  cleaned=[]
  for ridx,row in enumerate(rows[1:],start=2):
   if all(v is None for v in row):continue
   stage=str(row[stageidx]).strip().lower() if row[stageidx] is not None else ''
   if not re.fullmatch('stage\\d+',stage):
    raise ValueError(f'Invalid stage value {stage!r} in sheet {sh!r} row {ridx}')
   try:layer=int(float(row[seqidx]))
   except:raise ValueError(f'Invalid layer_seq in {sh} row {ridx}')
   sample=row[sampleidx] if sampleidx is not None else layer
   metrics={}
   for m,c in indices.items():
    val=row[c]
    if val is None:raise ValueError(f'Missing {m} in {sh} row {ridx}')
    v=float(val)
    if not math.isfinite(v):raise ValueError(f'Nonfinite metric in {sh} row {ridx}')
    metrics[m]=v
   cleaned.append({'layer':layer,'sample':str(sample),'stage':stage,'metric':metrics,'sheet':sh})
  cleaned.sort(key=lambda r:r['layer'])
  if len(set(r['layer'] for r in cleaned)) != len(cleaned):raise ValueError(f'Duplicate layer_seq: {sh}')
  actualstage=[r['stage'] for r in cleaned];collapsed=[s for i,s in enumerate(actualstage) if i==0 or s!=actualstage[i-1]]
  if len(set(collapsed))!=len(collapsed):raise ValueError(f'Non-contiguous stage segments in {sh}: {collapsed}')
  assert collapsed==sorted(collapsed,key=stnum), (sh,collapsed)
  raw_sheet_notes.append({'sheet':sh,'rows':len(cleaned),'stages':{s:actualstage.count(s) for s in collapsed},'metrics':list(indices)})
  basins[sh]=cleaned
 return basins,raw_sheet_notes

def zscore(series):
 a=np.array(series,dtype=float)
 mu=float(a.mean());s=float(a.std(ddof=0))
 if s<1e-12:s=1.0
 return (a-mu)/s,mu,s

def frac_sample_positions(records):
 n=len(records)
 return np.linspace(0,1,n) if n>1 else np.array([0.5])

def save_figure(fig,figure_dir,name):
 figure_dir.mkdir(parents=True,exist_ok=True)
 paths=[]
 for ext in ['png','pdf']:
  p=figure_dir/f'{name}.{ext}'
  fig.savefig(p,dpi=300 if ext=='png' else None,bbox_inches='tight',pad_inches=0.18)
  paths.append(str(p))
 plt.close(fig)
 return paths

def ax_bg(ax,bounds,stages,alpha=0.10):
 for k,stage in enumerate(stages):
  ax.axvspan(bounds[k],bounds[k+1],color=FIGURE_STAGES[k%len(FIGURE_STAGES)],alpha=alpha,zorder=0,lw=0)

def source_stage_bounds(recs,stages):
                                                                            
 n=len(recs)
 result=[0.0]
 for stage in stages:
  c=sum(r['stage']==stage for r in recs)
  result.append(result[-1]+c/n)
 result[-1]=1.0
 return result


def plot_native(dataset):
 metrics,basins,stages=dataset['metrics'],dataset['basins'],dataset['stages']
 colors=dataset['colors'];figure_dir=dataset['figures']
 bg=dataset['native_bg_bounds']
 def ax_plot(ax,m,full=False):
  ax_bg(ax,bg,stages,alpha=0.09)
  for b,rec in basins.items():
   if m not in dataset['zs'][b]:continue
   vals=dataset['zs'][b][m]
   label=f'{b}'
   if not full:
    avg=dataset['basin_overall'].get((b,m),{}).get('nDTW')
    if avg is not None:label+=f' (nDTW={avg:.3f})'
   ax.plot(frac_sample_positions(rec),vals,color=colors[b],lw=1.55,alpha=.90,label=label)
  template=dataset['template_all'][m]
  ax.plot(np.linspace(0,1,len(template)),template,color='#181818',ls='--',lw=2.3,label='DBA 综合模板',zorder=5)
  ax.set_xlim(0,1)
  ax.grid(alpha=.15,lw=.7)
  ax.set_ylabel('z-score')
  ax.set_title(m,loc='left',fontweight='bold',pad=9)
 for m in metrics:
  fig,ax=plt.subplots(figsize=(13.4,5.3))
  ax_plot(ax,m)
  ax.set_xlabel('各盆地自身相对地层时间 (0–1；未 warping 重采样)')
  ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False,fontsize=8)
  fig.subplots_adjust(bottom=.24)
  save_figure(fig,figure_dir,'01_native_'+m)
 fig,axes=plt.subplots(len(metrics),1,figsize=(13.4,3.3*len(metrics)+1.5),sharex=True,constrained_layout=False)
 axes=np.atleast_1d(axes)
 for ax,m in zip(axes,metrics):ax_plot(ax,m,True)
 axes[-1].set_xlabel('各盆地自身相对地层时间 (0–1；原始采样层数比例)')
 leg=[Line2D([],[],color=colors[b],lw=2,label=b) for b in basins]
 leg+=[Line2D([],[],color='black',ls='--',lw=2.4,label='DBA 综合模板')]
 fig.legend(handles=leg,loc='lower center',bbox_to_anchor=(.5,.005),ncol=min(len(leg),5),frameon=False,fontsize=8)
 fig.subplots_adjust(hspace=.4,bottom=0.08 if len(metrics)>=3 else .12,top=.98,left=.1,right=.98)
 save_figure(fig,figure_dir,f'01_native_combined_{len(metrics)}x1')
                                                                             
 ncol=2 if len(basins)<=6 else 3
 nrow=math.ceil(len(basins)/ncol)
 for m in metrics:
  fig,axes=plt.subplots(nrow,ncol,figsize=(7.3*ncol,2.7*nrow),squeeze=False)
  for ax in axes.flat:ax.set_visible(False)
  for ax,(b,rec) in zip(axes.flat,basins.items()):
   ax.set_visible(True)
   if m not in dataset['zs'][b]:continue
   ax_bg(ax,source_stage_bounds(rec,stages),stages,.12)
   ax.plot(frac_sample_positions(rec),dataset['zs'][b][m],color=colors[b],lw=2)
   ax.set_title(b,loc='left',fontsize=10)
   ax.set_xlim(0,1);ax.grid(alpha=.16)
   ax.set_ylabel('z-score');ax.set_xlabel('本盆地相对层序时间')
  fig.suptitle(f'真实分期边界核验 | {m} | 每盆地独立原始层序',y=.995,fontsize=13,fontweight='bold')
  fig.tight_layout(rect=[0,.02,1,.975])
  save_figure(fig,figure_dir,'01_native_exact_stage_facets_'+m)


def plot_warped(dataset):
 metrics,basins,stages=dataset['metrics'],dataset['basins'],dataset['stages']
 colors=dataset['colors'];figure_dir=dataset['figures']
 def ax_plot(ax,m,full=False):
  templ=dataset['template_all'][m]
  bounds=dataset['template_bounds'][m]
  total=len(templ)
  ax_bg(ax,[v/total for v in bounds],stages,.095)
  xx=np.linspace(0,1,total)
  for b in basins:
   if m not in dataset['zs'][b]:continue
   y=dataset['aligned'][m][b]
   if np.isfinite(y).sum()<1:continue
   label=b
   if not full:
    dist=dataset['basin_overall'].get((b,m),{}).get('nDTW')
    if dist is not None:label+=f' (nDTW={dist:.3f})'
   ax.plot(xx,y,color=colors[b],lw=1.65,alpha=.95,label=label)
  ax.plot(xx,templ,color='#181818',ls='--',lw=2.4,label='DBA 综合模板',zorder=5)
  ax.set_xlim(0,1);ax.grid(alpha=.17);ax.set_ylabel('z-score');ax.set_title(m,loc='left',fontweight='bold',pad=9)
 for m in metrics:
  fig,ax=plt.subplots(figsize=(13.4,5.4))
  ax_plot(ax,m)
  ax.set_xlabel('DBA 模板形态时间 (0–1；由 warping path 映射)')
  ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False,fontsize=8)
  fig.subplots_adjust(bottom=.24)
  save_figure(fig,figure_dir,'02_warped_'+m)
 fig,axes=plt.subplots(len(metrics),1,figsize=(13.4,3.3*len(metrics)+1.6),sharex=True)
 axes=np.atleast_1d(axes)
 for ax,m in zip(axes,metrics):ax_plot(ax,m,True)
 axes[-1].set_xlabel('DBA 模板形态时间 (0–1；warping 后对齐)')
 leg=[Line2D([],[],color=colors[b],lw=2,label=b) for b in basins]
 leg+=[Line2D([],[],color='black',ls='--',lw=2.4,label='DBA 综合模板')]
 fig.legend(handles=leg,loc='lower center',bbox_to_anchor=(.5,.006),ncol=min(5,len(leg)),frameon=False,fontsize=8)
 fig.subplots_adjust(hspace=.4,bottom=.08 if len(metrics)>=3 else .13,top=.98,left=.1,right=.98)
 save_figure(fig,figure_dir,f'02_warped_combined_{len(metrics)}x1')


def panel_shape(n):
 ncol=1 if n==1 else (2 if n<=4 else 3)
 nrow=math.ceil(n/ncol)
 return nrow,ncol

def plot_statistics(dataset):
 basins=list(dataset['basins']);metrics,stages=dataset['metrics'],dataset['stages']
 figdir=dataset['figures'];colors=dataset['colors']
 nrow,ncol=panel_shape(len(metrics))
                                         
 fig,axes=plt.subplots(nrow,ncol,figsize=(9*ncol,4.3*nrow),squeeze=False)
 for ax in axes.flat:ax.set_visible(False)
 for ax,m in zip(axes.flat,metrics):
  ax.set_visible(True);offsets=np.linspace(-.39,.39,len(basins));bw=min(.11,.72/max(1,len(basins)))
  for si,s in enumerate(stages):
   for bi,b in enumerate(basins):
    residual=np.array(dataset['residual_lookup'].get((m,s,b),[]),dtype=float)
    if not len(residual):continue
    bp=ax.boxplot([residual],positions=[si+offsets[bi]],widths=[bw],patch_artist=True,showfliers=False,manage_ticks=False,whis=(5,95))
    for box in bp['boxes']:box.set_facecolor(colors[b]);box.set_alpha(.78);box.set_linewidth(.6)
    for median_artist in bp['medians']:median_artist.set_color('black');median_artist.set_linewidth(1)
    for k in ['caps','whiskers']:
     for artist in bp[k]:artist.set_linewidth(.8)
  ax.set_xticks(range(len(stages)),stages);ax.set_xlim(-.55,len(stages)-.45)
  ax.set_title(m,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.18)
  ax.set_ylabel('残差：观测值 − DBA 匹配节点 (z-score)')
 leg=[Patch(facecolor=colors[b],label=b) for b in basins]
 fig.legend(handles=leg,loc='lower center',bbox_to_anchor=(.5,0),ncol=min(len(basins),6),frameon=False,fontsize=8)
 fig.subplots_adjust(bottom=.13 if nrow<3 else .10,wspace=.24,hspace=.32,left=.08,right=.98,top=.93)
 fig.suptitle('DTW 匹配残差分布｜分 stage × 分盆地',fontweight='bold',fontsize=14)
 save_figure(fig,figdir,f'03_statistics_residual_boxplots_{len(metrics)}panel')
                                                                                       
 fig,axes=plt.subplots(nrow,ncol,figsize=(max(7,2+3.7*ncol),max(4.0,1.5+0.56*len(basins))*nrow),squeeze=False)
 for ax in axes.flat:ax.set_visible(False)
 scores=[v['nDTW'] for v in dataset['ndtw_records']]
 clim=max(scores) if scores else 1.0
 cmap=plt.get_cmap('YlOrRd')
 for ax,m in zip(axes.flat,metrics):
  ax.set_visible(True)
  for bi,b in enumerate(basins):
   for sj,st in enumerate(stages):
    rr=dataset['per_dist'].get((m,st,b))
    if rr is None:color='#F1F2F4'
    else:color=cmap(.15+.80*rr['nDTW']/max(clim,.01))
    ax.add_patch(Rectangle((sj,bi),1,1,facecolor=color,edgecolor='white',linewidth=.7))
    s=f'{rr["nDTW"]:.3f}' if rr is not None else '—'
    col='white' if rr and rr['nDTW']/max(clim,.01)>.8 else '#1D2A3B'
    ax.text(sj+.5,bi+.5,s,ha='center',va='center',color=col,fontsize=9)
  ax.set_xlim(0,len(stages));ax.set_ylim(0,len(basins));ax.invert_yaxis();ax.set_aspect('auto')
  ax.set_yticks(np.arange(len(basins))+.5,basins,fontsize=9)
  ax.set_xticks(np.arange(len(stages))+.5,stages)
  ax.set_title(m,loc='left',fontweight='bold',pad=9)
  for spine in ax.spines.values():spine.set_visible(False)
 fig.suptitle('nDTW 热力图｜灰格为缺失 stage 或指标',fontsize=14,fontweight='bold')
 fig.subplots_adjust(hspace=.36,wspace=.50,bottom=.08,top=.92,left=.20,right=.97)
 save_figure(fig,figdir,f'03_statistics_ndtw_heatmap_{len(metrics)}panel')
                                                                    
 for m in metrics:
  fig,ax=plt.subplots(figsize=(6.4, .52*len(basins)+2.1))
  for bi,b in enumerate(basins):
   for sj,st in enumerate(stages):
    rr=dataset['per_dist'].get((m,st,b))
    col=cmap(.15+.8*rr['nDTW']/max(clim,.01)) if rr else '#F1F2F4'
    ax.add_patch(Rectangle((sj,bi),1,1,facecolor=col,edgecolor='white'))
    ax.text(sj+.5,bi+.5, f'{rr["nDTW"]:.3f}' if rr else '—',ha='center',va='center',fontsize=9)
  ax.set_xlim(0,len(stages));ax.set_ylim(0,len(basins));ax.invert_yaxis();ax.set_xticks(np.arange(len(stages))+.5,stages)
  ax.set_yticks(np.arange(len(basins))+.5,basins);ax.set_title('nDTW | '+m,pad=10)
  for spine in ax.spines.values():spine.set_visible(False)
  fig.tight_layout()
  save_figure(fig,figdir,f'03_statistics_ndtw_heatmap_{m}')
               
 fig,axes=plt.subplots(nrow,ncol,figsize=(max(11,6*ncol),4.5*nrow),squeeze=False)
 for ax in axes.flat:ax.set_visible(False)
 for ax,m in zip(axes.flat,metrics):
  ax.set_visible(True);bx=np.arange(len(basins)); width=.80/max(1,len(stages))
  for sj,st in enumerate(stages):
   vals=[dataset['per_dist'].get((m,st,b),{}).get('nDTW',np.nan) for b in basins]
   ax.bar(bx-.4+(sj+.5)*width,vals,width=width*.95,label=st,color=FIGURE_STAGES[sj%len(FIGURE_STAGES)],edgecolor='#606060',linewidth=.5)
  ax.set_xticks(bx,basins,rotation=35,ha='right',fontsize=8)
  ax.set_title(m,loc='left',fontweight='bold');ax.set_ylabel('nDTW')
  ax.grid(axis='y',alpha=.18)
 axes.flat[0].legend(frameon=False,ncol=len(stages),fontsize=8)
 fig.suptitle('分 stage nDTW｜盆地与 DBA 模板',fontsize=14,fontweight='bold')
 fig.subplots_adjust(bottom=.15 if nrow<=2 else .13,hspace=.55,wspace=.23,top=.92,left=.07,right=.98)
 save_figure(fig,figdir,f'03_statistics_ndtw_grouped_bars_{len(metrics)}panel')


def plot_paths(dataset):
 rep='node_count' if 'node_count' in dataset['metrics'] else 'trophic_link_count'
 basins=list(dataset['basins']);stages=dataset['stages'];figdir=dataset['figures']
 ncol=2 if len(basins)<=6 else 3;nrow=math.ceil(len(basins)/ncol)
 fig,axes=plt.subplots(nrow,ncol,figsize=(6.5*ncol,3.9*nrow),squeeze=False)
 for ax in axes.flat:ax.set_visible(False)
 for ax,b in zip(axes.flat,basins):
  ax.set_visible(True);data=dataset['basins'][b]
  n=len(data);L=len(dataset['template_all'][rep]);shift=0
  for sj,st in enumerate(stages):
   sub=dataset['path_lookup'].get((rep,st,b))
   t_off=dataset['template_bounds'][rep][sj]
   if not sub:continue
   indices=[i for i,r in enumerate(data) if r['stage']==st]
   original_off=min(indices)
   xx=np.array([i+original_off+1 for i,j in sub]);yy=np.array([j+t_off+1 for i,j in sub])
   ax.plot(xx,yy,'-',color=PALETTE[sj],lw=1.8,alpha=.9,label=st)
   xi=original_off+1; xf=max(indices)+1; y0=t_off+1; yf=dataset['template_bounds'][rep][sj+1]
   ax.plot([xi,xf],[y0,yf],ls=':',color='#8E929B',alpha=.8,lw=1)
   if len(indices)==1 and len(sub)>1:
    ax.plot(xx,yy,color='#D21F39',lw=4,zorder=9)
    ax.annotate(f'{b} {st}: 1层→{len(sub)}模板节点',xy=(xx[len(xx)//2],yy[len(yy)//2]),xytext=(0.05,.86),textcoords='axes fraction',color='#C51929',fontsize=8,arrowprops={'arrowstyle':'->','color':'#C51929'})
  ax.set_title(b,loc='left',fontweight='bold');ax.set_xlabel('原始盆地采样序号（不重采样）');ax.set_ylabel('DBA 模板节点序号')
  ax.grid(alpha=.15);ax.set_xlim(.5,n+.5);ax.set_ylim(.5,L+.5)
 handles=[Line2D([],[],color=PALETTE[sj],lw=2,label=st) for sj,st in enumerate(stages)]
 handles+=[Line2D([],[],color='#8E929B',ls=':',lw=2,label='分期线性参考线')]
 fig.legend(handles=handles,loc='lower center',ncol=len(handles),bbox_to_anchor=(.5,0),frameon=False)
 fig.suptitle('DTW Warping Path 轨迹图｜代表指标：'+rep,fontsize=14,fontweight='bold')
 fig.subplots_adjust(bottom=.09,top=.94,hspace=.50,wspace=.30,left=.07,right=.97)
 save_figure(fig,figdir,f'04_{rep}_warping_paths_{len(basins)}basin')


def sensitivity(dataset,reps=50):
 basins=dataset['basins'];metrics=dataset['metrics'];stages=dataset['stages']
 nd=dataset['per_dist']
 outputs=[]
 eligible=[b for b,seq in basins.items() if len(seq)>=2*median([len(x) for n,x in basins.items() if n!=b]) and len(basins)>=4]
                                                                                                     
 eligible=sorted(set(eligible),key=lambda b:-len(basins[b]))[:2]
 rng=np.random.default_rng(20261026)
 for b in eligible:
  rec=basins[b]
  for fraction in [.5,.25]:
   for rep in range(reps):
    keep=[]
    for st in stages:
     indices=[i for i,row in enumerate(rec) if row['stage']==st]
     if len(indices)==0:continue
     num=max(1,int(round(len(indices)*fraction)))
                                               
     if num>=len(indices):sel=indices
     elif num==1:sel=[indices[len(indices)//2]]
     elif num==2:sel=[indices[0],indices[-1]]
     else:
      sel=[indices[0],indices[-1]]+sorted(rng.choice(indices[1:-1],size=num-2,replace=False).tolist())
     keep.extend(sel)
    keep=sorted(set(keep))
    for m in metrics:
     if m not in dataset['zs'][b]:continue
     cost=0;length=0;covered=0
     for st in stages:
      vec=np.array([dataset['zs'][b][m][i] for i in keep if rec[i]['stage']==st])
      if not len(vec):continue
      temp=dataset['dba'][(m,st)]
      d,p=dtw(vec,temp);cost+=d;length+=len(p);covered+=1
     if length:
      orig=dataset['basin_overall'][(b,m)]['nDTW']
      outputs.append([b,m,fraction,rep,len(keep),covered,cost/length,orig,cost/length-orig])
 out=dataset['csvdir']
 csvwrite(out/'sampling_sensitivity_trials.csv', ['Basin','indicator','retained_fraction','replicate','retained_layers','stages','nDTW_reduced','nDTW_original','delta_nDTW'],outputs)
 summ=[]
 for key in {(r[0],r[1],r[2]) for r in outputs}:
  vals=np.array([r[6] for r in outputs if (r[0],r[1],r[2])==key])
  orig=next(r[7] for r in outputs if (r[0],r[1],r[2])==key)
  summ.append([*key,len(vals),orig,float(vals.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975)),float(vals.mean()-orig)])
 summ.sort(key=lambda r:(r[0],r[1],-r[2]))
 csvwrite(out/'sampling_sensitivity_summary.csv',['Basin','indicator','retained_fraction','n_trials','nDTW_original','nDTW_reduced_mean','nDTW_reduced_p025','nDTW_reduced_p975','mean_delta'],summ)
 return eligible,summ


def input_coverage(dataset):
 basins=dataset['basins'];stages=dataset['stages']
 rows=[]
 for b,rec in basins.items():
  for st in stages:
   row=[b,st,sum(x['stage']==st for x in rec),len(rec),','.join(dataset['zs'][b])]
   rows.append(row)
 return rows


def run_analysis(name, basins, base, description='', original=None, plots=True, sens=True):
 base.mkdir(parents=True,exist_ok=True)
 figs=base/'figures';csvdir=base/'csv';reports=base/'reports'
 for d in (figs,csvdir,reports):d.mkdir(exist_ok=True,parents=True)
 basins=OrderedDict((b,records) for b,records in basins.items())
 stages=sorted({r['stage'] for r in basins.values() for r in r},key=stnum)
 all_indicators=[m for m in ALL_METRICS if any(m in r[0]['metric'] for r in basins.values())]
 eligible=[m for m in all_indicators if sum(m in r[0]['metric'] for r in basins.values())>=2]
 omitted=[m for m in all_indicators if m not in eligible]
 if not eligible:raise ValueError(f'Insufficient common metrics in {name}')
 palette={b:PALETTE[i%len(PALETTE)] for i,b in enumerate(basins)}
                                                                                                     
 zs={};zs_records=[];stats=[];rawrows=[]
 for b,rec in basins.items():
  zs[b]={}
  for m in rec[0]['metric']:
   seq=np.array([r['metric'][m] for r in rec])
   z,mean,std=zscore(seq)
   zs[b][m]=z
   stats.append([b,m,len(seq),mean,std])
  for i,r in enumerate(rec):
   rawrows.append([b,r['layer'],r['sample'],r['stage']]+[r['metric'].get(m) for m in ALL_METRICS])
   zs_records.append([b,r['layer'],r['sample'],r['stage']]+[float(zs[b][m][i]) if m in zs[b] else None for m in ALL_METRICS])
 csvwrite(csvdir/'raw_original_all.csv',['Basin','layer_seq','sample','stage']+ALL_METRICS,rawrows)
 csvwrite(csvdir/'zscore_all.csv',['Basin','layer_seq','sample','stage']+[m+'_z' for m in ALL_METRICS],zs_records)
 csvwrite(csvdir/'zscore_params.csv',['Basin','indicator','n_samples','mean_all_stages','std_all_stages_ddof0'],stats)
 csvwrite(csvdir/'stage_sample_counts.csv',['Basin','stage','n_layers','basin_total','indicators_present'], [[b,st,sum(x['stage']==st for x in rec),len(rec),','.join(zs[b])] for b,rec in basins.items() for st in stages])
                                                                                       
 dba={};meta=[];template_all={};bounds={}
 for m in eligible:
  for st in stages:
   seqs=[zs[b][m][[i for i,r in enumerate(rec) if r['stage']==st]] for b,rec in basins.items() if m in zs[b] and any(r['stage']==st for r in rec)]
   contributors=[b for b,rec in basins.items() if m in zs[b] and any(r['stage']==st for r in rec)]
   if len(seqs)<2:
    raise ValueError(f'Only {len(seqs)} sequence(s) for {m} {st}: DBA prohibited. Adjust grouping')
   L=max(1,int(math.floor(np.median([len(s) for s in seqs])+.5)))
   center=make_dba(seqs,iterations=8,L=L)
   dba[(m,st)]=center
   meta.append([m,st,L,len(seqs),'; '.join(contributors),'; '.join(map(str,[len(s) for s in seqs])),8,'median','mean of linear interpolations','L1'])
  total=[];ends=[0]
  for st in stages:
   total.extend(dba[(m,st)]);ends.append(len(total))
  template_all[m]=np.array(total);bounds[m]=ends
 csvwrite(csvdir/'dba_settings.csv',['indicator','stage','template_nodes','n_contributors','contributors','stage_sample_lengths','iterations','template_len_method','initialization','distance'],meta)
 trows=[]
 for m in eligible:
  for sj,st in enumerate(stages):
   offset=bounds[m][sj]
   for j,v in enumerate(dba[(m,st)]):trows.append([m,st,j+1,offset+j+1,float(v)])
 csvwrite(csvdir/'dba_templates_long.csv',['indicator','stage','stage_node','global_template_node','dba_z'],trows)
 for m in eligible:
  csvwrite(csvdir/f'dba_{m}_combined.csv',['global_template_node','stage','stage_node','dba_z'],[[i+1,st,j+1,float(dba[(m,st)][j])] for sidx,st in enumerate(stages) for i,j in [(bounds[m][sidx]+j,j) for j in range(len(dba[(m,st)]))]])
                                                                       
 ndtw_records=[];per_dist={};pair_records=[];pathrows=[];resrows=[];alignedrows=[]
 path_lookup={};residual_lookup={};aligned={m:{b:np.full(len(template_all[m]),np.nan) for b in basins} for m in eligible}
 stagevals={}
 for m in eligible:
  for st in stages:
   template=dba[(m,st)]
   sub={b:zs[b][m][[i for i,r in enumerate(rec) if r['stage']==st]] for b,rec in basins.items() if m in zs[b] and any(r['stage']==st for r in rec)}
   stagevals[(m,st)]=sub
   offset=bounds[m][stages.index(st)]
   for b,vec in sub.items():
    cost,path=dtw(vec,template)
    assert np.isclose(cost,sum(abs(vec[i]-template[j]) for i,j in path),atol=1e-9)
    assert path[0]==(0,0) and path[-1]==(len(vec)-1,len(template)-1)
    assert all((i2-i1,j2-j1) in [(1,1),(1,0),(0,1)] for (i1,j1),(i2,j2) in zip(path,path[1:]))
    dist=cost/len(path)
    rr={'Basin':b,'indicator':m,'stage':st,'DTW':cost,'warping_path_length':len(path),'nDTW':dist,'sample_n':len(vec),'template_n':len(template)}
    per_dist[(m,st,b)]=rr;ndtw_records.append(rr)
    path_lookup[(m,st,b)]=path
    indices=[i for i,r in enumerate(basins[b]) if r['stage']==st]
    bins=[[] for _ in range(len(template))]
    residual=[]
    for i,j in path:
     bins[j].append(float(vec[i]))
     residual.append(float(vec[i]-template[j]))
     pathrows.append([m,st,b,basins[b][indices[i]]['layer'],i+1,indices[i]+1,j+1,offset+j+1,basins[b][indices[i]]['sample']])
     resrows.append([m,st,b,indices[i]+1,basins[b][indices[i]]['sample'],j+1,offset+j+1,float(vec[i]),float(template[j]),float(vec[i]-template[j])])
    residual_lookup[(m,st,b)]=residual
    for j,vals in enumerate(bins):
     aligned[m][b][offset+j]=float(np.mean(vals)) if vals else np.nan
     alignedrows.append([m,st,b,j+1,offset+j+1,len(vals),float(np.mean(vals)) if vals else None,float(template[j])])
                                     
   all_series={**sub,'DBA_template':template}
   names=list(all_series)
   for a in names:
    for b in names:
     if (m,st,a)==(m,st,b):cost=0.0;L=len(all_series[a])
     else:cost,path=dtw(all_series[a],all_series[b]);L=len(path)
     pair_records.append([m,st,a,b,cost,L,cost/L])
 csvwrite(csvdir/'ndtw_by_indicator_stage.csv',['Basin','indicator','stage','DTW_cost','warping_path_length','nDTW','basin_layer_n','template_node_n'],[[r[k] for k in ('Basin','indicator','stage','DTW','warping_path_length','nDTW','sample_n','template_n')] for r in ndtw_records])
 csvwrite(csvdir/'ndtw_full_matrix_long.csv',['indicator','stage','entity_i','entity_j','DTW_cost','path_length','nDTW'],pair_records)
 csvwrite(csvdir/'warping_paths_long.csv',['indicator','stage','Basin','original_layer_seq','stage_local_sample_idx','basin_global_idx','stage_local_template_idx','template_global_idx','original_sample_id'],pathrows)
 csvwrite(csvdir/'residuals_long.csv',['indicator','stage','Basin','original_global_idx','sample_id','stage_template_node','global_template_node','observed_z','dba_z','residual_z'],resrows)
 csvwrite(csvdir/'warped_aligned_long.csv',['indicator','stage','Basin','stage_template_node','global_template_node','warping_assignments','aligned_mean_z','dba_z'],alignedrows)
                                                          
 pair_map={(r[0],r[1],r[2],r[3]):r[6] for r in pair_records}
 for m in eligible:
  for st in stages:
   entities=[b for b in basins if b in stagevals[(m,st)]]+['DBA_template']
   csvwrite(csvdir/f'ndtw_matrix_{m}_{st}.csv',['entity']+entities,[[a]+[pair_map[(m,st,a,b)] for b in entities] for a in entities])
                                                                                                
 longall=[];basin_overall={}
 for m in eligible:
  for a in list(basins)+['DBA_template']:
   for b in list(basins)+['DBA_template']:
    overlap=[s for s in stages if (m,s,a,b) in pair_map]
    psub=[r for r in pair_records if r[0]==m and r[1] in overlap and r[2]==a and r[3]==b]
    if not psub:continue
    cost=sum(p[4] for p in psub);length=sum(p[5] for p in psub)
    total=cost/length
    longall.append([m,a,b,len(overlap),';'.join(overlap),cost,length,total])
    if b=='DBA_template' and a!='DBA_template':
     basin_overall[(a,m)]={'nDTW':total,'stages':len(overlap),'coverage':overlap,'cost':cost,'pathlen':length}
 csvwrite(csvdir/'ndtw_full_timespan_per_indicator_long.csv',['indicator','entity_i','entity_j','n_common_stages','common_stages','summed_DTW_cost','summed_path_length','nDTW'],longall)
 for m in eligible:
  lookup={(row[1],row[2]):row[-1] for row in longall if row[0]==m}
  ent=list(basins)+['DBA_template']
  csvwrite(csvdir/f'ndtw_full_timespan_matrix_{m}.csv',['entity']+ent,[[a]+[lookup.get((a,b)) for b in ent] for a in ent])
                                                                                           
 entity=list(basins)+['DBA_template']
 perlookup={(r[0],r[1],r[2]):r[-1] for r in longall}
 mdrecords=[]
 for a in entity:
  for b in entity:
   used=[m for m in eligible if (m,a,b) in perlookup]
   dist=float(np.mean([perlookup[m,a,b] for m in used])) if used else None
   overlap=sorted(set(st for m in used for st in stages if (m,st,a,b) in pair_map),key=stnum)
   mdrecords.append([a,b,dist,len(used),';'.join(used),len(overlap),';'.join(overlap)])
 csvwrite(csvdir/'ndtw_multidimensional_matrix.csv',['entity']+entity,[[a]+[next((r[2] for r in mdrecords if r[0]==a and r[1]==b),None) for b in entity] for a in entity])
 csvwrite(csvdir/'ndtw_multidimensional_long.csv',['entity_i','entity_j','mean_nDTW','n_shared_metrics','shared_metrics','n_common_stages','common_stages'],mdrecords)
 csvwrite(csvdir/'ndtw_basin_vs_DBA_full.csv',['Basin','indicator','nDTW_full','n_covered_stages','covered_stages','sum_DTW','sum_path_len'],[[b,m,rr['nDTW'],rr['stages'],';'.join(rr['coverage']),rr['cost'],rr['pathlen']] for (b,m),rr in basin_overall.items()])
                                                                                          
 stage_summ=[]
 for m in eligible:
  for st in stages:
   present=[r for r in ndtw_records if r['indicator']==m and r['stage']==st]
   reliable=[r for r in present if r['sample_n']>1]
   stage_summ.append([m,st,len(present),len(reliable),float(np.mean([r['nDTW'] for r in present])),float(np.mean([r['nDTW'] for r in reliable])) if reliable else None,min(r['nDTW'] for r in present),max(r['nDTW'] for r in present)])
 csvwrite(csvdir/'ndtw_stage_descriptive_summary.csv',['indicator','stage','n_basins','n_with_ge2_layers','mean_nDTW_all','mean_nDTW_excl_single_layer','min_nDTW','max_nDTW'],stage_summ)
                                                                                             
 stage_raw_rows=[]
 for b,rec in basins.items():
  for m in zs[b]:
   prev=None
   for st in stages:
    v=np.array([r['metric'][m] for r in rec if r['stage']==st],dtype=float)
    if len(v)==0:continue
    pct=100*(v.mean()/prev-1) if prev is not None and prev!=0 else None
    stage_raw_rows.append([b,m,st,len(v),float(v.mean()),float(v.min()),float(v.max()),pct])
    prev=float(v.mean())
 csvwrite(csvdir/'raw_stage_means_and_change_pct.csv',['Basin','indicator','stage','n_layers','raw_mean','raw_min','raw_max','change_from_previous_present_stage_pct'],stage_raw_rows)
                        
 morphology=[]
 for m in eligible:
  for st in stages:
   z=dba[(m,st)]
   morphology.append([m,st,len(z),float(z[0]),float(z[-1]),float(np.min(z)),float(np.max(z)),float(np.ptp(z)),float(z[-1]-z[0]),int(np.argmin(z)+1),int(np.argmax(z)+1)])
 csvwrite(csvdir/'dba_morphology_summary.csv',['indicator','stage','template_len','first_z','last_z','min_z','max_z','range_z','last_minus_first_z','argmin_node_1based','argmax_node_1based'],morphology)
                                                                                                                 
 nfracs=[]
 for b,rec in basins.items():
  n=len(rec);nfracs.append([sum(r['stage']==st for r in rec)/n for st in stages])
 meanfrac=np.array(nfracs).mean(axis=0);meanfrac=meanfrac/meanfrac.sum()
 native_bg_bounds=np.cumsum([0]+meanfrac.tolist())
 dataset={'name':name,'basins':basins,'stages':stages,'metrics':eligible,'all_metrics':all_indicators,'omitted':omitted,'zs':zs,
          'dba':dba,'template_all':template_all,'template_bounds':bounds,'colors':palette,'figures':figs,'csvdir':csvdir,
          'native_bg_bounds':native_bg_bounds,'per_dist':per_dist,'ndtw_records':ndtw_records,'basin_overall':basin_overall,
          'path_lookup':path_lookup,'aligned':aligned,'residual_lookup':residual_lookup,'pair_records':pair_records,
          'stage_summ':stage_summ,'morphology':morphology,'stage_raw_rows':stage_raw_rows}
 if plots:
  plot_native(dataset)
  plot_warped(dataset)
  plot_statistics(dataset)
  plot_paths(dataset)
 sensitivity_basin=[];sensitivity_summary=[]
 if sens:
  sensitivity_basin,sensitivity_summary=sensitivity(dataset,50)
 dataset['sensitivity_basin']=sensitivity_basin
 dataset['sensitivity_summary']=sensitivity_summary
 write_report(dataset,base,reports,description,original)
                                                                             
 nchecks=0
 for m in eligible:
  for st in stages:
   keys=list(stagevals[(m,st)])+['DBA_template']
   for a in keys:
    for b in keys:
     x=pair_map[m,st,a,b];y=pair_map[m,st,b,a]
     assert abs(x-y)<1e-10,(m,st,a,b,x,y);nchecks+=1
     if a==b:assert abs(x)<1e-12;nchecks+=1
 for i,r in enumerate(ndtw_records):
  assert r['nDTW']>=-1e-12;nchecks+=1
 figfiles=list(figs.glob('*'))
 pngs=list(figs.glob('*.png'));pdfs=list(figs.glob('*.pdf'))
 assert len(pngs)==len(pdfs)
 for p in pdfs:
  doc=fitz.open(p)
  for page in doc:
   assert not page.get_images(full=True),f'Raster embedded in PDF {p}'
  nchecks+=1
 manifest={'name':name,'description':description,'date':'2026-10-26','input':str(original) if original else '',
           'basins':list(basins),'stages':stages,'eligible_indicators':eligible,'excluded_single_basin_indicators':omitted,
           'stage_counts':{b:{st:sum(r['stage']==st for r in rec) for st in stages} for b,rec in basins.items()},
           'figure_pairs':len(pngs),'csv_count':len(list(csvdir.glob('*.csv'))),'pdf_report':(reports/'analysis_report.pdf').exists(),
           'validation_checks':nchecks,'DBA_iterations':8,'DBA_template_len':'median of available basin lengths',
           'DTW_distance':'absolute difference','DTW_warping':'fixed endpoints, three standard steps, no window',
           'nDTW':'cumulative absolute cost / warping path length','standardization':'each basin/metric across ALL its stages, ddof=0',
           'native_axis':'raw basin-relative layer time no warping','warped_axis':'per indicator, stage-concatenated DBA template nodes 0->1'}
 (base/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 return dataset,manifest


def write_report(dataset,base,reports,description='',original=None):
 ds=dataset
 basins=list(ds['basins']);stages=ds['stages'];metrics=ds['metrics']
 md=[];body=[]
 def h(level,text):
  md.append('#'*level+' '+text+'\n');body.append(f'<h{level}>{html.escape(text)}</h{level}>')
 def p(text):
  md.append(text+'\n');body.append(f'<p>{html.escape(text)}</p>')
 def b(phrase):
  md.append('- '+phrase);body.append('<li>'+html.escape(phrase)+'</li>')
 def tab(headers,rows,widths=None):
  clean=[[str(x) if x is not None else '—' for x in row] for row in rows]
  md.append('| '+' | '.join(headers)+' |')
  md.append('| '+' | '.join(['---']*len(headers))+' |')
  md.extend('| '+' | '.join(row)+' |' for row in clean)
  md.append('')
  t='<table><thead><tr>'+''.join(f'<th>{html.escape(c)}</th>' for c in headers)+'</tr></thead><tbody>'
  for row in clean:
   t+='<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in row)+'</tr>'
  t+='</tbody></table>'
  body.append(t)
 title=f'{ds["name"]}｜分期 DBA/DTW 波形匹配分析报告'
 h(1,title)
 p('分析日期：2026-10-26。输入：'+(str(original.name) if original else '原始数据工作簿')+'。'+description)
 h(2,'1. 数据结构与分析口径')
 p(f'独立工作表（盆地/剖面）数：{len(basins)}；符合多盆地 DBA 构建条件的指标：'+', '.join(metrics)+'；原始地层分期：'+', '.join(stages)+'。')
 if ds['omitted']:
  p('以下指标仅由单一工作表提供，已经保留在原始与 z-score 数据表中，不纳入多盆地 DBA 或其 nDTW 矩阵：'+', '.join(ds['omitted'])+'。')
 rows=[]
 for bname,rec in ds['basins'].items():
  rows.append([bname,str(len(rec))]+[str(sum(r['stage']==st for r in rec)) for st in stages]+[', '.join(ds['zs'][bname])])
 tab(['盆地 / 工作表','总采样层']+stages+['有效指标'],rows)
 p('预处理：同一盆地的同一指标使用其全部原始采样层（包括 stage0 或 stage4，如有）共同计算 z-score；标准差采用总体定义 ddof=0，禁止按期各自标准化。')
 p('模板：每个指标、每个 stage 独立构建，模板节点数取该期所有可用盆地采样长度的中位数（遇 .5 向上取整）。将各盆地线性插值到该长度后计算均值作为非单盆地初值；DBA 固定更新 8 轮，不做后平滑。')
 p('距离：按 stage 分别求 DTW，L1 绝对差局部代价，固定起止点，允许 (1,1)/(1,0)/(0,1) 三种单调步进，无 Sakoe-Chiba 窗；nDTW=DTW 累积代价 / warping path 节点数。完整阶段的 nDTW 通过先累计各 stage 代价和路径长度，再相除得到。多指标距离为有效共同指标的等权平均。')
 p('主图：盆地自身原始层序的相对时间 0–1，保留所有采样点、没有 warping 重采样。由于不同盆地 stage 所占比例不同，叠加图的淡色背景是各盆地 stage 比例的平均展示参照，不能代表每个盆地真实 stage 分界。每盆地独立分期的核验图在 01_native_exact_stage_facets_* 中。')
 p('补充图：以各 stage 连续拼接的 DBA 模板节点构成形态时间 0–1，用 warping path 将盆地观测值映射到每一模板节点；缺失 stage 为断线，且各期路径绝不跨阶段。')
 h(2,'2. 完整 nDTW 结果（盆地对综合模板）')
                            
 for m in metrics:
  h(3,m)
  rr=[]
  for basin in basins:
   vals=[]
   for st in stages:
    x=ds['per_dist'].get((m,st,basin))
    vals.append(f'{x["nDTW"]:.3f}' if x else '—')
   overall=ds['basin_overall'].get((basin,m))
   vals.append(f'{overall["nDTW"]:.3f}' if overall else '—')
   rr.append([basin]+vals)
  tab(['盆地']+stages+['全覆盖阶段合并 nDTW'],rr)
 h(2,'3. stage 差异与综合描述')
 sums=[]
 for row in ds['stage_summ']:
  m,s,n_all,n_eff,allmean,excl,minval,maxval=row
  sums.append([m,s,str(n_all),str(n_eff),f'{allmean:.3f}',f'{excl:.3f}' if excl is not None else '—',f'{minval:.3f}–{maxval:.3f}'])
 tab(['指标','stage','盆地数','≥2 层盆地数','平均 nDTW','剔除单层后均值','min–max'],sums)
 p('上表是各 stage 的“对该 stage 综合模板的形态偏离”，不是网络绝对连接数或生态功能强度。同一 stage 的平均值仅汇总该期有样本且该指标可用的盆地；随着参与盆地集合变化，跨 stage 直接比较均值可能混入样本构成效应。')
 for m in metrics:
  for st in stages:
   rr=[r for r in ds['ndtw_records'] if r['indicator']==m and r['stage']==st]
   minr=min(rr,key=lambda x:x['nDTW']);maxr=max(rr,key=lambda x:x['nDTW'])
   p(f'{m} | {st}：最接近模板的是 {minr["Basin"]}（nDTW={minr["nDTW"]:.3f}），偏离最大的是 {maxr["Basin"]}（nDTW={maxr["nDTW"]:.3f}）。')
 h(2,'4. 四类指标的模板形态演化（按实际可用指标）')
 if not metrics: p('无可比较指标。')
 for m in metrics:
  h(3,m)
  morphology=[row for row in ds['morphology'] if row[0]==m]
  tab(['时期','DBA 节点','初值 z','末值 z','最低 z','最高 z','振幅 Δz','末-初 z'],
      [[r[1],str(r[2]),f'{r[3]:+.2f}',f'{r[4]:+.2f}',f'{r[5]:+.2f}',f'{r[6]:+.2f}',f'{r[7]:.2f}',f'{r[8]:+.2f}'] for r in morphology])
  p('上述振幅是标准化波形模板的极值差。需要换算原始网络连接数时，应使用各盆地本指标自己的原始标准差；百分比应直接从相同地层窗口的原始边数、营养联系数计算，而不是将 z-score 变化当作百分比。')
 h(2,'5. 盆地间偏离模式')
 avgrows=[]
 for basin in basins:
  arr=[(m,ds['basin_overall'][(basin,m)]) for m in metrics if (basin,m) in ds['basin_overall']]
  val=float(np.mean([x['nDTW'] for m,x in arr])) if arr else None
  avgrows.append([basin, len(arr),val, '; '.join(f'{m}: {x["nDTW"]:.3f}({x["stages"]}期)' for m,x in arr)])
 avgrows.sort(key=lambda r:float('inf') if r[2] is None else r[2])
 tab(['盆地','有效指标数','指标等权 nDTW','分指标值与覆盖期数'],[[r[0],str(r[1]),f'{r[2]:.3f}' if r[2] is not None else '—',r[3]] for r in avgrows])
 p('表中 nDTW 大表示对 DBA 形态模板的偏离程度大；不表示生态网络本身一定减少。对于覆盖期数不同的盆地，应优先比较相同 stage 的 nDTW。')
 h(2,'6. 采样分辨率敏感性与数据局限')
                                           
 countrows=[]
 for b,rec in ds['basins'].items():
  for st in stages:
   n=sum(r['stage']==st for r in rec)
   if 0<n<=4:countrows.append((b,st,n))
 if countrows: p('稀疏分期：'+ '；'.join(f'{b} {st}={n} 层' for b,st,n in countrows)+'。这些格子对 DBA 的贡献和其 nDTW 的解释均受采样稀疏性影响，单层格子不建议参与跨盆地综合评价；CSV 中另外提供了排除单层后的均值。')
 if ds['sensitivity_summary']:
  p('采样降采样敏感性：针对样本长度明显高于其余盆地的序列，分别随机保留原始层数的 50% 与 25%（每个 stage 尽量保留端点），每档重复 50 次，保持原始数据计算的 DBA 模板不变以隔离采样密度效应。')
  tab(['盆地','指标','保留比例','原 nDTW','降采样均值','2.5%–97.5%','均值差'],
      [[r[0],r[1],f'{r[2]:.0%}',f'{r[4]:.3f}',f'{r[5]:.3f}',f'{r[6]:.3f}–{r[7]:.3f}',f'{r[8]:+.3f}'] for r in ds['sensitivity_summary']])
 else:p('当前没有满足“层数达到其他盆地采样量中位数约两倍”的降采样对象。')
 if any('stage0'==s for s in stages):
  p('stage0 特别说明：本批次 stage0 仅少数盆地有记录，因此 stage0 模板是少数盆地共同构造，不能与具有完整盆地覆盖的 stage1–stage3 直接等同。')
 if any('stage4'==s for s in stages):
  p('stage4 特别说明：只有具备四阶段记录的部分盆地参与，stage1–stage4 的跨阶段平均中盆地构成随期变化。')
 p('共现网络边数与食物网营养联系数来自不同网络构建规则；应结合节点数、样本化石记录分辨率、网络阈值及食物网构建敏感性评价其变化。DTW 形态对齐是非线性对应，不能独立证明不同盆地的真实年龄相同。')
 h(2,'7. 输出与复现说明')
 p('所有图件均成对输出 300 dpi PNG 与无栅格内容嵌入的矢量 PDF；figures/ 中包含未重采样的地层时间主图、形态时间对齐图、统计残差箱线图、分期 nDTW 热图/柱状图、原始采样序号对模板节点的 Warping Path，以及逐盆地真实分期背景的核验图。')
 p('csv/ 中包含每期/全期 nDTW、盆地间和模板间多维矩阵、原始数据、全期 z-score、DBA 节点、warping paths、残差、采样敏感性结果、原始层数及原始指标分期变化百分比。')
 p('复现代码见本压缩包根目录下的 run_full_analysis.py 和 xls_biff_reader.py；原始工作簿的副本放在对应数据组根目录下。')
 mdfile=reports/'analysis_report.md';mdfile.write_text('\n'.join(md),encoding='utf-8')
 body_html='\n'.join(body).replace('<li>','<p class="bullet">• ').replace('</li>','</p>')
 csstext='''
 @page { size: A4; margin: 19mm 16mm 19mm 16mm;
   @bottom-center { content: counter(page) ' / ' counter(pages); color: #748091; font-size: 8pt; } }
 html,body {font-family:'Noto Sans CJK SC',sans-serif; color:#172437; font-size:9.4pt; line-height:1.53;}
 h1 {font-size:19pt;line-height:1.3;color:#17385B;border-bottom:2px solid #356C99;padding-bottom:12px;margin-bottom:12px;}
 h2 {font-size:12.5pt; color:#1E537B; margin-top:17px; border-bottom:1px solid #D7E3EB; padding-bottom:4px;break-after:avoid;}
 h3 {font-size:10.3pt;color:#285B72;margin-top:12px;break-after:avoid;}
 p {margin:6px 0 10px;} p.bullet {margin:3px 0 3px 12px;}
 table {width:100%;border-collapse:collapse;font-size:7.7pt;margin:7px 0 14px;break-inside:auto;table-layout:auto;}
 thead {display:table-header-group;}tr {break-inside:avoid;}
 th {background:#E7EEF5;color:#234364;text-align:left;font-weight:600;}
 th,td {border:0.5px solid #D9E2EB;padding:4px 5px;vertical-align:top;word-break:normal;}
 tr:nth-child(even) td {background:#F8FAFC;}
 '''
 htmltext='<!doctype html><html lang="zh"><head><meta charset="utf-8"><style>'+csstext+'</style></head><body>'+body_html+'</body></html>'
 htmlfile=reports/'analysis_report.html';htmlfile.write_text(htmltext,encoding='utf-8')
 try:
  from weasyprint import HTML
  HTML(string=htmltext,base_url=str(reports)).write_pdf(reports/'analysis_report.pdf')
 except Exception as e:
                                                                          
  import warnings;warnings.warn('HTML-to-PDF failed, fallback: '+repr(e))
  fig,ax=plt.subplots(figsize=(8.27,11.69));ax.axis('off')
  arr=[title,'','指标：'+', '.join(metrics),'盆地：'+', '.join(basins),'阶段：'+', '.join(stages),'','完整报告请阅读同目录 analysis_report.md。']
  ax.text(.06,.95,'\n'.join(arr),va='top',fontsize=13,transform=ax.transAxes)
  fig.savefig(reports/'analysis_report.pdf',bbox_inches='tight');plt.close(fig)


def main():
    package_root = Path(__file__).resolve().parents[2]
    input_path = package_root / "data" / "dataset_006.xls"
    if not input_path.is_file():
        raise FileNotFoundError(f"未找到DTW输入数据: {input_path}")

    basins, _ = load_book(input_path)
    record_count = sum(len(records) for records in basins.values())
    print(
        f"Running ecology six DTW analysis: {len(basins)} sheets; "
        f"{record_count} records",
        flush=True,
    )
    manifest = run_analysis(
        name="ecology_six",
        basins=basins,
        base=ROOT,
        description="Single-workbook DTW/DBA analysis for the ecology six dataset.",
        original=input_path,
    )
    print(
        f"COMPLETED ecology six: {manifest['figure_pairs']} PNG/PDF pairs, "
        f"{manifest['csv_count']} CSV, {manifest['validation_checks']} checks",
        flush=True,
    )


if __name__ == "__main__":
    main()
