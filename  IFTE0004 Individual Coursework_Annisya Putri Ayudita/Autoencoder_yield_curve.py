"""
Autoencoder-Based Three-Factor Model for the U.S. Treasury Yield Curve
Adapted Replication of Suimon et al. (2020)

Usage:  python replication.py
Data:   Place USdataYC.csv in the same directory.
"""
import os, sys, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

SEED=42; np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)
MAIN_START='1993-10-01'; MAIN_END='2019-12-31'
STRESS_START='2020-01-01'; STRESS_END='2023-12-31'
COLS_PRIMARY=['2Y','5Y','7Y','10Y','20Y']
COLS_EXTENDED=['3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','20Y']
MAT5=[2,5,7,10,20]; MAT9=[0.25,0.5,1,2,3,5,7,10,20]
RUN_FULL_TRADING_GRID=False
OUTPUT_DIR=Path('yield_curve_outputs'); OUTPUT_DIR.mkdir(exist_ok=True)
DATA_PATHS=['USdataYC.csv','/content/USdataYC.csv','/mnt/data/USdataYC.csv']
C={'p':'#2E5090','s':'#C0392B','t':'#27AE60','q':'#8E44AD','a1':'#E67E22','a2':'#16A085','bg':'#FAFAFA','tx':'#2C3E50'}
MC=['#2E5090','#C0392B','#27AE60','#E67E22','#8E44AD','#16A085','#D4AC0D','#5D6D7E','#A93226']
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':C['bg'],'axes.edgecolor':'#CCC','axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':11,'axes.grid':True,'grid.alpha':0.3,'grid.color':'#DDD','xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'legend.framealpha':0.9,'font.family':'sans-serif','lines.linewidth':1.5})

def _sa(ax,t='',xl='',yl=''):
    ax.set_title(t,pad=10);ax.set_xlabel(xl);ax.set_ylabel(yl)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))

def find_path():
    for p in DATA_PATHS:
        if Path(p).exists(): return p
    print("ERROR: USdataYC.csv not found."); sys.exit(1)
def load_data(path=None):
    if path is None: path=find_path()
    print(f"Loading: {path}")
    df=pd.read_csv(path,na_values=['','NA','N/A','.','null']);df.columns=[c.strip() for c in df.columns]
    df['Date']=pd.to_datetime(df['Date'],errors='coerce');df=df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    for c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    print(f"  {df.shape[0]} daily obs, {df.index.min().date()} to {df.index.max().date()}"); return df
def make_weekly(df,cols,start=None,end=None):
    w=df[cols].copy().resample('W-FRI').last().dropna()
    if start: w=w.loc[start:]
    if end: w=w.loc[:end]
    return w
def fit_scaler(m,s=None):
    sc=StandardScaler();Xm=sc.fit_transform(m.values);Xs=sc.transform(s.values) if s is not None else None; return sc,Xm,Xs
def calc_m(Xt,Xr,l=''):
    return {'Label':l,'RMSE (%)':np.sqrt(mean_squared_error(Xt,Xr)),'RMSE (bps)':np.sqrt(mean_squared_error(Xt,Xr))*100,'MAE (%)':mean_absolute_error(Xt,Xr),'R2':r2_score(Xt.flatten(),Xr.flatten())}
def run_pca(Xs,sc,do,nc=3):
    p=PCA(n_components=nc);Z=p.fit_transform(Xs);Xr=sc.inverse_transform(p.inverse_transform(Z))
    ld=pd.DataFrame(p.components_.T,index=do.columns,columns=[f'PC{i+1}' for i in range(nc)])
    fa=pd.DataFrame(Z,index=do.index,columns=[f'PC{i+1}' for i in range(nc)])
    return {'pca':p,'loadings':ld,'factors':fa,'recon':Xr,'metrics':calc_m(do.values,Xr),'explained_var':p.explained_variance_ratio_}
def pca_stress(pobj,sc,Xss,dso):
    Zs=pobj.transform(Xss);Xsr=sc.inverse_transform(pobj.inverse_transform(Zs));return calc_m(dso.values,Xsr,'PCA Stress')
def build_ae(dim,hn,seed=SEED):
    tf.random.set_seed(seed);inp=layers.Input(shape=(dim,))
    enc=layers.Dense(hn,activation='tanh',name='bottleneck',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(inp)
    dec=layers.Dense(dim,activation='linear',name='output',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1))(enc)
    ae=Model(inp,dec,name=f'AE_{hn}');encoder=Model(inp,enc);ae.compile(optimizer='adam',loss='mse');return ae,encoder
def run_ae(Xs,sc,do,hn=3,Xss=None,dso=None,ep=1000,bs=32,pat=50,seed=SEED):
    tf.keras.backend.clear_session();np.random.seed(seed);random.seed(seed);tf.random.set_seed(seed)
    ae,enc=build_ae(Xs.shape[1],hn,seed)
    h=ae.fit(Xs,Xs,epochs=ep,batch_size=bs,validation_split=0.2,shuffle=False,callbacks=[EarlyStopping(monitor='val_loss',patience=pat,restore_best_weights=True,min_delta=1e-7)],verbose=0)
    Xr=sc.inverse_transform(ae.predict(Xs,verbose=0));fa=pd.DataFrame(enc.predict(Xs,verbose=0),index=do.index,columns=[f'AE{hn}_{i+1}' for i in range(hn)])
    mt=calc_m(do.values,Xr,f'AE({hn}) Train');dw=ae.get_layer('output').get_weights()[0];db=ae.get_layer('output').get_weights()[1]
    ms=None;sf=None;srd=None
    if Xss is not None and dso is not None:
        sr=sc.inverse_transform(ae.predict(Xss,verbose=0));sf=pd.DataFrame(enc.predict(Xss,verbose=0),index=dso.index,columns=[f'AE{hn}_{i+1}' for i in range(hn)])
        srd=pd.DataFrame(sr,index=dso.index,columns=dso.columns);ms=calc_m(dso.values,sr,f'AE({hn}) Stress')
    return {'ae':ae,'encoder':enc,'history':h,'factors':fa,'stress_factors':sf,'recon':pd.DataFrame(Xr,index=do.index,columns=do.columns),'stress_recon':srd,'decoder_weights':dw,'decoder_bias':db,'train_metrics':mt,'stress_metrics':ms,'epochs_run':len(h.history['loss'])}
def run_trading(wdf,cols,ly=5,hw=4,hn=3,te=MAIN_END,seed=SEED):
    data=wdf[cols].dropna().copy();ted=pd.to_datetime(te);et=data.index[0]+pd.DateOffset(years=ly)
    aeg,trg,lg,sg,td=[],[],[],[],[];cae=None;csc=None;ly_=None;nm=len(cols)
    for i,d in enumerate(data.index):
        if d<et or d>ted: continue
        if i+hw>=len(data): break
        if cae is None or d.year!=ly_:
            te_=d-pd.Timedelta(days=1);ts_=te_-pd.DateOffset(years=ly);win=data.loc[ts_:te_].dropna()
            if len(win)<ly*40: continue
            tf.keras.backend.clear_session();np.random.seed(seed);random.seed(seed);tf.random.set_seed(seed)
            csc=StandardScaler();Xw=csc.fit_transform(win.values);cae,_=build_ae(nm,hn,seed)
            cae.fit(Xw,Xw,epochs=400,batch_size=32,validation_split=0.2,shuffle=False,callbacks=[EarlyStopping(monitor='val_loss',patience=30,restore_best_weights=True)],verbose=0)
            ly_=d.year
        yn=data.iloc[i:i+1].values;yns=csc.transform(yn);yr=csc.inverse_transform(cae.predict(yns,verbose=0))
        sig=np.where(yn>yr,1,-1).flatten();yf=data.iloc[i+hw].values;yc=yf-yn.flatten()
        aeg.append(sig*(-yc)*100);lg.append((-yc)*100);sg.append(yc*100)
        tsig=np.where(data.iloc[i].values-data.iloc[i-1].values<0,1,-1) if i>0 else np.ones(nm)
        trg.append(tsig*(-yc)*100);td.append(d)
    return {'dates':td,'gains':{'Autoencoder':np.array(aeg),'Trend-Follow':np.array(trg),'Always Long':np.array(lg),'Always Short':np.array(sg)},'cols':cols}
def trading_table(r):
    rows=[]
    for s,g in r['gains'].items():
        row={'Strategy':s}
        for i,c in enumerate(r['cols']): row[c]=g[:,i].mean()
        row['Average']=g.mean();rows.append(row)
    return pd.DataFrame(rows)

def plot_eda(dm,od):
    fig,ax=plt.subplots(1,2,figsize=(16,5.5))
    for j,c in enumerate(COLS_PRIMARY): ax[0].plot(dm.index,dm[c],label=c,color=MC[j],lw=0.9)
    _sa(ax[0],'U.S. Treasury Yields (1993-2019)','','Yield (%)');ax[0].legend(frameon=True);ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    mk=['o','s','D','^']
    for k,d in enumerate(['2000-06-30','2007-06-29','2012-06-29','2019-06-28']):
        idx=dm.index.get_indexer([pd.Timestamp(d)],method='nearest')[0]
        ax[1].plot(MAT5,dm.iloc[idx].values,f'{mk[k]}-',label=dm.index[idx].strftime('%b %Y'),color=MC[k],lw=1.8,ms=7)
    _sa(ax[1],'Selected Yield Curves','Maturity (Years)','Yield (%)');ax[1].set_xticks(MAT5);ax[1].legend(frameon=True)
    plt.tight_layout();plt.savefig(od/'fig_eda.png',dpi=200,bbox_inches='tight');plt.close();print("  Saved: fig_eda.png")

def plot_factors(pl,aer,od):
    fig,ax=plt.subplots(1,3,figsize=(18,5.5));cc=[C['p'],C['s'],C['t']]
    for i in range(3): ax[0].plot(MAT5,pl.iloc[:,i],'o-',label=f'PC{i+1}',color=cc[i],lw=2.2,ms=8,markeredgecolor='white',markeredgewidth=1)
    ax[0].axhline(0,color='#999',ls='--',lw=0.8);_sa(ax[0],'PCA Factor Loadings','Maturity (Years)','Loading');ax[0].set_xticks(MAT5);ax[0].legend(frameon=True)
    dw=aer[3]['decoder_weights']
    for j in range(3): ax[1].plot(MAT5,dw[j,:],'o-',label=f'Node {j+1}',color=cc[j],lw=2.2,ms=8,markeredgecolor='white',markeredgewidth=1)
    ax[1].axhline(0,color='#999',ls='--',lw=0.8);_sa(ax[1],'AE(3) Decoder Coefficients','Maturity (Years)','Weight');ax[1].set_xticks(MAT5);ax[1].legend(frameon=True)
    sty={2:('--',0.6),3:('-',1.0),4:(':',0.6)};ci=0
    for n in [2,3,4]:
        w=aer[n]['decoder_weights'];ls,al=sty[n]
        for j in range(n): ax[2].plot(MAT5,w[j,:],ls,label=f'{n}-node N{j+1}',color=MC[ci],lw=1.6,alpha=al);ci+=1
    ax[2].axhline(0,color='#999',ls='--',lw=0.8);_sa(ax[2],'Decoder: 2/3/4 Nodes','Maturity (Years)','Weight');ax[2].set_xticks(MAT5);ax[2].legend(fontsize=7,ncol=2,frameon=True)
    plt.tight_layout();plt.savefig(od/'fig_factor_analysis.png',dpi=200,bbox_inches='tight');plt.close();print("  Saved: fig_factor_analysis.png")

def plot_recon(dm,Xpr,a3r,od):
    fig,ax=plt.subplots(2,2,figsize=(14,10))
    for a,d in zip(ax.flatten(),['2000-06-30','2007-06-29','2012-06-29','2019-06-28']):
        i=dm.index.get_indexer([pd.Timestamp(d)],method='nearest')[0]
        a.plot(MAT5,dm.iloc[i].values,'ko-',label='Actual',lw=2.2,ms=9,markeredgecolor='white',markeredgewidth=1)
        a.plot(MAT5,Xpr[i],'s--',label='PCA(3)',color=C['p'],lw=1.8,ms=7,markeredgecolor='white',markeredgewidth=1)
        a.plot(MAT5,a3r.iloc[i].values,'^--',label='AE(3)',color=C['s'],lw=1.8,ms=7,markeredgecolor='white',markeredgewidth=1)
        _sa(a,dm.index[i].strftime('%d %B %Y'),'Maturity (Years)','Yield (%)');a.set_xticks(MAT5);a.legend(frameon=True,fontsize=8)
    plt.suptitle('Actual vs Reconstructed Yield Curves',fontsize=14,fontweight='bold',y=1.01);plt.tight_layout()
    plt.savefig(od/'fig_reconstruction.png',dpi=200,bbox_inches='tight');plt.close();print("  Saved: fig_reconstruction.png")

def plot_resid(dm,a3r,od):
    res=dm.values-a3r.values;rd=pd.DataFrame(res,index=dm.index,columns=COLS_PRIMARY)
    fig,ax=plt.subplots(2,1,figsize=(14,7.5))
    for j,c in enumerate(['10Y','20Y']): ax[0].plot(rd.index,rd[c],lw=0.8,alpha=0.85,label=c,color=MC[j])
    ax[0].axhline(0,color='#999',lw=0.8);ax[0].fill_between(rd.index,rd['10Y'],0,alpha=0.08,color=MC[0])
    _sa(ax[0],'AE(3) Reconstruction Residuals — Valuation Signal','','Residual (pp)');ax[0].legend(frameon=True);ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    bars=ax[1].bar(COLS_PRIMARY,rd.abs().mean().values,color=[MC[i] for i in range(5)],alpha=0.85,edgecolor='white',linewidth=1.2)
    for b,v in zip(bars,rd.abs().mean().values): ax[1].text(b.get_x()+b.get_width()/2,b.get_height()+0.0005,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
    _sa(ax[1],'Mean Absolute Residual by Maturity','','MAE (pp)');plt.tight_layout()
    plt.savefig(od/'fig_residuals.png',dpi=200,bbox_inches='tight');plt.close();print("  Saved: fig_residuals.png")

def plot_trading(tr,title,fn,od):
    fig,ax=plt.subplots(figsize=(13,5.5))
    ss={'Autoencoder':(C['p'],'-',2.2),'Trend-Follow':(C['s'],'--',1.5),'Always Long':(C['t'],'-.',1.2),'Always Short':(C['q'],':',1.2)}
    for s,g in tr['gains'].items():
        co,ls,lw=ss.get(s,('#999','-',1));cum=g.mean(axis=1).cumsum();ax.plot(tr['dates'][:len(cum)],cum,ls=ls,color=co,label=s,lw=lw)
    ax.axhline(0,color='#999',lw=0.8);_sa(ax,title,'Date','Cumulative Capital Gain (bp)');ax.legend(frameon=True,loc='upper left');ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout();plt.savefig(od/fn,dpi=200,bbox_inches='tight');plt.close();print(f"  Saved: {fn}")

def plot_stress(f3, sf3, od):
    fig, ax = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True)
    cc = [C['p'], C['s'], C['t']]
    for i, c in enumerate(f3.columns):
        ax[i].plot(f3.index, f3[c], color=cc[i], lw=0.8, label='Train (1993-2019)')
        if sf3 is not None:
            ax[i].plot(sf3.index, sf3.iloc[:, i], color='#E74C3C', lw=0.8, label='Stress (2020-2023)')
        ax[i].axvline(pd.Timestamp('2020-01-01'), color='#999', ls='--', lw=1, alpha=0.7)
        _sa(ax[i], '', '', f'Factor {i+1}')
        if i == 0:
            ax[i].legend(frameon=True, fontsize=8)
    ax[0].set_title('AE(3) Hidden Factors: Training vs Stress Period', pad=10, fontsize=13, fontweight='bold')
    ax[2].set_xlabel('Date')
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(od / 'fig_stress_factors.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_stress_factors.png")

def main():
    print("="*70+"\nAUTOENCODER YIELD CURVE REPLICATION\n"+"="*70)
    print("\n[1/12] Loading...");raw=load_data()
    print("\n[2/12] Weekly samples...");dm=make_weekly(raw,COLS_PRIMARY,MAIN_START,MAIN_END);ds=make_weekly(raw,COLS_PRIMARY,STRESS_START,STRESS_END);sc,Xm,Xs=fit_scaler(dm,ds)
    print(f"  Main: {len(dm)} obs | Stress: {len(ds)} obs")
    print("\n[3/12] EDA...");plot_eda(dm,OUTPUT_DIR)
    print("\n[4/12] PCA...");pr=run_pca(Xm,sc,dm);psm=pca_stress(pr['pca'],sc,Xs,ds)
    for i,v in enumerate(pr['explained_var']): print(f"  PC{i+1}: {v*100:.2f}% (cum: {pr['explained_var'][:i+1].sum()*100:.2f}%)")
    print(f"  Train: {pr['metrics']['RMSE (bps)']:.2f}bps | Stress: {psm['RMSE (bps)']:.2f}bps")
    print("\n[5/12] Autoencoders...");aer={}
    for n in [2,3,4]:
        print(f"  AE({n})...",end=" ",flush=True);aer[n]=run_ae(Xm,sc,dm,hn=n,Xss=Xs,dso=ds);r=aer[n]
        print(f"Ep:{r['epochs_run']} Train:{r['train_metrics']['RMSE (bps)']:.2f}bps Stress:{r['stress_metrics']['RMSE (bps)']:.2f}bps")
    a3=aer[3]
    print("\n[6/12] Factors...");px=pd.DataFrame({'Level':dm.mean(axis=1),'Slope':dm['20Y']-dm['2Y'],'Curvature':2*dm['10Y']-dm['2Y']-dm['20Y']},index=dm.index)
    co=a3['factors'].join(px).corr().loc[list(a3['factors'].columns),['Level','Slope','Curvature']];print(co.round(3).to_string())
    for f in co.index: b=co.loc[f].abs().idxmax();print(f"  {f}->{b} (r={co.loc[f,b]:.3f})")
    dwdf=pd.DataFrame(a3['decoder_weights'],index=[f'Node{i+1}' for i in range(3)],columns=COLS_PRIMARY);print("\n  Decoder weights:");print(dwdf.round(3).to_string())
    print("\n[7/12] Plots...");plot_factors(pr['loadings'],aer,OUTPUT_DIR);plot_recon(dm,pr['recon'],a3['recon'],OUTPUT_DIR);plot_resid(dm,a3['recon'],OUTPUT_DIR)
    print("\n[8/12] Table 1...");rows=[];rows.append({**pr['metrics'],'Model':'PCA(3)','Sample':'Train'});rows.append({**psm,'Model':'PCA(3)','Sample':'Stress'})
    for n in [2,3,4]:
        rows.append({**aer[n]['train_metrics'],'Model':f'AE({n})','Sample':'Train'})
        if aer[n]['stress_metrics']: rows.append({**aer[n]['stress_metrics'],'Model':f'AE({n})','Sample':'Stress'})
    dfr=pd.DataFrame(rows)[['Model','Sample','RMSE (%)','RMSE (bps)','R2']];print(dfr.round(4).to_string(index=False))
    print("\n[9/12] Trading...");fw=make_weekly(raw,COLS_PRIMARY,MAIN_START,MAIN_END);t51=run_trading(fw,COLS_PRIMARY,ly=5,hw=4);tr=trading_table(t51);print(tr.round(2).to_string(index=False))
    plot_trading(t51,'Cumulative Capital Gain: 5Y Learning, 1M Horizon','fig_trading.png',OUTPUT_DIR)
    if RUN_FULL_TRADING_GRID:
        for ly in [2,5,10]:
            for hw in [4,13]:
                if ly==5 and hw==4: continue
                k=f'{ly}Y_{"1M" if hw==4 else "3M"}';print(f"  {k}...",end=" ");r=run_trading(fw,COLS_PRIMARY,ly=ly,hw=hw);print(f"AE:{r['gains']['Autoencoder'].mean():.2f}bp")
    print("\n[10/12] Stress...");plot_stress(a3['factors'],a3['stress_factors'],OUTPUT_DIR)
    print(f"  {'':20s}{'Train':>12s}{'Stress':>12s}");print(f"  {'PCA RMSE(bps)':20s}{pr['metrics']['RMSE (bps)']:>12.2f}{psm['RMSE (bps)']:>12.2f}")
    print(f"  {'AE(3) RMSE(bps)':20s}{a3['train_metrics']['RMSE (bps)']:>12.2f}{a3['stress_metrics']['RMSE (bps)']:>12.2f}")
    print("\n[11/12] Extended...");dem=make_weekly(raw,COLS_EXTENDED,MAIN_START,MAIN_END);des=make_weekly(raw,COLS_EXTENDED,STRESS_START,STRESS_END);sce,Xem,Xes=fit_scaler(dem,des)
    per=run_pca(Xem,sce,dem);pesm=pca_stress(per['pca'],sce,Xes,des);aee=run_ae(Xem,sce,dem,hn=3,Xss=Xes,dso=des)
    epx=pd.DataFrame({'Level':dem.mean(axis=1),'Slope':dem['20Y']-dem['3M'],'Curvature':2*dem['10Y']-dem['3M']-dem['20Y']},index=dem.index)
    eco=aee['factors'].join(epx).corr().loc[list(aee['factors'].columns),['Level','Slope','Curvature']]
    print(f"  PCA cum: {per['explained_var'].sum()*100:.2f}%");print(f"  PCA T:{per['metrics']['RMSE (bps)']:.2f} S:{pesm['RMSE (bps)']:.2f}bps")
    print(f"  AE  T:{aee['train_metrics']['RMSE (bps)']:.2f} S:{aee['stress_metrics']['RMSE (bps)']:.2f}bps");print(eco.round(3).to_string())
    esr=[];esr.append({**per['metrics'],'Model':'PCA(3)','Sample':'Train'});esr.append({**pesm,'Model':'PCA(3)','Sample':'Stress'})
    esr.append({**aee['train_metrics'],'Model':'AE(3)','Sample':'Train'});esr.append({**aee['stress_metrics'],'Model':'AE(3)','Sample':'Stress'})
    dfe=pd.DataFrame(esr)[['Model','Sample','RMSE (%)','RMSE (bps)','R2']]
    comp=pd.DataFrame([{'Specification':'Primary (5 maturities)','N':5,'Maturities':', '.join(COLS_PRIMARY),'Short_end':'No','PCA_cum_var':f"{pr['explained_var'].sum()*100:.2f}%",'PCA_train_bps':round(pr['metrics']['RMSE (bps)'],2),'PCA_stress_bps':round(psm['RMSE (bps)'],2),'AE3_train_bps':round(a3['train_metrics']['RMSE (bps)'],2),'AE3_stress_bps':round(a3['stress_metrics']['RMSE (bps)'],2),'Note':'Paper-aligned'},{'Specification':'Extended (9 maturities)','N':9,'Maturities':', '.join(COLS_EXTENDED),'Short_end':'Yes','PCA_cum_var':f"{per['explained_var'].sum()*100:.2f}%",'PCA_train_bps':round(per['metrics']['RMSE (bps)'],2),'PCA_stress_bps':round(pesm['RMSE (bps)'],2),'AE3_train_bps':round(aee['train_metrics']['RMSE (bps)'],2),'AE3_stress_bps':round(aee['stress_metrics']['RMSE (bps)'],2),'Note':'Broader U.S. curve incl. short-end'}])
    print("\n  TABLE 3: Primary vs Extended");print(comp.to_string(index=False))
    print("\n[12/12] Export...");dfr.to_csv(OUTPUT_DIR/'table1_reconstruction.csv',index=False);co.to_csv(OUTPUT_DIR/'factor_correlations_ae3.csv')
    tr.to_csv(OUTPUT_DIR/'table2_trading_5y_1m.csv',index=False);dwdf.to_csv(OUTPUT_DIR/'decoder_weights_ae3.csv')
    dfe.to_csv(OUTPUT_DIR/'extended_specification_summary.csv',index=False);eco.to_csv(OUTPUT_DIR/'extended_factor_correlations_ae3.csv')
    comp.to_csv(OUTPUT_DIR/'table3_extended_comparison.csv',index=False)
    print(f"\n  Saved to: {OUTPUT_DIR.resolve()}")
    for f in sorted(OUTPUT_DIR.glob('*')): print(f"    {f.name}")
    print(f"\n{'='*70}\nCOMPLETE\n  Page 1: Summary + methodology\n  Page 2: TABLE 1 + fig_factor_analysis.png\n  Page 3: TABLE 2 + fig_trading.png + discussion\n  Appendix: TABLE 3, extended, stress, residuals\n{'='*70}")

if __name__=='__main__': main()
