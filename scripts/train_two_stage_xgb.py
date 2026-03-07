#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DB_PATH = Path('/app/data/ml_training.db')
OUT = Path('/app/data/ml/artifacts/two_stage_xgb_metrics.json')
SEED = 42
PLANET = 'KNOWN_PLANET'
BASE = [
    'period','depth','snr','duration','secondary_depth','odd_even_sigma',
    'sinusoid_better','sectors_ratio','tmag','teff','star_radius'
]

LABEL2ID = {'FALSE_POSITIVE':0,'ECLIPSING_BINARY':1,'NOISE':2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}


def safe_div(a,b,default=0.0):
    out=np.full_like(a,default,dtype=np.float64)
    m=np.isfinite(a)&np.isfinite(b)&(np.abs(b)>1e-12)
    out[m]=a[m]/b[m]
    return out


def build_features(rows):
    base={c:np.array([float(r[c]) if r[c] is not None else np.nan for r in rows],dtype=np.float64) for c in BASE}
    period=base['period']; depth=base['depth']; snr=base['snr']; duration=base['duration']; sec=base['secondary_depth']; odd=base['odd_even_sigma']; sectors=base['sectors_ratio']; teff=base['teff']; radius=base['star_radius']
    eng=[
        safe_div(duration, period),
        safe_div(sec, depth),
        depth*snr,
        np.log1p(np.clip(snr,0,None)),
        safe_div(np.ones_like(odd),1.0+np.abs(odd)),
        sectors*np.log1p(np.clip(snr,0,None)),
        teff*radius,
    ]
    X=np.column_stack([base[c] for c in BASE]+eng).astype(np.float64)
    return X


def clip_by_train(Xtr, Xva, Xte):
    lo=np.nanquantile(Xtr,0.005,axis=0); hi=np.nanquantile(Xtr,0.995,axis=0)
    return np.clip(Xtr,lo,hi), np.clip(Xva,lo,hi), np.clip(Xte,lo,hi)


def split(X,y,g):
    gss1=GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=SEED)
    tr,tmp=next(gss1.split(X,y,groups=g))
    Xtr,ytr,gtr=X[tr],y[tr],g[tr]
    Xtmp,ytmp,gtmp=X[tmp],y[tmp],g[tmp]
    gss2=GroupShuffleSplit(n_splits=1,test_size=0.50,random_state=SEED)
    va,te=next(gss2.split(Xtmp,ytmp,groups=gtmp))
    return Xtr,ytr,Xtmp[va],ytmp[va],Xtmp[te],ytmp[te]


def train_stage_a(Xtr,ytr):
    ybin=(ytr==PLANET).astype(int)
    # oversample planet x4
    m=ytr==PLANET
    Xp,yp=Xtr[m],ytr[m]
    Xaug=np.concatenate([Xtr,Xp,Xp,Xp],axis=0)
    yaug=np.concatenate([ytr,yp,yp,yp],axis=0)
    yaug_bin=(yaug==PLANET).astype(int)

    w=compute_sample_weight(class_weight='balanced',y=yaug_bin)
    model=XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(Xaug,yaug_bin,sample_weight=w)
    return model


def train_stage_b(Xtr,ytr):
    m=ytr!=PLANET
    Xnp,ynp=Xtr[m],ytr[m]
    yn=np.array([LABEL2ID[v] for v in ynp],dtype=int)
    w=compute_sample_weight(class_weight='balanced',y=yn)
    model=XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(Xnp,yn,sample_weight=w)
    return model


def predict_two_stage(a,b,X,th):
    p=a.predict_proba(X)[:,1]
    is_planet=p>=th
    out=np.empty(len(X),dtype=object)
    out[is_planet]=PLANET
    if np.any(~is_planet):
        ids=b.predict(X[~is_planet]).astype(int)
        out[~is_planet]=[ID2LABEL[int(i)] for i in ids]
    return out


def tune_threshold(a,b,Xv,yv):
    best=None
    for th in np.linspace(0.05,0.30,26):
        pred=predict_two_stage(a,b,Xv,th)
        yb=(yv==PLANET).astype(int); hb=(pred==PLANET).astype(int)
        p,r,f,_=precision_recall_fscore_support(yb,hb,average='binary',zero_division=0)
        # strong preference for recall>=0.82, then F1
        score = (1 if r>=0.82 else 0, f, r, p)
        if best is None or score>best['score']:
            best={'threshold':float(th),'precision':float(p),'recall':float(r),'f1':float(f),'score':score}
    return best


def main():
    conn=sqlite3.connect(str(DB_PATH)); conn.row_factory=sqlite3.Row
    rows=conn.execute(f"SELECT tic_id,label,{','.join(BASE)} FROM training_data").fetchall(); conn.close()
    X=build_features(rows)
    y=np.array([r['label'] for r in rows],dtype=object)
    g=np.array([int(r['tic_id']) for r in rows],dtype=np.int64)

    Xtr,ytr,Xv,yv,Xte,yte=split(X,y,g)
    Xtr,Xv,Xte=clip_by_train(Xtr,Xv,Xte)

    a=train_stage_a(Xtr,ytr)
    b=train_stage_b(Xtr,ytr)
    th=tune_threshold(a,b,Xv,yv)

    yv_pred=predict_two_stage(a,b,Xv,th['threshold'])
    yte_pred=predict_two_stage(a,b,Xte,th['threshold'])

    m={
        'model':'two_stage_xgboost',
        'threshold':{k:v for k,v in th.items() if k!='score'},
        'val_macro_f1':float(f1_score(yv,yv_pred,average='macro')),
        'test_macro_f1':float(f1_score(yte,yte_pred,average='macro')),
        'val_report':classification_report(yv,yv_pred,output_dict=True,zero_division=0),
        'test_report':classification_report(yte,yte_pred,output_dict=True,zero_division=0),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(m,indent=2),encoding='utf-8')

    kp=m['test_report'][PLANET]
    print('=== Two-stage XGBoost done ===')
    print('threshold',m['threshold'])
    print('test_macro_f1',round(m['test_macro_f1'],4))
    print('planet_precision',round(kp['precision'],4))
    print('planet_recall',round(kp['recall'],4))
    print('planet_f1',round(kp['f1-score'],4))


if __name__=='__main__':
    main()
