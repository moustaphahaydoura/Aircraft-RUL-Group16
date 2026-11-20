"""Publication figures derived exclusively from saved predictions and run logs."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rul_pipeline import load_frame, FEATURES

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.titleweight':'bold', 'figure.facecolor':'white',
                     'axes.labelcolor':'#24384b', 'text.color':'#172d40', 'savefig.bbox':'tight'})
BLUE, TEAL, ORANGE, RED = '#245b86', '#148579', '#dd963d', '#b64c55'


def save(fig, name):
    fig.savefig(ROOT/f'figures/{name}.png', dpi=200)
    fig.savefig(ROOT/f'figures/{name}.pdf')
    plt.close(fig)


def main(include_poster=False):
    (ROOT/'figures').mkdir(exist_ok=True)
    m = pd.read_csv(ROOT/'artifacts/test_metrics.csv').set_index('model')
    pred = pd.read_csv(ROOT/'artifacts/test_predictions.csv')
    frozen = json.loads((ROOT/'artifacts/frozen_development.json').read_text())
    chosen = frozen['selected_model']
    train = load_frame(ROOT/'train_FD001.txt')
    test = load_frame(ROOT/'test_FD001.txt')
    # Workflow remains legible at report-column size.
    fig, ax = plt.subplots(figsize=(10.4,3.2))
    ax.set(xlim=(0,10.4), ylim=(0,3.2)); ax.axis('off')
    boxes = [(0,.85,2.1,1.5,'100 development\nengines\n\n60 fit · 20 validate\n20 calibrate',BLUE),
             (2.7,.85,2.1,1.5,'Training only\n\nScale and window\nFit model candidates',TEAL),
             (5.4,.85,2.1,1.5,'Freeze decisions\n\nSelect on validation\nSet interval widths',BLUE),
             (8.1,.85,2.1,1.5,'100 test engines\n\nOne endpoint each\nFinal evaluation',TEAL)]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=color,edgecolor='none'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',color='white',fontsize=10)
    for x in [2.15,4.85,7.55]:
        ax.annotate('',xy=(x+.5,1.6),xytext=(x,1.6),arrowprops={'arrowstyle':'->','color':BLUE,'lw':2})
    ax.text(5.2,.28,'Engine identities are disjoint; no official test target is used for model selection.',ha='center',fontsize=10)
    save(fig,'workflow')
    fig, axs = plt.subplots(1,3,figsize=(11.3,3.1),layout='constrained')
    axs[0].hist(train.groupby('unit_id').size(), bins=14,color=BLUE,alpha=.8,label='Training lifetimes')
    axs[0].hist(test.groupby('unit_id').size(), bins=14,color=ORANGE,alpha=.6,label='Observed test histories')
    axs[0].set(xlabel='Cycles',ylabel='Engines',title='Lifetime and observation length'); axs[0].legend(fontsize=7)
    unit=train[train.unit_id==1]
    raw=len(unit)-unit.time_cycles
    axs[1].plot(unit.time_cycles,raw,color=BLUE,label='Raw RUL')
    axs[1].plot(unit.time_cycles,np.minimum(raw,125),color=TEAL,label='Training target (cap 125)')
    axs[1].set(xlabel='Cycle (training engine 1)',ylabel='Remaining cycles',title='Target definition');axs[1].legend(fontsize=7)
    for uid in [1,2,3]:
        g=train[train.unit_id==uid]
        axs[2].plot(g.time_cycles,g.sensor_11.rolling(15,min_periods=1).mean(),label=f'Engine {uid}')
    axs[2].set(xlabel='Cycle',ylabel='Sensor 11 (native units)',title='Different degradation trajectories');axs[2].legend(fontsize=7)
    save(fig,'dataset')
    names=['Mean baseline','Linear history','RF instant','RF history','XGB instant','XGB mean_std','XGB history','CNN MSE','CNN asymmetric','LSTM MSE']
    fig,axs=plt.subplots(1,2,figsize=(10.6,4.0),layout='constrained')
    colors=[TEAL if n==chosen else BLUE for n in names]
    axs[0].barh(names,m.loc[names,'rmse'],color=colors)
    axs[0].invert_yaxis();axs[0].set(xlabel='Raw test RMSE (cycles); lower is better',title='Accuracy at 100 test endpoints')
    for i,n in enumerate(names):axs[0].text(m.loc[n,'rmse']+.3,i,f'{m.loc[n,"rmse"]:.1f}',va='center',fontsize=8)
    axs[0].set_xlim(0,m.loc[names,'rmse'].max()*1.2)
    axs[1].barh(names,m.loc[names,'phm'],color=colors)
    axs[1].invert_yaxis();axs[1].set(xscale='log',xlabel='PHM score (log scale); lower is better',title='Overestimation carries a larger penalty')
    save(fig,'model_comparison')
    fig,axs=plt.subplots(1,2,figsize=(10.3,3.3),layout='constrained')
    names=['XGB instant','XGB mean_std','XGB history','XGB sensors only','XGB top6 sensors']
    xx=np.arange(len(names)); width=.36
    axs[0].bar(xx-width/2,m.loc[names,'validation_rmse'],width,label='Validation',color=BLUE)
    axs[0].bar(xx+width/2,m.loc[names,'rmse'],width,label='Test',color=TEAL)
    axs[0].set_xticks(xx,['Current','+ Mean/std','+ Slope','14 sensors','Top 6'],rotation=20)
    axs[0].set(ylabel='Raw RMSE (cycles)',title='History and sensor ablations');axs[0].legend(fontsize=8)
    imp=pd.read_csv(ROOT/'artifacts/feature_importance.csv').head(8).iloc[::-1]
    axs[1].barh(imp.feature,imp.training_gain_importance,color=BLUE)
    axs[1].set(xlabel='Summed training gain importance',title='Which channels do trees use?')
    save(fig,'ablations')
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.5),layout='constrained')
    for ax,name in zip(axs,['XGB history','CNN MSE','CNN asymmetric']):
        d=pred[name]-pred.true_rul
        ax.scatter(pred.true_rul,pred[name],c=np.where(d>0,RED,BLUE),s=21,alpha=.75,edgecolors='white',linewidths=.25)
        ax.plot([0,150],[0,150],color='gray',ls='--',lw=1)
        ax.axhline(125,color=ORANGE,ls=':',lw=1)
        ax.set(xlabel='True RUL (cycles)',ylabel='Predicted RUL',title=name,xlim=(0,155),ylim=(0,155),aspect='equal')
    save(fig,'failure_scatter')
    stage=pd.read_csv(ROOT/'artifacts/stage_metrics.csv')
    fig,axs=plt.subplots(1,2,figsize=(10.3,3.1),layout='constrained')
    for name,color in [('XGB history',BLUE),('CNN MSE',TEAL),('CNN asymmetric',ORANGE),('LSTM MSE',RED)]:
        sub=stage[stage.model==name]
        axs[0].plot(sub.stage,sub.rmse,'o-',label=name,color=color)
        axs[1].plot(sub.stage,sub.bias,'o-',label=name,color=color)
    axs[0].set(xlabel='True RUL group (cycles)',ylabel='RMSE (cycles)',title='Where accuracy breaks down');axs[0].legend(fontsize=7)
    axs[1].axhline(0,color='gray',ls='--',lw=1)
    axs[1].set(xlabel='True RUL group (cycles)',ylabel='Mean prediction minus truth',title='Positive bias means overestimation')
    save(fig,'failure_stages')
    order=np.argsort(pred.true_rul.to_numpy())
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.4),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
    xx=np.arange(100)
    axs[0].fill_between(xx,pred[chosen+'__lower90'].to_numpy()[order],pred[chosen+'__upper90'].to_numpy()[order],alpha=.22,color=BLUE,label='Nominal 90% interval')
    axs[0].scatter(xx,pred.true_rul.to_numpy()[order],s=10,color='black',label='True RUL')
    axs[0].plot(xx,pred[chosen].to_numpy()[order],lw=1,color=TEAL,label='Prediction')
    axs[0].set(xlabel='Test engines sorted by true RUL',ylabel='Cycles',title=f'Intervals: {chosen}');axs[0].legend(fontsize=7)
    sub=stage[stage.model==chosen]
    axs[1].bar(sub.stage,sub.coverage90*100,color=BLUE)
    axs[1].axhline(90,color=ORANGE,ls='--',label='Nominal 90%')
    axs[1].set(ylim=(0,105),ylabel='Empirical coverage (%)',xlabel='True RUL group',title='Coverage is not uniform');axs[1].legend(fontsize=8)
    save(fig,'uncertainty')
    sens=pd.read_csv(ROOT/'artifacts/split_sensitivity.csv')
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.3),layout='constrained')
    names=['XGB instant','XGB history','RF history','CNN MSE','CNN asymmetric','LSTM MSE']
    for name,color in zip(names,[BLUE,TEAL,ORANGE,RED,'#805ba7','#657a55']):
        axs[0].scatter(m.loc[name,'fit_seconds'],m.loc[name,'rmse'],s=55,color=color,label=name)
        sub=sens[sens.model==name]
        axs[1].plot(sub.seed.astype(str),sub.rmse,'o-',color=color,label=name)
    axs[0].set(xscale='log',xlabel='Selected fit time (seconds; log scale)',ylabel='Test RMSE',title='Training cost versus test accuracy');axs[0].legend(fontsize=7)
    axs[1].set(xlabel='Development split seed',ylabel='Validation raw RMSE',title='Engine-split sensitivity');axs[1].legend(fontsize=7)
    save(fig,'complexity_sensitivity')
    hist=json.loads((ROOT/'artifacts/training_history.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.1),layout='constrained')
    for name in ['CNN MSE','CNN asymmetric','LSTM MSE']:
        candidates={k:v for k,v in hist.items() if k.startswith(name+'_')}
        criterion='phm' if name=='CNN asymmetric' else 'rmse'
        best=min(candidates,key=lambda k:min(r[criterion] for r in candidates[k]))
        rows=pd.DataFrame(candidates[best])
        axs[0].plot(rows.epoch,rows.loss,label=name)
        axs[1].plot(rows.epoch,rows.rmse,label=name)
    axs[0].set(xlabel='Epoch',ylabel='Normalized training loss',title='Selected neural candidate histories');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Epoch',ylabel='Validation RMSE (cycles)',title='Checkpoint selection uses validation');axs[1].legend(fontsize=8)
    save(fig,'learning_curves')
    if include_poster:
        make_poster_panels(m, pred, chosen, order)
    # Compatibility with original output filename, now backed by saved predictions.
    import shutil
    shutil.copyfile(ROOT/'figures/failure_scatter.png',ROOT/'rul_failure_analysis.png')
    print('Created nine report figures in PNG and vector PDF formats.')


def make_poster_panels(m, pred, chosen, order):
    with plt.rc_context({'font.size':14, 'axes.titlesize':16, 'axes.labelsize':14}):
        names=['RF history','XGB instant','XGB history','CNN MSE','CNN asymmetric','LSTM MSE']
        if chosen not in names:names.append(chosen)
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        ax.barh(names,m.loc[names,'rmse'],color=[TEAL if n==chosen else BLUE for n in names])
        ax.invert_yaxis();ax.set(xlabel='Test RMSE (cycles)',title='Equal test endpoints; lower is better')
        for i,n in enumerate(names):ax.text(m.loc[n,'rmse']+.25,i,f'{m.loc[n,"rmse"]:.2f}',va='center',fontsize=12)
        ax.set_xlim(0,m.loc[names,'rmse'].max()*1.25)
        save(fig,'poster_models')
        fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
        names=['XGB instant','XGB mean_std','XGB history','XGB sensors only','XGB top6 sensors']
        ax.barh(['Current','+ Mean / std','+ Trend','14 sensors','Top 6 sensors'],m.loc[names,'rmse'],color=[BLUE,BLUE,TEAL,BLUE,BLUE])
        ax.invert_yaxis();ax.set(xlabel='Test RMSE (cycles)',title='Historical-feature and sensor ablations')
        for i,n in enumerate(names):ax.text(m.loc[n,'rmse']+.2,i,f'{m.loc[n,"rmse"]:.2f}',va='center',fontsize=12)
        ax.set_xlim(0,m.loc[names,'rmse'].max()*1.25)
        save(fig,'poster_ablations')
        fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
        d=pred[chosen]-pred.true_rul
        ax.scatter(pred.true_rul,pred[chosen],c=np.where(d>0,RED,BLUE),s=35,alpha=.8)
        ax.plot([0,150],[0,150],color='gray',ls='--',lw=1)
        ax.axhline(125,color=ORANGE,ls=':',lw=1)
        ax.set(xlabel='True RUL (cycles)',ylabel='Predicted RUL',title=f'Failure analysis: {chosen}',xlim=(0,155),ylim=(0,155))
        save(fig,'poster_failures')
        fig,ax=plt.subplots(figsize=(7,3.2),layout='constrained')
        xx=np.arange(100)
        ax.fill_between(xx,pred[chosen+'__lower90'].to_numpy()[order],pred[chosen+'__upper90'].to_numpy()[order],color=BLUE,alpha=.2,label='90% interval')
        ax.scatter(xx,pred.true_rul.to_numpy()[order],s=16,color='black',label='Truth')
        ax.plot(xx,pred[chosen].to_numpy()[order],color=TEAL,lw=1,label='Prediction')
        ax.set(xlabel='Test engines sorted by true RUL',ylabel='Cycles',title='How informative are the intervals?');ax.legend(fontsize=10,ncol=3,loc='upper left')
        save(fig,'poster_intervals')
        fig,ax=plt.subplots(figsize=(7,3.4),layout='constrained')
        for name,color in [('XGB history',BLUE),('RF history',ORANGE),('CNN MSE',TEAL),('CNN asymmetric',RED),('LSTM MSE','#805ba7')]:
            ax.scatter(m.loc[name,'fit_seconds'],m.loc[name,'rmse'],s=70,color=color,label=name)
        ax.set(xscale='log',xlabel='Selected fit time (s; log scale)',ylabel='Test RMSE',title='Accuracy must justify its cost')
        ax.legend(fontsize=10,ncol=2)
        save(fig,'poster_cost')
    print('Created five poster panels in PNG and vector PDF formats.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poster', action='store_true', help='Also generate the poster panels')
    main(include_poster=parser.parse_args().poster)
