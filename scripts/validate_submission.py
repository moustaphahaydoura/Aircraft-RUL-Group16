"""Check that the delivered documents, checkpoints, and numeric artifacts agree."""
import hashlib
import json
import sys
from pathlib import Path
import joblib
import nbformat
import numpy as np
import pandas as pd
import pymupdf
import torch
from docx import Document
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rul_pipeline import (CAP, FEATURES, WINDOW, LSTMPredictor, RULPredictor,
                          conformal_radius, load_frame, make_windows, metrics,
                          predict_bundle, seed_all)

def main():
    seed_all(42)
    manifest=json.loads((ROOT/'artifacts/frozen_development.json').read_text())
    test_summary=json.loads((ROOT/'artifacts/test_summary.json').read_text())
    for path,digest in [('src/rul_pipeline.py',manifest['source_sha256']),
                        ('train_FD001.txt',manifest['training_sha256']),
                        ('test_FD001.txt',test_summary['test_sha256']),
                        ('RUL_FD001.txt',test_summary['labels_sha256'])]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    splits=json.loads((ROOT/'artifacts/splits.json').read_text())
    assert set(splits)=={str(seed) for seed in manifest['seeds']}
    for split in splits.values():
        sets=[set(split[k]) for k in ['train','validation','calibration']]
        assert [len(s) for s in sets]==[60,20,20]
        assert len(set.union(*sets))==100
        assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
    p=pd.read_csv(ROOT/'artifacts/test_predictions.csv')
    m=pd.read_csv(ROOT/'artifacts/test_metrics.csv').set_index('model')
    tuning=pd.read_csv(ROOT/'artifacts/tuning_results.csv')
    assert tuning.groupby('seed').size().to_dict()=={7:8,21:8,42:40}
    assert set(m.index)==set(manifest['selected'])
    assert p.unit_id.tolist()==list(range(1,101))
    np.testing.assert_array_equal(p.true_rul,np.loadtxt(ROOT/'RUL_FD001.txt'))
    y=p.true_rul.to_numpy()
    models=joblib.load(ROOT/'models/development_models.joblib')
    scaler=joblib.load(ROOT/'models/scaler.joblib')
    x,_,meta=make_windows(load_frame(ROOT/'test_FD001.txt'),scaler,'test')
    pd.testing.assert_frame_equal(meta,p[meta.columns])
    assert set(models)==set(m.index)
    standalone_count=native_count=0
    for name,info in models.items():
        bundle=info['bundle']
        reloaded=predict_bundle(bundle,x)
        # CSV uses shortest round-trippable text for each predictor's original dtype.
        saved_predictions=p[name].to_numpy(dtype=reloaded.dtype)
        np.testing.assert_allclose(reloaded,saved_predictions,rtol=1e-6,atol=1e-5)
        actual=metrics(y,saved_predictions)
        for key,value in actual.items():
            np.testing.assert_allclose(value,m.loc[name,key],rtol=1e-10,atol=1e-10)
        assert p[name].between(0,CAP).all()
        frozen=manifest['selected'][name]
        radius=conformal_radius(np.abs(frozen['calibration_residuals']))
        lo,hi=(p[name+suffix].to_numpy(dtype=reloaded.dtype) for suffix in ['__lower90','__upper90'])
        np.testing.assert_allclose(radius,frozen['radius90'])
        np.testing.assert_allclose(lo,np.maximum(0,saved_predictions-radius),atol=1e-5)
        np.testing.assert_allclose(hi,saved_predictions+radius,atol=1e-5)
        np.testing.assert_allclose(m.loc[name,'coverage90'],((y>=lo)&(y<=hi)).mean())
        np.testing.assert_allclose(m.loc[name,'mean_width90'],(hi-lo).mean())
        np.testing.assert_allclose(m.loc[name,'capped_rmse'],metrics(np.minimum(y,CAP),saved_predictions)['rmse'])
        filename=ROOT/'models'/name.lower().replace(' ','_')
        if bundle['kind'] in ('cnn','lstm'):
            state=torch.load(filename.with_suffix('.pt'),map_location='cpu',weights_only=True)
            assert state['features']==FEATURES and state['window']==WINDOW and state['cap']==CAP
            width=state['config']['width']
            net=RULPredictor(len(FEATURES),width=width,window=WINDOW) if state['kind']=='cnn' else LSTMPredictor(len(FEATURES),width)
            net.load_state_dict(state['state_dict'])
            saved={**bundle,'model':net}
        elif bundle['kind']=='constant':
            continue
        else:
            saved=joblib.load(filename.with_suffix('.joblib'))
        np.testing.assert_allclose(predict_bundle(saved,x),p[name],rtol=1e-6,atol=1e-5)
        standalone_count+=1
        if bundle['kind']=='xgb':
            native=XGBRegressor(n_jobs=manifest['threads'])
            native.load_model(filename.with_suffix('.ubj'))
            np.testing.assert_allclose(predict_bundle({**bundle,'model':native},x),p[name],rtol=1e-6,atol=1e-5)
            native_count+=1
    selected=min(manifest['selected'],key=lambda n:manifest['selected'][n]['validation']['rmse'])
    assert selected==manifest['selected_model']
    inference=pd.read_csv(ROOT/'artifacts/inference_predictions.csv')
    pd.testing.assert_frame_equal(inference[meta.columns],meta)
    assert inference.model.eq(selected).all()
    np.testing.assert_allclose(inference[['predicted_rul','lower90','upper90']],
                               p[[selected,selected+'__lower90',selected+'__upper90']])
    cal=pd.read_csv(ROOT/'artifacts/calibration_endpoints.csv')
    assert len(cal)==20 and cal.unit_id.nunique()==20
    val=pd.read_csv(ROOT/'artifacts/validation_endpoints.csv')
    assert len(val)==60 and val.unit_id.nunique()==20
    primary=splits[str(manifest['primary_seed'])]
    assert set(val.unit_id)==set(primary['validation']) and set(cal.unit_id)==set(primary['calibration'])
    notebook=nbformat.read(ROOT/'aircraft_rul_pipeline.ipynb',as_version=4)
    nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert [c.execution_count for c in code]==list(range(1,len(code)+1))
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    assert sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)==8
    doc=pymupdf.open(ROOT/'Final_Report.pdf')
    text='\n'.join(page.get_text() for page in doc)
    assert len(doc)>=8
    for section in ['Abstract','Introduction','Dataset','Models','Results','Conclusion','References']:
        assert section in text,section
    for value in m.rmse:
        assert f'{value:.2f}' in text
    assert all(glyph in text for glyph in ['Σ','≥','→','²'])
    assert len(doc.get_page_images(0))==1
    assert sum(len(page.get_images()) for page in doc)==9
    assert all(len(page.get_text())>350 for page in doc)
    word=Document(ROOT/'Final_Report.docx')
    blocks=json.loads((ROOT/'artifacts/report_content.json').read_text(encoding='utf-8'))
    assert [para.text for para in word.paragraphs if para.text.strip()]==[
        'GitHub repository\n'+block[1] if block[0]=='link' else block[-1]
        for block in blocks if block[0] in ('title','h','sub','p','fig','link')]
    repo_file=ROOT/'repository_url.txt'
    if repo_file.exists():
        url=repo_file.read_text(encoding='utf-8').strip()
        assert url in text and any(link.get('uri')==url for page in doc for link in page.get_links())
        assert any(link.url==url for para in word.paragraphs for link in para.hyperlinks)
    assert len(word.inline_shapes)==9
    assert [[[cell.text for cell in row.cells] for row in table.rows] for table in word.tables]==[
        [[str(value) for value in row] for row in [block[1],*block[2]]] for block in blocks if block[0]=='table']
    result=dict(status='passed',models_checked=len(models),test_engines=100,validation_endpoints=60,
                calibration_engines=20,notebook_code_cells=len(code),report_pages=len(doc),
                figure_png_count=len(list((ROOT/'figures').glob('*.png'))),
                primary_candidates=40,sensitivity_refits=16,standalone_checkpoints=standalone_count,
                native_xgboost_checkpoints=native_count,docx_tables=len(word.tables),docx_figures=len(word.inline_shapes),
                label_free_inference_rows=len(inference))
    (ROOT/'artifacts/submission_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
