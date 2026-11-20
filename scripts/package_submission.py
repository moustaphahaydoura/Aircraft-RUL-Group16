"""Package the completed deliverables, reproducible source, data, and fitted models."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    names=['Final_Report.pdf','Final_Report.docx','aircraft_rul_pipeline.ipynb',
           'README.md','requirements.txt','requirements-lock.txt','.gitignore',
           'train_FD001.txt','test_FD001.txt','RUL_FD001.txt']
    if (ROOT/'repository_url.txt').exists():names.append('repository_url.txt')
    paths=[ROOT/n for n in names]
    for folder in ['src','scripts','tests','artifacts','models','figures']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ['.log','.pyc'] and not p.name.startswith('poster_'))
    paths=sorted(set(paths))
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with ZipFile(ROOT/'Final_Submission.zip','w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('SHA256SUMS.json',json.dumps(manifest,indent=2))
    with ZipFile(ROOT/'Final_Submission.zip') as z:
        assert z.testzip() is None
        archived=json.loads(z.read('SHA256SUMS.json'))
        assert archived==manifest
        assert sorted(z.namelist())==sorted([*manifest,'SHA256SUMS.json'])
        for name,digest in archived.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    print(f'Packaged {len(paths)} files plus SHA256SUMS.json. Archive size: '
          f'{(ROOT/"Final_Submission.zip").stat().st_size/1024**2:.1f} MiB. '
          'ZIP integrity and every archived SHA-256 verified.')
