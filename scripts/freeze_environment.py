"""Record the tested direct dependencies and the complete installed environment."""
from importlib.metadata import distributions, version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECT=['numpy','pandas','scikit-learn','matplotlib','torch','xgboost','joblib',
        'python-docx','pymupdf','reportlab','pillow','nbformat','nbclient','ipykernel','pytest']
if __name__=='__main__':
    lines=['# Tested with Python 3.14.8; CPU execution.']+[f'{p}=={version(p)}' for p in DIRECT]
    (ROOT/'requirements.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    locked=[]
    for dist in distributions():
        name=dist.metadata['Name']
        suffix='; sys_platform == "win32"' if name.lower() in ['pywin32','pywinpty'] else ''
        locked.append(f'{name}=={dist.version}{suffix}')
    (ROOT/'requirements-lock.txt').write_text('\n'.join(sorted(locked,key=str.lower))+'\n',encoding='utf-8')
    print(f'Recorded {len(DIRECT)} direct dependencies and {len(locked)} installed distributions.')
