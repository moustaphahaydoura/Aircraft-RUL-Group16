"""Execute the completed notebook in review mode and preserve visible outputs."""
import sys
from pathlib import Path
import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    path=ROOT/'aircraft_rul_pipeline.ipynb'
    nb=nbformat.read(path,as_version=4)
    kernel=KernelManager(kernel_name='python3')
    kernel.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=kernel,timeout=600,resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbformat.write(nb,path)
    print('Notebook executed with all outputs and no errors.')
