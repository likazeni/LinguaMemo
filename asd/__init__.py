import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
try:
    a = os.path.dirname(os.path.abspath(__file__))
    # os.listdir(a + '/' + 'frontend')
    os.listdir(a + '/' + 'backend')
except ModuleNotFoundError:
    raise FileNotFoundError("In directory not found package 'backend' or directory 'frontend'") from None