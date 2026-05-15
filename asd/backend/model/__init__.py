import os
try:
    a = os.path.dirname(os.path.abspath(__file__))
    os.listdir(a + '/' + 'checkpoint_515')
except ModuleNotFoundError:
    raise FileNotFoundError("In directory not found package 'backend' or directory 'frontend'") from None
