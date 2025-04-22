import sys
import os

# This makes nested test directories a little more resiliant (and avoids all the __init__.py files).
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)