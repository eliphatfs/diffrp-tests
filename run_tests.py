import os
import sys

if len(sys.argv) == 2:
    assert 0 == os.system(f"python -m unittest -v {sys.argv[-1]}")
elif len(sys.argv) == 1:
    assert 0 == os.system(f"coverage run --source=diffrp --concurrency=thread -m unittest discover -v")
    assert 0 == os.system(f"coverage html")
else:
    raise ValueError("unknown usage")
