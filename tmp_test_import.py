import importlib.util, pathlib, sys
path=pathlib.Path('dashboard/app.py').resolve()
print('loading', path)
try:
    spec = importlib.util.spec_from_file_location('dashboard.app', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print('imported successfully')
except Exception:
    import traceback; traceback.print_exc(); sys.exit(1)
