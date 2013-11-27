import os
def listfile(path):
  for path, dirs, files in os.walk(path):
    print path
    for f in files:
      print f

def path_include_ext(path,sext):
  t = set()
  for path, dirs, files in os.walk(path):
    for fn in files:
      base,ext = os.path.splitext(fn)
      if ext == sext :
        print path
        t.add(path)
        break
  return t
