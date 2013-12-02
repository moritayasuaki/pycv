import os
def listfile(path):
  for path, dirs, files in os.walk(path):
    print path
    for f in files:
      print f

def path_include_ext(path,sext,n=100):
  t = set()
  i = 0
  for path, dirs, files in os.walk(path):
    i += 1
    if (i == n) :
      break;
    for fn in files:
      base,ext = os.path.splitext(fn)
      if ext == sext :
        print path
        t.add(path)
        break
  return t
