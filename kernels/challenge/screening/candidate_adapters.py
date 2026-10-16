import importlib
def load_adapter(c):
 return importlib.import_module("screening.adapter_"+c["id"]).Adapter(c)
