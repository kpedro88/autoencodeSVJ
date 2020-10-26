from module.SignalElement import SignalElement

class DataHolder(object):
    def __init__(self, **kwargs):
        
        self.KEYS = {}
        names = sorted(kwargs.keys())
        
        for name in names:
            path = kwargs[name]
            if name[0].isdigit():
                name = 'Zprime_' + name
            name = name.replace('.', '')
            # print('loading {} from path \'{}\'...'.format(name, path))
            setattr(self, name, SignalElement(path, name))
            self.KEYS[name] = getattr(self, name)
            self.KEYS[name] = getattr(self, name)
    
    def load(self, hlf=True, eflow=True, hlf_to_drop=['Energy', 'Flavor']):
        for key, value in list(self.KEYS.items()):
            value._load(hlf, eflow, hlf_to_drop)