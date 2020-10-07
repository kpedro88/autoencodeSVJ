from module.signalElement import signal_element

class data_holder(object):
    def __init__(self, **kwargs):
        self.KEYS = {}
        names = sorted(kwargs.keys())
        
        for name in names:
            path = kwargs[name]
            if name[0].isdigit():
                name = 'Zprime_' + name
            name = name.replace('.', '')
            # print('loading {} from path \'{}\'...'.format(name, path))
            setattr(self, name, signal_element(path, name))
            self.KEYS[name] = getattr(self, name)
        
        print(('found {} datasets'.format(len(names))))
    
    def load(self, hlf=True, eflow=True, hlf_to_drop=['Energy', 'Flavor']):
        for k, v in list(self.KEYS.items()):
            v._load(hlf, eflow, hlf_to_drop)
    
    def add_attribute(self, name, function):
        for k, v in list(self.KEYS.items()):
            v._add_attribute(name, function)
    
    def rm_attribute(self, name):
        for k, v in list(self.KEYS.items()):
            v._rm_attribute(name)
    
    def get(self, name):
        return [getattr(v, name) for v in list(self.KEYS.values())]