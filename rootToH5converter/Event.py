import numpy as np

class Event:
    def __init__(self):
        self.met = 99999
        self.metEta = 99999
        self.metPhi = 99999
        self.Mjj = 99999
        self.MT = 99999
        
        self.jets = []
    
    def calculate_internals(self):
        
        if len(self.jets) <= 1:
            print("ERROR -- events has less than 2 jets, which should never happen!")
        
        if len(self.jets) != 2:
            print("ERROR -- expected two jets in the event, but there are ", len(self.jets))
        
        Vjj = self.jets[0].P4() + self.jets[1].P4()
        met_py = self.met * np.sin(self.metPhi)
        met_px = self.met * np.cos(self.metPhi)
    
        Mjj = Vjj.M()
        Mjj2 = Mjj * Mjj
        ptjj = Vjj.Pt()
        ptjj2 = ptjj * ptjj
        ptMet = Vjj.Px() * met_px + Vjj.Py() * met_py
    
        MT = np.sqrt(Mjj2 + 2. * (np.sqrt(Mjj2 + ptjj2) * self.met - ptMet))
        
        self.Mjj = Mjj
        self.MT = MT

    def get_features(self):
        return [
            self.met,
            self.metEta,
            self.metPhi,
            self.MT,
            self.Mjj,
        ]
    
    @staticmethod
    def get_features_names():
        return [
            'MET',
            'METEta',
            'METPhi',
            'MT',
            'Mjj',
        ]