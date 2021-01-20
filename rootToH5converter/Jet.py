import ROOT

class Jet:
    
    def __init__(self):
        self.eta = 99999
        self.phi = 99999
        self.pt = 99999
        self.mass = 99999
        
    def print(self):
        print("Eta: ", self.eta, "\tphi: ", self.phi, "\tpt: ", self.pt, "\tm: ", self.mass)
        
    def P4(self):
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM (self.pt, self.eta, self.phi, self.mass)
        return vector
    