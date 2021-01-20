import ROOT

class PhysObject:
    def __init__(self, eta, phi, pt, mass):
        
        self.eta = eta
        self.phi = phi
        self.pt = pt
        self.mass = mass
        
    def get_four_vector(self):
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM(self.pt, self.eta, self.phi, self.mass)
        return vector
        