import ROOT


class PhysObject:
    def __init__(self, eta, phi, pt, mass):
        """
        Initializes simple representation of a physics object (track/neutral hadron/photon), containing
        just eta, phi, pt and mass.
        """
        
        self.eta = eta
        self.phi = phi
        self.pt = pt
        self.mass = mass if mass is not None else 0
        
    def get_four_vector(self):
        """
        Returns ROOT TLorentzVector corresponding to this physics object.
        """
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM(self.pt, self.eta, self.phi, self.mass)
        return vector