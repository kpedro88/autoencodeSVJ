import ROOT
import numpy as np
import energyflow as ef

class Jet:
    
    def __init__(self, tree, input_type, iEvent, iJet):
    
        if input_type == "Delphes":
            self.eta = tree["Jet/Jet.Eta"].array()[iEvent][iJet]
            self.phi = tree["Jet/Jet.Phi"].array()[iEvent][iJet]
            self.pt = tree["Jet/Jet.PT"].array()[iEvent][iJet]
            self.mass = tree["Jet/Jet.Mass"].array()[iEvent][iJet]
            self.nCharged = tree["Jet/Jet.NCharged"].array()[iEvent][iJet]
            self.nNeutral = tree["Jet/Jet.NNeutrals"].array()[iEvent][iJet]
            self.flavor = tree["Jet/Jet.Flavor"].array()[iEvent][iJet]

            n_total = self.nCharged + self.nNeutral
            self.chargedHadronEnergyFraction = self.nCharged / n_total if n_total > 0 else -1
            self.neutralHadronEnergyFraction = self.nNeutral / n_total if n_total > 0 else -1
            
        elif input_type == "nanoAOD" or input_type == "PFnanoAOD":
            self.eta = tree["Jet_eta"].array()[iEvent][iJet]
            self.phi = tree["Jet_phi"].array()[iEvent][iJet]
            self.pt = tree["Jet_pt"].array()[iEvent][iJet]
            self.mass = tree["Jet_mass"].array()[iEvent][iJet]
            self.flavor = -1
        
            self.chargedHadronEnergyFraction = tree["Jet_chHEF"].array()[iEvent][iJet]
            self.neutralHadronEnergyFraction = tree["Jet_neHEF"].array()[iEvent][iJet]
            
        
        self.constituents = []
        
    def print(self):
        print("Eta: ", self.eta, "\tphi: ", self.phi, "\tpt: ", self.pt, "\tm: ", self.mass)
        
    def get_four_vector(self):
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM (self.pt, self.eta, self.phi, self.mass)
        return vector

    @staticmethod
    def get_feature_names():
        return [
            'Eta',
            'Phi',
            'Pt',
            'M',
            'ChargedFraction',
            'PTD',
            'Axis2',
            'Flavor',
            'Energy',
        ]

    def get_features(self):
        return [
            self.eta,
            self.phi,
            self.pt,
            self.mass,
            self.chargedHadronEnergyFraction,
            self.get_ptD(),
            self.get_axis2(),
            self.flavor,
            self.get_four_vector().E(),
        ]

    @staticmethod
    def get_constituent_feature_names():
        return [
            'Eta',
            'Phi',
            'PT',
            'Rapidity',
            'Energy',
        ]

    def get_ptD(self):

        sum_weight = 0
        sum_pt = 0
        
        for i, c in enumerate(self.constituents):
            sum_weight += c.Pt() ** 2
            sum_pt += c.Pt()
            
        ptD = np.sqrt(sum_weight) / sum_pt if sum_weight > 0 else 0

        return ptD

    def get_axis2(self):
    
        sum_weight = 0
        sum_deta = 0
        sum_dphi = 0
        sum_deta2 = 0
        sum_detadphi = 0
        sum_dphi2 = 0
    
        for i, c in enumerate(self.constituents):
            deta = c.Eta() - self.eta
            dphi = c.DeltaPhi(self.get_four_vector())
            weight = c.Pt() ** 2
        
            sum_weight += weight
            sum_deta += deta * weight
            sum_dphi += dphi * weight
            sum_deta2 += deta * deta * weight
            sum_detadphi += deta * dphi * weight
            sum_dphi2 += dphi * dphi * weight
    
        a, b, c, ave_deta, ave_dphi, ave_deta2, ave_dphi2 = 0, 0, 0, 0, 0, 0, 0
    
        if sum_weight > 0.:
            ave_deta = sum_deta / sum_weight
            ave_dphi = sum_dphi / sum_weight
            ave_deta2 = sum_deta2 / sum_weight
            ave_dphi2 = sum_dphi2 / sum_weight
            a = ave_deta2 - ave_deta * ave_deta
            b = ave_dphi2 - ave_dphi * ave_dphi
            c = -(sum_detadphi / sum_weight - ave_deta * ave_dphi)
    
        delta = np.sqrt(np.abs((a - b) * (a - b) + 4 * c * c))
        axis2 = np.sqrt(0.5 * (a + b - delta)) if a + b - delta > 0 else 0
    
        return axis2

    def get_constituents(self, physObjects, pt_cut, delta_r, iJet=-1, track_jet_index=None):
        constituents = []
        
        if iJet < 0:
            for i, object in enumerate(physObjects):
                if object.pt > pt_cut:
                    vec = object.get_four_vector()
                    
                    delta_eta = object.eta - self.eta
                    delta_phi = self.get_four_vector().DeltaPhi(vec)
                    if delta_eta ** 2. + delta_phi ** 2. < delta_r ** 2.:
                        constituents.append(vec)
        else:
            for i, track in enumerate(physObjects):
                jet_index = track_jet_index[i]
                if jet_index == iJet:
                    constituents.append(track.get_four_vector())

        return constituents

    def fill_constituents(self, tracks, neutral_hadrons, photons, delta_r, iJet, track_jet_index):
        
        self.constituents.extend(self.get_constituents(tracks, 0.1, delta_r, iJet, track_jet_index))
        self.constituents.extend(self.get_constituents(neutral_hadrons, 0.5, delta_r))
        self.constituents.extend(self.get_constituents(photons, 0.2, delta_r))

    def get_EFPs(self, EFP_set):
        if len(self.constituents) == 0:
            return
        
        return EFP_set.compute(
            ef.utils.ptyphims_from_p4s(
                [(c.E(), c.Px(), c.Py(), c.Pz()) for c in self.constituents]
            )
        )