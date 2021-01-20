import ROOT
import numpy as np

class Jet:
    
    def __init__(self):
        self.eta = 99999
        self.phi = 99999
        self.pt = 99999
        self.mass = 99999
        
        self.nCharged = 99999
        self.nNeutral = 99999
        
        self.chargedHadronEnergyFraction = 99999
        self.neutralHadronEnergyFraction = 99999
        
    def print(self):
        print("Eta: ", self.eta, "\tphi: ", self.phi, "\tpt: ", self.pt, "\tm: ", self.mass)
        
    def P4(self):
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM (self.pt, self.eta, self.phi, self.mass)
        return vector

    # def get_features(self):
    #
    #     n_total = self.nCharged + self.nNeutral
    #     jet_cfrac = self.nCharged / n_total if n_total > 0 else -1
    #
    #     ptd, axis2 = self.jet_axis2_pt2(self.P4(), constituentp4s)
    #
    #     return [
    #         self.eta,
    #         self.phi,
    #         self.pt,
    #         self.mass,
    #         jet_cfrac,
    #         ptd,
    #         axis2,
    #         jet_raw.Flavor,
    #         j.E(),
    #     ]
    #
    # def jet_axis2_pt2(self, jet, constituents):
    #
    #     sum_weight = 0
    #     sum_pt = 0
    #     sum_deta = 0
    #     sum_dphi = 0
    #     sum_deta2 = 0
    #     sum_detadphi = 0
    #     sum_dphi2 = 0
    #
    #     for i, c in enumerate(constituents):
    #         deta = c.Eta() - jet.Eta()
    #         dphi = c.DeltaPhi(jet)
    #         cpt = c.Pt()
    #         weight = cpt * cpt
    #
    #         sum_weight += weight
    #         sum_pt += cpt
    #         sum_deta += deta * weight
    #         sum_dphi += dphi * weight
    #         sum_deta2 += deta * deta * weight
    #         sum_detadphi += deta * dphi * weight
    #         sum_dphi2 += dphi * dphi * weight
    #
    #     a, b, c, ave_deta, ave_dphi, ave_deta2, ave_dphi2 = 0, 0, 0, 0, 0, 0, 0
    #
    #     if sum_weight > 0.:
    #         ave_deta = sum_deta / sum_weight
    #         ave_dphi = sum_dphi / sum_weight
    #         ave_deta2 = sum_deta2 / sum_weight
    #         ave_dphi2 = sum_dphi2 / sum_weight
    #         a = ave_deta2 - ave_deta * ave_deta
    #         b = ave_dphi2 - ave_dphi * ave_dphi
    #         c = -(sum_detadphi / sum_weight - ave_deta * ave_dphi)
    #
    #     delta = np.sqrt(np.abs((a - b) * (a - b) + 4 * c * c))
    #     axis2 = np.sqrt(0.5 * (a + b - delta)) if a + b - delta > 0 else 0
    #     ptD = np.sqrt(sum_weight) / sum_pt if sum_weight > 0 else 0
    #
    #     return ptD, axis2
    #
    # def get_constituent_p4s(self, tree, jets, dr=0.8):
    #
    #     constituents = [[] for i in range(len(jets))]
    #
    #     for i, c in enumerate(tree.EFlowTrack):  # .1
    #         if c.PT > 0.1:
    #             vec = c.P4()
    #             for j, jet in enumerate(jets):
    #                 deta = vec.Eta() - jet.Eta()
    #                 dphi = jet.DeltaPhi(vec)
    #                 if deta ** 2. + dphi ** 2. < dr ** 2.:
    #                     constituents[j].append(vec)
    #
    #     for i, c in enumerate(tree.EFlowNeutralHadron):  # .5
    #         if c.ET > 0.5:
    #             vec = c.P4()
    #             for j, jet in enumerate(jets):
    #                 deta = vec.Eta() - jet.Eta()
    #                 dphi = jet.DeltaPhi(vec)
    #                 if deta ** 2. + dphi ** 2. < dr ** 2.:
    #                     constituents[j].append(vec)
    #
    #     for i, c in enumerate(tree.EFlowPhoton):  # .2
    #         if c.ET > 0.2:
    #             vec = c.P4()
    #             for j, jet in enumerate(jets):
    #                 deta = vec.Eta() - jet.Eta()
    #                 dphi = jet.DeltaPhi(vec)
    #                 if deta ** 2. + dphi ** 2. < dr ** 2.:
    #                     constituents[j].append(vec)
    #
    #     return list(map(np.asarray, constituents))