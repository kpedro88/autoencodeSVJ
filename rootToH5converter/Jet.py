import ROOT
import numpy as np
import energyflow as ef


class Jet:
    
    def __init__(self, eta, phi, pt, mass, flavor, n_charged=None, n_neutral=None, ch_hef=None, ne_hef=None):
        """
        Initializes jet variables from the provided tree, using event index iEvent and jet index iJet
        to find the corresponding entry. Branches' names will be determined based on the provided input_typ
        (can be "Delphes", "nanoAOD" or "PFnanoAOD").
        """
        self.eta = eta
        self.phi = phi
        self.pt = pt
        self.mass = mass
        self.flavor = flavor

        self.n_charged = n_charged
        self.n_neutral = n_neutral

        self.chargedHadronEnergyFraction = ch_hef
        self.neutralHadronEnergyFraction = ne_hef
    
        if self.chargedHadronEnergyFraction is None:
            # try to re-calculate charged and neutral energy fractions (for Delphes)
            n_total = self.n_charged + self.n_neutral
            self.chargedHadronEnergyFraction = self.n_charged / n_total if n_total > 0 else -1
            self.neutralHadronEnergyFraction = self.n_neutral / n_total if n_total > 0 else -1
    
        self.constituents = []
        
    def print(self):
        """
        Prints basic information about the jet.
        """
        print("Eta: ", self.eta, "\tphi: ", self.phi, "\tpt: ", self.pt, "\tm: ", self.mass)
        
    def get_four_vector(self):
        """
        Returns ROOT TLorentzVector of the jet.
        """
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM (self.pt, self.eta, self.phi, self.mass)
        return vector

    @staticmethod
    def get_feature_names():
        """
        Returns names of the jet features.
        """
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
        """
        Returns jet features.
        """
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
        """
        Returns names of the jet constituent features
        """
        
        return [
            'Eta',
            'Phi',
            'PT',
            'Rapidity',
            'Energy',
        ]

    def get_ptD(self):
        """
        Calculates and returns the ptD variable based on the jet constutuents.
        """

        sum_weight = 0
        sum_pt = 0
        
        for i, c in enumerate(self.constituents):
            sum_weight += c.Pt() ** 2
            sum_pt += c.Pt()
            
        ptD = np.sqrt(sum_weight) / sum_pt if sum_weight > 0 else 0

        return ptD

    def get_axis2(self):
        """
        Calculates and returns the axis2 variable based on the jet constituents.
        """
    
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

    def add_constituents(self, physObjects, pt_cut, delta_r, iJet=-1, track_jet_index=None):
        """
        Adds constituents from physObjects collection, which pass the pt cut. If iJet and track_jet_index
        are not specified, it will add constituents within delta_r. Otherwise, it will check jet index for each
        constituent and add it only if it's the same as provided jet index iJet.
        """
        
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

        self.constituents.extend(constituents)

    def fill_constituents(self, tracks, neutral_hadrons, photons, delta_r, iJet, track_jet_index):
        """
        Fills collection of jet constituents with tracks, neutral hadrons and photons. If iJet and track_jet_index
        are not specified, it will add constituents within delta_r. Otherwise, it will check jet index for each
        constituent and add it only if it's the same as provided jet index iJet.
        """
        
        self.add_constituents(tracks, 0.1, delta_r, iJet, track_jet_index)
        self.add_constituents(neutral_hadrons, 0.5, delta_r)
        self.add_constituents(photons, 0.2, delta_r)

    def get_EFPs(self, EFP_set):
        """
        Calculates and returns EFPs from jet constituents with the provided EFP set.
        """
        
        if len(self.constituents) == 0:
            return
        
        return EFP_set.compute(
            ef.utils.ptyphims_from_p4s(
                [(c.E(), c.Px(), c.Py(), c.Pz()) for c in self.constituents]
            )
        )

    def get_constituents(self, max):
        """
        Returns np array with all jet constituents, up to specified maximum. If there are less constituents than
        specified max, remaining entries will be padded with zeros.
        """
    
        constituents = -np.ones((len(self.constituents), len(Jet.get_constituent_feature_names())))
        for i, c in enumerate(self.constituents):
            constituents[i, :] = [c.Eta(), c.Phi(), c.Pt(), c.Rapidity(), c.E()]

        constituents = constituents[np.argsort(constituents[:, 2]), :][::-1][:max, :]
        constituents = np.pad(constituents, ((0, max - constituents.shape[0]), (0, 0)), 'constant')
    
        return constituents