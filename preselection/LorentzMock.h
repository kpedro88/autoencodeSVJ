#include "Helpers.hpp"

#ifndef LorentzMock_h
#define LorentzMock_h

class LorentzMock{
    public:
        LorentzMock() = delete; 

        LorentzMock(Float_t Pt_, Float_t Eta_) {
            this->Eta_ = Eta_;
            this->Pt_ = Pt_; 
        }

        LorentzMock(Float_t Pt_, Float_t Eta_, Float_t Isolation_) {
            this->Eta_ = Eta_;
            this->Pt_ = Pt_; 
            this->Isolation_ = Isolation_;
        }
        
        LorentzMock(Float_t Pt_, Float_t Eta_, Float_t Isolation_, Float_t EhadOverEem_) {
            this->Eta_ = Eta_;
            this->Pt_ = Pt_; 
            this->Isolation_ = Isolation_;
            this->EhadOverEem_ = EhadOverEem_; 
        }

        Float_t Eta() {
            return this->Eta_; 
        }
        Float_t Pt() {
            return this->Pt_; 
        }
        Float_t Isolation() {
            return this->Isolation_;
        }
        Float_t EhadOverEem() {
            return this->EhadOverEem_; 
        }
        
    private:
        Float_t Eta_, Pt_, Isolation_, EhadOverEem_;
};

#endif /* LorentzMock_h */
