#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "G4ThreeVector.hh"
#include <vector>

class EventAction : public G4UserEventAction
{
public:
    EventAction() = default;
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;

    // Called by SteppingAction to record hits
    void RecordTrackerHit(G4int trackerID, G4ThreeVector pos, G4ThreeVector mom);
    void AddCalorEnergy(G4double edep) { fCaloEdep += edep; }

private:
    // Per-event storage
    G4ThreeVector fHitPos[4];   // tracker positions (0-3)
    G4ThreeVector fHitMom[4];   // tracker momenta (0-3)
    G4bool        fHitFlag[4] = {false, false, false, false};
    G4double      fCaloEdep = 0.0;
};

#endif
