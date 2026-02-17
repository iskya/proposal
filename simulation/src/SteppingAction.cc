#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"

#include "G4Step.hh"
#include "G4RunManager.hh"

SteppingAction::SteppingAction(EventAction* eventAction)
    : fEventAction(eventAction)
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    auto volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    
    auto det = static_cast<const DetectorConstruction*>(
        G4RunManager::GetRunManager()->GetUserDetectorConstruction());

    // Record first hit in each tracker plane
    auto pos = step->GetPreStepPoint()->GetPosition();
    auto mom = step->GetPreStepPoint()->GetMomentum();

    if      (volume == det->GetTracker1LV()) fEventAction->RecordTrackerHit(0, pos, mom);
    else if (volume == det->GetTracker2LV()) fEventAction->RecordTrackerHit(1, pos, mom);
    else if (volume == det->GetTracker3LV()) fEventAction->RecordTrackerHit(2, pos, mom);
    else if (volume == det->GetTracker4LV()) fEventAction->RecordTrackerHit(3, pos, mom);

    // Accumulate energy in calorimeter
    if (volume == det->GetCaloLV()) {
        fEventAction->AddCalorEnergy(step->GetTotalEnergyDeposit());
    }
}
