#include "EventAction.hh"
#include "RunAction.hh"

#include "G4Event.hh"
#include "G4AnalysisManager.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>

void EventAction::BeginOfEventAction(const G4Event*)
{
    for (int i = 0; i < 4; ++i) {
        fHitFlag[i] = false;
        fHitPos[i] = G4ThreeVector();
        fHitMom[i] = G4ThreeVector();
    }
    fCaloEdep = 0.0;
}

void EventAction::RecordTrackerHit(G4int id, G4ThreeVector pos, G4ThreeVector mom)
{
    if (id >= 0 && id < 4 && !fHitFlag[id]) {
        fHitPos[id] = pos;
        fHitMom[id] = mom;
        fHitFlag[id] = true;
    }
}

void EventAction::EndOfEventAction(const G4Event*)
{
    // Require hits in all 4 trackers (particle traversed full setup)
    for (int i = 0; i < 4; ++i) {
        if (!fHitFlag[i]) return;
    }

    // Compute incoming direction from trackers 1,2
    G4ThreeVector inDir = (fHitPos[1] - fHitPos[0]).unit();

    // Compute outgoing direction from trackers 3,4
    G4ThreeVector outDir = (fHitPos[3] - fHitPos[2]).unit();

    // 3D scattering angle
    G4double cosTheta = inDir.dot(outDir);
    if (cosTheta > 1.0) cosTheta = 1.0;
    if (cosTheta < -1.0) cosTheta = -1.0;
    G4double theta3D = std::acos(cosTheta);  // radians

    // Projected scattering angles (x and y planes)
    G4double thetaX = std::atan2(outDir.x(), outDir.z()) - std::atan2(inDir.x(), inDir.z());
    G4double thetaY = std::atan2(outDir.y(), outDir.z()) - std::atan2(inDir.y(), inDir.z());

    // Momentum before and after target
    G4double pIn  = fHitMom[1].mag();  // at tracker 2 (just before target)
    G4double pOut = fHitMom[2].mag();  // at tracker 3 (just after target)
    G4double deltaP = pIn - pOut;      // momentum loss in target

    // Position at target (interpolated)
    G4double xTarget = (fHitPos[1].x() + fHitPos[2].x()) / 2.0;
    G4double yTarget = (fHitPos[1].y() + fHitPos[2].y()) / 2.0;

    // Fill ntuple
    auto analysisManager = G4AnalysisManager::Instance();
    analysisManager->FillNtupleDColumn(0, theta3D / mrad);
    analysisManager->FillNtupleDColumn(1, thetaX / mrad);
    analysisManager->FillNtupleDColumn(2, thetaY / mrad);
    analysisManager->FillNtupleDColumn(3, pIn / GeV);
    analysisManager->FillNtupleDColumn(4, pOut / GeV);
    analysisManager->FillNtupleDColumn(5, deltaP / MeV);
    analysisManager->FillNtupleDColumn(6, xTarget / mm);
    analysisManager->FillNtupleDColumn(7, yTarget / mm);
    analysisManager->FillNtupleDColumn(8, fCaloEdep / MeV);
    analysisManager->AddNtupleRow();
}
