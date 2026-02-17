#include "RunAction.hh"
#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"

RunAction::RunAction()
{
    auto analysisManager = G4AnalysisManager::Instance();
    analysisManager->SetDefaultFileType("csv");
    analysisManager->SetVerboseLevel(1);

    // Create ntuple with all observables
    analysisManager->CreateNtuple("beamscan", "BeamScan MCS and energy loss data");
    analysisManager->CreateNtupleDColumn("theta3D_mrad");    // 0: 3D scattering angle
    analysisManager->CreateNtupleDColumn("thetaX_mrad");     // 1: projected X scattering
    analysisManager->CreateNtupleDColumn("thetaY_mrad");     // 2: projected Y scattering
    analysisManager->CreateNtupleDColumn("pIn_GeV");         // 3: momentum before target
    analysisManager->CreateNtupleDColumn("pOut_GeV");        // 4: momentum after target
    analysisManager->CreateNtupleDColumn("deltaP_MeV");      // 5: momentum loss
    analysisManager->CreateNtupleDColumn("xTarget_mm");      // 6: x position at target
    analysisManager->CreateNtupleDColumn("yTarget_mm");      // 7: y position at target
    analysisManager->CreateNtupleDColumn("caloEdep_MeV");    // 8: calorimeter energy deposit
    analysisManager->FinishNtuple();
}

void RunAction::BeginOfRunAction(const G4Run*)
{
    auto analysisManager = G4AnalysisManager::Instance();
    analysisManager->OpenFile("beamscan_output");
}

void RunAction::EndOfRunAction(const G4Run*)
{
    auto analysisManager = G4AnalysisManager::Instance();
    analysisManager->Write();
    analysisManager->CloseFile();
}
