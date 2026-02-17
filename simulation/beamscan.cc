//
// BeamScan: BL4S 2026 — Material classification via MCS and energy loss
// Main application
//

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "FTFP_BERT.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4PhysListFactory.hh"
#include "G4StepLimiterPhysics.hh"

int main(int argc, char** argv)
{
    // Detect interactive mode
    G4UIExecutive* ui = nullptr;
    if (argc == 1) {
        ui = new G4UIExecutive(argc, argv);
    }

    // Construct the run manager (auto-detects MT capability)
    auto runManager = G4RunManagerFactory::CreateRunManager(
        G4RunManagerType::Default);

    // Physics list: FTFP_BERT with high-precision EM (option4 for accurate MCS)
    auto physicsList = new FTFP_BERT;
    physicsList->ReplacePhysics(new G4EmStandardPhysics_option4());
    physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    runManager->SetUserInitialization(physicsList);

    // Detector geometry
    auto detConstruction = new DetectorConstruction();
    runManager->SetUserInitialization(detConstruction);

    // User actions
    auto eventAction = new EventAction();
    auto runAction = new RunAction();

    runManager->SetUserAction(new PrimaryGeneratorAction());
    runManager->SetUserAction(runAction);
    runManager->SetUserAction(eventAction);
    runManager->SetUserAction(new SteppingAction(eventAction));

    // Visualization
    auto visManager = new G4VisExecutive;
    visManager->Initialize();

    auto UImanager = G4UImanager::GetUIpointer();

    if (!ui) {
        // Batch mode — run macro from command line
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command + fileName);
    } else {
        // Interactive mode
        UImanager->ApplyCommand("/control/execute macros/init_vis.mac");
        ui->SessionStart();
        delete ui;
    }

    delete visManager;
    delete runManager;
    return 0;
}
