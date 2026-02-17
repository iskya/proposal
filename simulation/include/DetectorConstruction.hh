#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;

    // Accessors for event/stepping action
    G4LogicalVolume* GetTracker1LV() const { return fTracker1LV; }
    G4LogicalVolume* GetTracker2LV() const { return fTracker2LV; }
    G4LogicalVolume* GetTracker3LV() const { return fTracker3LV; }
    G4LogicalVolume* GetTracker4LV() const { return fTracker4LV; }
    G4LogicalVolume* GetTargetLV()   const { return fTargetLV; }
    G4LogicalVolume* GetCaloLV()     const { return fCaloLV; }

private:
    void DefineMaterials();
    void DefineCommands();

    // Configurable via macro commands
    G4String  fTargetMaterial = "G4_POLYETHYLENE";
    G4double  fTargetThickness = 10.0*mm;
    G4bool    fUseMagnet = false;
    G4double  fMagnetFieldT = 0.5; // Tesla

    // Logical volumes for sensitive detectors
    G4LogicalVolume* fTracker1LV = nullptr;
    G4LogicalVolume* fTracker2LV = nullptr;
    G4LogicalVolume* fTracker3LV = nullptr;
    G4LogicalVolume* fTracker4LV = nullptr;
    G4LogicalVolume* fTargetLV   = nullptr;
    G4LogicalVolume* fCaloLV     = nullptr;

    G4GenericMessenger* fMessenger = nullptr;
};

#endif
