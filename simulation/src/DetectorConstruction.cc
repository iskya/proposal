//
// DetectorConstruction.cc — BeamScan geometry
//
// Layout along Z-axis (beam direction):
//   DWC1 → DWC2 → [gap] → TARGET → [gap] → DWC3 → DWC4 → Calorimeter (optional)
//
// Target material and thickness are configurable via macro commands:
//   /beamscan/target/material G4_POLYETHYLENE
//   /beamscan/target/thickness 10 mm
//

#include "DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4UniformMagField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4GenericMessenger.hh"
#include "G4Material.hh"
#include "G4Element.hh"

DetectorConstruction::DetectorConstruction()
{
    DefineCommands();
}

DetectorConstruction::~DetectorConstruction()
{
    delete fMessenger;
}

void DetectorConstruction::DefineMaterials()
{
    auto nist = G4NistManager::Instance();

    // Standard NIST materials (plastics)
    nist->FindOrBuildMaterial("G4_POLYETHYLENE");
    nist->FindOrBuildMaterial("G4_POLYPROPYLENE");
    nist->FindOrBuildMaterial("G4_POLYSTYRENE");
    nist->FindOrBuildMaterial("G4_PLEXIGLASS");          // PMMA
    nist->FindOrBuildMaterial("G4_MYLAR");                // PET (Mylar = biaxially-oriented PET)
    nist->FindOrBuildMaterial("G4_POLYVINYL_CHLORIDE");
    nist->FindOrBuildMaterial("G4_NYLON-6-6");

    // Heritage/geological materials
    nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE");      // Quartz (Sierras de Córdoba)
    nist->FindOrBuildMaterial("G4_CALCIUM_CARBONATE");    // Calcite / limestone / clay proxy
    nist->FindOrBuildMaterial("G4_ALUMINUM_OXIDE");       // Alumina / ceramic proxy
    nist->FindOrBuildMaterial("G4_FERRIC_OXIDE");         // Fe2O3 / iron pigment

    // Elements for custom materials
    auto Si = nist->FindOrBuildElement("Si");
    auto O  = nist->FindOrBuildElement("O");
    auto Al = nist->FindOrBuildElement("Al");
    auto Na = nist->FindOrBuildElement("Na");
    auto K  = nist->FindOrBuildElement("K");
    auto Fe = nist->FindOrBuildElement("Fe");
    auto Ca = nist->FindOrBuildElement("Ca");

    // Custom: Granite (approximate)
    // ~72% SiO2, ~14% Al2O3, ~4% K2O, ~3% Na2O, ~3% Fe2O3, ~2% CaO, ~2% MgO
    auto Mg = nist->FindOrBuildElement("Mg");
    auto granite = new G4Material("Granite", 2.70*g/cm3, 8);
    granite->AddElement(Si, 0.3363);
    granite->AddElement(O,  0.4810);
    granite->AddElement(Al, 0.0741);
    granite->AddElement(K,  0.0332);
    granite->AddElement(Na, 0.0223);
    granite->AddElement(Fe, 0.0210);
    granite->AddElement(Ca, 0.0143);
    granite->AddElement(Mg, 0.0178);

    // Detector materials
    // DWC gas: Ar/CO2 80/20 mixture at STP (standard beamline DWC fill)
    auto Ar = nist->FindOrBuildElement("Ar");
    auto C_el = nist->FindOrBuildElement("C");
    auto O_dwc = nist->FindOrBuildElement("O");
    auto dwcGas = new G4Material("DWC_ArCO2", 1.72e-3*g/cm3, 3);  // ~80/20 Ar/CO2 density
    dwcGas->AddElement(Ar, 0.80);
    dwcGas->AddElement(C_el, 0.0545);
    dwcGas->AddElement(O_dwc, 0.1455);
    nist->FindOrBuildMaterial("G4_AIR");
    nist->FindOrBuildMaterial("G4_LEAD_OXIDE");           // Lead glass proxy
    nist->FindOrBuildMaterial("G4_Pb");
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    DefineMaterials();
    auto nist = G4NistManager::Instance();

    // ===== WORLD VOLUME =====
    G4double worldSize = 3.0*m;
    auto worldS = new G4Box("World", worldSize/2, worldSize/2, worldSize/2);
    auto worldLV = new G4LogicalVolume(worldS, nist->FindOrBuildMaterial("G4_AIR"), "World");
    worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());
    auto worldPV = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "World", nullptr, false, 0);

    // ===== GEOMETRY PARAMETERS =====
    // DWC planes: 10 mm gas gap, 10×10 cm² active area
    // (Delay Wire Chambers — standard BL4S tracking detectors)
    G4double dwcThick = 10.0*mm;
    G4double dwcXY    = 10.0*cm;

    // Z positions along beam axis (beam enters at Z = -80 cm)
    G4double z_tracker1 = -50.0*cm;
    G4double z_tracker2 = -25.0*cm;
    // Target centered at Z = 0
    G4double z_tracker3 =  25.0*cm;
    G4double z_tracker4 =  50.0*cm;
    G4double z_calo     = 100.0*cm;

    // ===== TRACKER PLANES =====
    auto dwcS = new G4Box("DWC", dwcXY/2, dwcXY/2, dwcThick/2);
    auto dwcMat = G4Material::GetMaterial("DWC_ArCO2");

    // DWC 1
    fTracker1LV = new G4LogicalVolume(dwcS, dwcMat, "DWC1LV");
    fTracker1LV->SetVisAttributes(new G4VisAttributes(G4Colour::Cyan()));
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, z_tracker1), fTracker1LV, "DWC1", worldLV, false, 0);

    // DWC 2
    fTracker2LV = new G4LogicalVolume(dwcS, dwcMat, "DWC2LV");
    fTracker2LV->SetVisAttributes(new G4VisAttributes(G4Colour::Cyan()));
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, z_tracker2), fTracker2LV, "DWC2", worldLV, false, 0);

    // DWC 3
    fTracker3LV = new G4LogicalVolume(dwcS, dwcMat, "DWC3LV");
    fTracker3LV->SetVisAttributes(new G4VisAttributes(G4Colour::Cyan()));
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, z_tracker3), fTracker3LV, "DWC3", worldLV, false, 0);

    // DWC 4
    fTracker4LV = new G4LogicalVolume(dwcS, dwcMat, "DWC4LV");
    fTracker4LV->SetVisAttributes(new G4VisAttributes(G4Colour::Cyan()));
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, z_tracker4), fTracker4LV, "DWC4", worldLV, false, 0);

    // ===== TARGET =====
    auto targetMat = nist->FindOrBuildMaterial(fTargetMaterial);
    if (!targetMat) {
        // Try custom materials
        targetMat = G4Material::GetMaterial(fTargetMaterial);
    }
    if (!targetMat) {
        G4cerr << "ERROR: Material " << fTargetMaterial << " not found! Using G4_POLYETHYLENE." << G4endl;
        targetMat = nist->FindOrBuildMaterial("G4_POLYETHYLENE");
    }

    auto targetS = new G4Box("Target", 5.0*cm, 5.0*cm, fTargetThickness/2);  // 10×10 cm² face
    fTargetLV = new G4LogicalVolume(targetS, targetMat, "TargetLV");
    fTargetLV->SetVisAttributes(new G4VisAttributes(G4Colour::Red()));
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), fTargetLV, "Target", worldLV, false, 0);

    G4cout << "\n========================================" << G4endl;
    G4cout << " BeamScan Target Configuration:" << G4endl;
    G4cout << "   Material:  " << targetMat->GetName() << G4endl;
    G4cout << "   Thickness: " << fTargetThickness/mm << " mm" << G4endl;
    G4cout << "   Density:   " << targetMat->GetDensity()/(g/cm3) << " g/cm3" << G4endl;
    G4cout << "   Rad.Length: " << targetMat->GetRadlen()/cm << " cm" << G4endl;
    G4cout << "========================================\n" << G4endl;

    // ===== CALORIMETER (lead glass block) =====
    G4double caloXY = 15.0*cm;
    G4double caloZ  = 30.0*cm;
    auto caloS = new G4Box("Calo", caloXY/2, caloXY/2, caloZ/2);
    auto caloMat = nist->FindOrBuildMaterial("G4_LEAD_OXIDE");
    fCaloLV = new G4LogicalVolume(caloS, caloMat, "CaloLV");
    fCaloLV->SetVisAttributes(new G4VisAttributes(G4Colour::Yellow()));
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, z_calo), fCaloLV, "Calo", worldLV, false, 0);

    return worldPV;
}

void DetectorConstruction::ConstructSDandField()
{
    // Optional: uniform magnetic field for spectrometer mode
    if (fUseMagnet) {
        auto field = new G4UniformMagField(G4ThreeVector(0, fMagnetFieldT*tesla, 0));
        auto fieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
        fieldMgr->SetDetectorField(field);
        fieldMgr->CreateChordFinder(field);
        G4cout << "Magnetic field enabled: B = " << fMagnetFieldT << " T (y-direction)" << G4endl;
    }
}

void DetectorConstruction::DefineCommands()
{
    fMessenger = new G4GenericMessenger(this, "/beamscan/target/",
                                        "Target configuration");

    fMessenger->DeclareProperty("material", fTargetMaterial,
        "Target material name (NIST or custom, e.g. G4_POLYETHYLENE, Granite)")
        .SetParameterName("mat", false);

    fMessenger->DeclarePropertyWithUnit("thickness", "mm", fTargetThickness,
        "Target thickness")
        .SetParameterName("t", false)
        .SetRange("t>0.");

    auto& magnetCmd = fMessenger->DeclareProperty("useMagnet", fUseMagnet,
        "Enable uniform magnetic field (spectrometer mode)");

    fMessenger->DeclareProperty("magnetFieldT", fMagnetFieldT,
        "Magnetic field strength in Tesla")
        .SetParameterName("B", false)
        .SetRange("B>=0.");
}
