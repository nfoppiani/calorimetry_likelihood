#include "Riostream.h"
#include <map>
#include <iostream>
#include <cstdlib>

void slimmer_calo_mc(TString dir, TString fname)
{
  float detector_x[2] = {-1.55, 254.8};
  float detector_y[2] = {-115.53, 117.47};
  float detector_z[2] = {0.1, 1036.9};

  // Get old file, old tree and set top branch address
  TString input_file_path = dir + fname + ".root";
  gSystem->ExpandPathName(dir);
  TFile oldfile(input_file_path);
  TTree *oldtree;
  oldfile.GetObject("nuselection/CalorimetryAnalyzer", oldtree);

  int numevts = 0;

  const auto nentries = oldtree->GetEntries();

  // Deactivate all branches
  oldtree->SetBranchStatus("*", 1);

  int backtracked_pdg;

  float trk_sce_start_x;
  float trk_sce_start_y;
  float trk_sce_start_z;
  float trk_sce_end_x;
  float trk_sce_end_y;
  float trk_sce_end_z;

  oldtree->SetBranchAddress("backtracked_pdg", &backtracked_pdg);
    
  oldtree->SetBranchAddress("trk_sce_start_x", &trk_sce_start_x);
  oldtree->SetBranchAddress("trk_sce_start_y", &trk_sce_start_y);
  oldtree->SetBranchAddress("trk_sce_start_z", &trk_sce_start_z);
  oldtree->SetBranchAddress("trk_sce_end_x", &trk_sce_end_x);
  oldtree->SetBranchAddress("trk_sce_end_y", &trk_sce_end_y);
  oldtree->SetBranchAddress("trk_sce_end_z", &trk_sce_end_z);


  // Create a new file + a clone of old tree in new file
  TString output_file_path = dir + fname + "_mc_skimmed.root";
  TFile newfile(output_file_path, "recreate");
  TDirectory *searchingfornues = newfile.mkdir("nuselection");
  searchingfornues->cd();

  auto newtree = oldtree->CloneTree(0);

  std::cout << "Start loop with entries " << nentries << std::endl;

  for (auto i : ROOT::TSeqI(nentries))
  {
    if (i % 10000 == 0)
    {
      std::cout << "Entry num  " << i << std::endl;
    }

    oldtree->GetEntry(i);

    if (
         ((backtracked_pdg == 321) ||
         (backtracked_pdg == 2212) ||
         (backtracked_pdg == -321) ||
         (backtracked_pdg == -2212) ||
         (backtracked_pdg == 0)) &&
         (trk_sce_start_x > (detector_x[0]+20)) &&
         (trk_sce_start_x < (detector_x[1]-20)) &&
         (trk_sce_start_y > (detector_y[0]+20)) &&
         (trk_sce_start_y < (detector_y[1]-20)) &&
         (trk_sce_start_z > (detector_z[0]+20)) &&
         (trk_sce_start_z < (detector_z[1]-20)) &&
         (trk_sce_end_x > (detector_x[0]+20)) &&
         (trk_sce_end_x < (detector_x[1]-20)) &&
         (trk_sce_end_y > (detector_y[0]+20)) &&
         (trk_sce_end_y < (detector_y[1]-20)) &&
         (trk_sce_end_z > (detector_z[0]+20)) &&
         (trk_sce_end_z < (detector_z[1]-20))
       )
    {
      newtree->Fill();
    }// if cuts pass
  }// for all entries
  newtree->Print();

  TTree *subrunTree;
  oldfile.GetObject("nuselection/SubRun", subrunTree);
  auto newSubrunTree = subrunTree->CloneTree();
  newSubrunTree->Print();

  TTree *eventTree;
  oldfile.GetObject("nuselection/NeutrinoSelectionFilter", eventTree);
  auto newEventTree = eventTree->CloneTree();
  newEventTree->Print();

  newfile.Write();
}