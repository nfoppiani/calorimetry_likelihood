#include "Riostream.h"
// #include "TString.h"
// #include "TFile.h"
// #include "TTree.h"
#include <map>
#include <iostream>
#include <cstdlib>

void slimmer_1enp_far_sideband()
{
  // Get old file, old tree and set top branch address
  TString finname = "~/Desktop/MicroBooNE/bnb_nue_analysis/PELEE/0304/run1/nuepresel/data_bnb_peleeTuple_unblinded_uboone_v08_00_00_42_run1_C1_nuepresel.root";
  TString foutname = "~/Desktop/MicroBooNE/bnb_nue_analysis/PELEE/0304/run1/nuepresel/data_bnb_peleeTuple_unblinded_uboone_v08_00_00_42_run1_C1_nuepresel_1enp_far_sideband_skimmed.root";
  TFile oldfile(finname);
  TTree *oldtree;
  oldfile.GetObject("searchingfornues/NeutrinoSelectionFilter", oldtree);

  int numevts = 0;

  const auto nentries = oldtree->GetEntries();

  // Deactivate all branches
  oldtree->SetBranchStatus("*", 1);

  int backtracked_pdg;

  int nslice;
  int selected;
  float shr_energy_tot_cali;
  uint n_tracks_contained;
  uint n_showers_contained;
  float bdt_pi0_np;
  float bdt_nonpi0_np;
  float reco_e;

  oldtree->SetBranchAddress("nslice", &nslice);
  oldtree->SetBranchAddress("selected", &selected);
  oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  oldtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  oldtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  oldtree->SetBranchAddress("bdt_pi0_np", &bdt_pi0_np);
  oldtree->SetBranchAddress("bdt_nonpi0_np", &bdt_nonpi0_np);
  oldtree->SetBranchAddress("reco_e", &reco_e);


  // Create a new file + a clone of old tree in new file
  TFile newfile(foutname, "recreate");
  TDirectory *searchingfornues = newfile.mkdir("searchingfornues");
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

    bool preseq = (nslice == 1) &&
         (selected == 1) &&
         (shr_energy_tot_cali > 0.07);

    bool np_preseq = preseq && (n_tracks_contained > 0);
    bool np_preseq_one_shower = np_preseq && (n_showers_contained == 1);
    // bool low_pid = ((bdt_pi0_np > 0) && (bdt_pi0_np < 0.1)) || ((bdt_nonpi0_np > 0) && (bdt_nonpi0_np < 0.1));
    bool low_pid = ((0 < bdt_pi0_np < 0.1) || (0 < bdt_nonpi0_np < 0.1));
    // bool high_energy = (reco_e > 1.05) && (reco_e < 2.05);
    bool high_energy = (1.05 < reco_e < 2.05);
    bool far_sideband = np_preseq_one_shower && (low_pid || high_energy);

    if (far_sideband)
    {
      newtree->Fill();
    }// if cuts pass
  }// for all entries
  newtree->Print();
  newfile.Write();
}