#include <cstring>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TString.h>


void clone_tree_with_condition(const char* input_file, const char* output_file) {
  TFile *fin = TFile::Open(input_file);
  TTree *tree_in = (TTree*)fin->Get("Nominal");

  TFile *fout = TFile::Open(output_file, "recreate");
  TTree *tree_out = tree_in->CloneTree(0);

  float pTV;
  int nbJets, nJets;
  string *sample;

  tree_in->SetBranchAddress("pTV", &pTV);
  tree_in->SetBranchAddress("nbJets", &nbJets);
  tree_in->SetBranchAddress("nJets", &nJets);
  tree_in->SetBranchAddress("sample", &sample);

  for (int i = 0; i < tree_in->GetEntries(); i++) {
    if (i%1000000 == 0) std::cout<<"Event "<<i<<std::endl;
    tree_in->GetEntry(i);
    if (pTV < 75 || pTV > 150) continue;
    if (nbJets!=2) continue;
    if (nJets!=2 && nJets!=3) continue;
    if (*sample == "data") continue;
    tree_out->Fill();
  }

  tree_out->Write();
  fout->Close();
  fin->Close();
}
