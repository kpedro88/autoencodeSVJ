//
//  drawLossEvolution.cpp
//  xTrainingAnalysis
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#include "Helpers.hpp"

#include "Result.hpp"
#include "ResultsProcessor.hpp"

string aucsPath =  "../trainingResults/aucs/customStandardScaler/";
string resultsPath =  "../trainingResults/trainingRuns/customStandardScaler/";
string filePattern = "hlf_eflow3_8_v";


void drawGraph(const ModelStats &stats, string title)
{
  vector<Epoch> epochsInModel = stats.epochs;
  
  TGraph *trainingLoss = new TGraph();
  TGraph *validationLoss = new TGraph();
  
  int iPoint=0;
  
  for(Epoch epoch : epochsInModel){
    trainingLoss->SetPoint(iPoint, iPoint, epoch.trainingLoss);
    validationLoss->SetPoint(iPoint, iPoint, epoch.validationLoss);
    iPoint++;
  }
  
  trainingLoss->SetMarkerStyle(20);
  trainingLoss->SetMarkerSize(0.3);
  trainingLoss->SetMarkerColor(kRed+1);
  
  validationLoss->SetMarkerStyle(20);
  validationLoss->SetMarkerSize(0.3);
  validationLoss->SetMarkerColor(kGreen+1);
    
  
  trainingLoss->Draw("AP");
  validationLoss->Draw("Psame");
  
  trainingLoss->SetTitle(title.c_str());
  trainingLoss->GetXaxis()->SetTitle("Epoch");
  trainingLoss->GetYaxis()->SetTitle("Loss");
  
  trainingLoss->GetXaxis()->SetLimits(0, 200);
//  trainingLoss->GetYaxis()->SetRangeUser(0, 0.15);
  
  TLegend *legend = new TLegend(0.5, 0.7, 0.9, 0.9);
  legend->AddEntry(trainingLoss, "training loss", "p");
  legend->AddEntry(validationLoss, "validation loss", "p");
  legend->Draw();
}

int main()
{
  cout<<"Starting drawLossEvolution"<<endl;
  gStyle->SetOptStat(0);
  useCommaAsDecimalSeparator();
 
  cout<<"Creating application"<<endl;
  TApplication app("", 0, {});
  
  cout<<"Reading results from files"<<endl;
  auto resultsProcessor = make_unique<ResultsProcessor>();
  vector<ModelStats> stats = resultsProcessor->getModelStatsFromPathMarchingPatter(aucsPath, resultsPath, filePattern);
  
  resultsProcessor->sortModelsByAucAverageOverAllSignals(stats);
  
  cout<<"The best model based on average AUC over all signals: "<<stats.front().aucsFileName<<endl;
  
  cout<<"Plotting results"<<endl;
  TCanvas *canvas = new TCanvas("c1", "c1", 600, 1000);
  canvas->Divide(1, 3);
  
  canvas->cd(1);
  drawGraph(stats.front(), "The best model");
  
  canvas->cd(2);
  drawGraph(stats[stats.size()/2.], "Average model");
  
  canvas->cd(3);
  drawGraph(stats.back(), "The worst model");
  
  canvas->Update();
  
  cout<<"Running the application"<<endl;
  app.Run();
  return 0;
}

