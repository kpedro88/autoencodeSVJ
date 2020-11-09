//
//  ResultsProcessor.cpp
//  drawAucDistributions
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#include "ResultsProcessor.hpp"


void ResultsProcessor::sortModelsByAucAverageOverAllSignals(vector<ModelStats> &models)
{
  sort(models.begin(), models.end(), ModelStats::sortByAucAverageOverAllSignals());
}

vector<ModelStats> ResultsProcessor::getModelStatsFromPathMarchingPatter(string aucsPath, string resultsPath, string filePattern)
{
  vector<ModelStats> stats;
  
  vector<string> fileNames = getFileInPathMatchingPattern(aucsPath, filePattern);
  
  for(string fileName : fileNames){
    ModelStats statsForTraining;
    
    statsForTraining.aucsFileName = aucsPath+fileName;
    statsForTraining.lossesFileName = resultsPath+fileName+".csv";
    
    statsForTraining.results = getResultsFromFile(statsForTraining.aucsFileName);
    statsForTraining.epochs = getEpochsFromFile(statsForTraining.lossesFileName);
    
    stats.push_back(statsForTraining);
  }
  return stats;
}


vector<Result> ResultsProcessor::getResultsFromFile(string inFilePath)
{
  ifstream infile(inFilePath);
  
  vector<Result> results;
  
  string line;
  
  bool firstLine = true;
  
  while(getline(infile, line)){
    if(firstLine){
      firstLine = false;
      continue;
    }
    vector<string> lineByCommas = splitByComma(line);
    
    if(lineByCommas.size() != 5) continue;
    
    Result r;
    
    r.name = lineByCommas[1];
    r.AUC = stod(lineByCommas[2]);
    r.mass = stod(lineByCommas[3]);
    r.r_inv = stod(lineByCommas[4]);
    
    results.push_back(r);
  }
  return results;
}


vector<Epoch> ResultsProcessor::getEpochsFromFile(string inFilePath)
{
  ifstream infile(inFilePath);
  
  vector<Epoch> epochs;
  
  string line;
  
  bool firstLine = true;
  
  while(getline(infile, line)){
    if(firstLine){
      firstLine = false;
      continue;
    }
    vector<string> lineByCommas = splitByComma(line);
    
    if(lineByCommas.size() != 4) continue;
    
    Epoch e;
    
    e.trainingLoss = stod(lineByCommas[1]);
    e.learningRate = stod(lineByCommas[2]);
    e.validationLoss = stod(lineByCommas[3]);
    
    epochs.push_back(e);
  }
  return epochs;
}

