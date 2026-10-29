#include "solver/solver.h"
#include "tasks/rank_loss.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace STreeD;
int main(int argc,char** argv){
 try{
  if(argc!=2) throw std::runtime_error("usage rankloss_cli case.txt");
  std::ifstream in(argv[1]);
  if(!in)throw std::runtime_error("cant_open_case");
  int E,P,B,D,L;
  if(!(in>>E>>P>>B>>D>>L))throw std::runtime_error("bad_header");
  if(E<1||E>1000||P<1||P>2048||B<1||D<0||D>5||L<1||L>32)throw std::runtime_error("invalid_limits");
  std::vector<std::vector<int>> features;
  std::vector<int> y, offsets;int unknown=0,targets=0;
  for(int i=0;i<E;i++){
   int valid,target;
   if(!(in>>valid>>target))throw std::runtime_error("bad_rows");
   if((valid!=0&&valid!=1)||(target!=0&&target!=1)||(!valid&&target))throw std::runtime_error("invalid_valid_target");
   std::vector<int> row(P);
   for(int j=0;j<P;j++)if(!(in>>row[j]))throw std::runtime_error("bad_facts");
   if(!valid){for(int x:row)if(x!=-1)throw std::runtime_error("invalid_unknown_not_global");unknown++;}
   else{
    for(int x:row)if(x!=0&&x!=1)throw std::runtime_error("invalid_valid_fact");
    features.push_back(row);y.push_back(target);offsets.push_back(unknown);targets+=target;
   }
  }
  if(features.empty())throw std::runtime_error("no_valid_rows");
  AData data(P);std::vector<std::vector<const AInstance*>> per(2);
  for(int i=0;i<(int)features.size();i++){
   std::vector<bool> bits(P);for(int j=0;j<P;j++)bits[j]=features[i][j]!=0;
   auto* instance=new Instance<int,ExtraData>(i,bits,y[i]);data.AddInstance(instance);per[y[i]].push_back(instance);
  }
  ADataView dv(&data,per,{});
  ParameterHandler params=ParameterHandler::DefineParameters();
  params.SetIntegerParameter("max-depth",D);params.SetIntegerParameter("max-num-nodes",std::min(L-1,(1<<D)-1));
  params.SetIntegerParameter("min-leaf-node-size",1);
  params.SetBooleanParameter("all-trees",false);params.SetBooleanParameter("verbose",false);
  params.SetBooleanParameter("use-terminal-solver",false);params.SetBooleanParameter("use-similarity-lower-bound",false);
  params.SetBooleanParameter("use-task-lower-bound",false);params.SetFloatParameter("time",60.0);
  std::default_random_engine rng(13);
  Solver<RankLoss> solver(params,&rng);
  solver.GetTask()->SetOffsets(offsets,B);
  solver.PreprocessData(data);
  auto st=std::chrono::steady_clock::now();
  auto result=solver.Solve(dv);
  double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count();
  if(!result->IsFeasible())throw std::runtime_error("no_feasible_result");
  std::cout<<"RESULT STATUS="<<(result->IsProvenOptimal()?"OPTIMAL":"TIME_LIMIT")
   <<" RECOVERED="<<result->GetBestScore()->score
   <<" TARGETS="<<targets
   <<" NODES="<<result->GetBestNodeCount()
   <<" DEPTH="<<result->GetBestDepth()
   <<" SECONDS="<<sec<<"\n";
  return result->IsProvenOptimal()?0:3;
 }catch(const std::exception&e){std::cerr<<"ERROR="<<e.what()<<"\n";return 2;}
}
