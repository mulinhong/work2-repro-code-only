#pragma once
#include "tasks/optimization_task.h"
#include <stdexcept>
#include <numeric>

namespace STreeD {
class RankLoss : public Classification {
public:
  using SolType=int;
  using SolD2Type=int;
  using TestSolType=int;
  static const bool total_order=true;
  static const bool custom_leaf=true;
  static const bool use_terminal=false;
  static const bool element_additive=false;
  static const bool custom_similarity_lb=false;
  static const bool custom_lower_bound=false;
  static constexpr int best=0;
  static constexpr int worst=INT32_MAX;
  explicit RankLoss(const ParameterHandler& params):Classification(params) {}
  void SetOffsets(const std::vector<int>& offsets, int top_b) {
    if(top_b<1) throw std::runtime_error("rank_loss_invalid_B");
    for(size_t i=0;i<offsets.size();i++) if(offsets[i]<0 || (i>0 && offsets[i]<offsets[i-1]))
      throw std::runtime_error("rank_loss_offsets_not_monotone");
    unknown_before=offsets; B=top_b;
  }
  void InformTrainData(const ADataView& data, const DataSummary& summary) {
    OptimizationTask::InformTrainData(data,summary);
    if(int(unknown_before.size())!=data.Size())
      throw std::runtime_error("rank_loss_offset_size_mismatch");
    num_targets=0;
    for(int k=0;k<data.NumLabels();k++)
      for(const AInstance* x:data.GetInstancesForLabel(k))
        num_targets+=(GetInstanceLabel<int>(x)!=0);
  }
  int GetLeafCosts(const ADataView& data, const BranchContext&, int) const {
    std::vector<std::pair<int,bool>> ordered;
    ordered.reserve(data.Size());
    for(int k=0;k<data.NumLabels();k++)
      for(const AInstance* instance: data.GetInstancesForLabel(k))
        ordered.emplace_back(instance->GetID(),GetInstanceLabel<int>(instance)!=0);
    std::sort(ordered.begin(),ordered.end());
    int loss=0,prior_valid=0;
    for(auto [id,is_target]:ordered){
      if(is_target && 1+unknown_before.at(id)+prior_valid>B) ++loss;
      ++prior_valid;
    }
    return loss;
  }
  Node<RankLoss> SolveLeafNode(const ADataView& data,const BranchContext& context) const {
    return Node<RankLoss>(0,GetLeafCosts(data,context,0));
  }
  int GetTestLeafCosts(const ADataView& data,const BranchContext& context,int label) const {
    return GetLeafCosts(data,context,label);
  }
  int Classify(const AInstance*,int label) const {return label;}
  double ComputeTrainScore(int loss) const {return num_targets-loss;}
  double ComputeTrainTestScore(int loss) const {return num_targets-loss;}
  double ComputeTestTestScore(int loss) const {return num_targets-loss;}
private:
  std::vector<int> unknown_before;
  int B=1;
  int num_targets=0;
};
}
