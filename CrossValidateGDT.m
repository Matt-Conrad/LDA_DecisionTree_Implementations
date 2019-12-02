function [ MaxSplitsOpt, StoppingCriteriaOpt, MaxDepthOpt ] = ...
    CrossValidateGDT( TrainFeatures, TrainLabels,...
    MaxSplits, StoppingCriteria, MaxDepth, k )
%CROSSVALIDATEGDT.m is a cross-validation wrapper function to determine the
%best hyperparameters of the training dataset.
% 
%   Inputs: 
%   
%   TrainFeatures - A matrix of size M1xN where M1 is the number of training
%   data images and N is the number of training data features.
%   
%   TrainLabels - A matrix of size M1x1 where M1 is the number of training
%   images.
% 
%   MaxSplits - The maximum number of splits that can occur in the building
%   of a tree. The input is a vector of MaxSplits which are tested
%   sequentially to tell which one yields the highest accuracy on the
%   k-fold splits of the training data.
% 
%   StoppingCriteria - The value at which the tree is said to have
%   converged. If the L1 difference in GiniCost from parent to child nodes
%   is very small the algorithm is said to have converged. The input is a 
%   vector of MaxSplits values which are tested sequentially to tell which 
%   one yields the highest accuracy on the k-fold splits of the training
%   data.
% 
%   MaxDepth - The maximum depth that a tree can grow before nodes at that
%   depth are no longer selected for splitting. The input is a vector of 
%   MaxDepth values which are tested
%   sequentially to tell which one yields the highest accuracy on the
%   k-fold splits of the training data.
% 
%   k - The number of folds in your cross-validation. Must be an integer
%   between 2 and Infinity.
% 
%   Output:
%   
%   MaxSplitsOpt, StoppingCriteriaOpt, MaxDepthOpt - The optimal
%   hyperparameters found during cross-validation.

% First separate the data into k-folds

RandomSplit = randperm(size(TrainFeatures,1));

Results = zeros(k, length(MaxSplits), length(StoppingCriteria), length(MaxDepth));
AveragedResults = ones(length(MaxSplits), length(StoppingCriteria), length(MaxDepth));
for i = 1:k
    TestF = TrainFeatures(RandomSplit(1:round(end/k)),:);
    TrainF = TrainFeatures(RandomSplit(round(end/k)+1:end),:);
    TestL = TrainLabels(RandomSplit(1:round(end/k)),:);
    TrainL = TrainLabels(RandomSplit(round(end/k)+1:end),:);
    % Now reorder the Random Split Matrix so that the next fold takes in
    % the next random 1/k proportion of the data
    RandomSplit = [RandomSplit(round(end/k)+1:end), RandomSplit(1:round(end/k))];
    
    for j = 1:length(MaxSplits)
        for p = 1:length(StoppingCriteria)
            for z = 1:length(MaxDepth)
                fprintf('%f,%f,%f', MaxSplits(j), StoppingCriteria(p), MaxDepth(z));
                TestPreds = GreedyDecisionTree2(TrainF, TrainL,TestF, ...
                MaxSplits(j), StoppingCriteria(p), MaxDepth(z),10);
                Accuracy = sum(TestPreds==TestL)/size(TestPreds,1);
                Results(i,j,p,z) = Accuracy;
            end
        end
    end    
end
for j = 1:length(MaxSplits)
    for p = 1:length(StoppingCriteria)
        for z = 1:length(MaxDepth)
            AveragedResults(j,p,z) = mean(Results(:,j,p,z));
        end
    end
end    
% All that is left is to average over each fold's results to find the most
% amazing hyperparameters.
MaxSplitAccuracy = 0;
for j = 1:length(MaxSplits)
    if max(max(AveragedResults(j,:,:))) > MaxSplitAccuracy
        MaxSplitsOpt = MaxSplits(j);
        MaxSplitAccuracy = max(max(AveragedResults(j,:,:)));
    end
end
BestStoppingCriteriaAccuracy = 0;
for p = 1:length(StoppingCriteria)
    if max(max(AveragedResults(:,p,:))) > BestStoppingCriteriaAccuracy
        StoppingCriteriaOpt = StoppingCriteria(p);
        BestStoppingCriteriaAccuracy = max(max(AveragedResults(:,p,:)));
    end
end
BestDepthAccuracy = 0;
for z = 1:length(MaxDepth)
    if max(max(AveragedResults(:,:,z))) > BestDepthAccuracy
        BestDepthAccuracy = max(max(AveragedResults(:,:,z)));
        MaxDepthOpt = MaxDepth(z);
    end
end
MaxSplitAccuracy
BestStoppingCriteriaAccuracy
BestDepthAccuracy
end
