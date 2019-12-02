function [ TestPredictions, Convergence ] = GreedyDecisionTree(TrainFeatures, TrainLabels,...
    TestFeatures, MaxSplits, StoppingCriteria, MaxDepth, MinSplitSize)
%   GreedyDecisionTree.m 
% 
%   Inputs: 
%   
%   TrainFeatures - A matrix of size M1xN where M is the number of training
%   data images and N is the number of training data features.
%   
%   TrainLabels - A matrix of size M1x1 where M is the number of training
%   images.
% 
%   TestFeatures - A matrix of size M2xN where M2 is the number of testing
%   data images and N is the number of test data features.
% 
%   MaxSplits - The maximum number of splits that can occur in the building
%   of a tree
% 
%   StoppingCriteria - The value at which the tree is said to have
%   converged. If the L1 difference in GiniCost from parent to child nodes
%   is very small the algorithm is said to have converged.
% 
%   MaxDepth - The maximum depth that a tree can grow before nodes at that
%   depth are no longer selected for splitting. 
%
%   MinSplitSize - The minimum number of points in a node for us to split
%   it. We don't want to keep splitting nodes of insignificant size just
%   because they are the largest decrease in impurity
% 
%   Output:
%   
%   TestPredictions - Returns the predicted class of the Test Features

Convergence = []; % Log of impurity decrease for successive splits 
% N is the number of features, M1 is number of observations
[M1,N] = size(TrainFeatures);
% The unique class labels within the training set
UniqueClasses = unique(TrainLabels);

% First let's initialize the TreeDataStructure
% It will consist of 1+2*NumSplits Nodes
% test point. and is a cell of structs which
% contain information about how to traverse the tree when classifying a
TreeDataStructure = cell(1+2*(MaxSplits),1);
EmptyNode.right = NaN;
EmptyNode.left = NaN;
EmptyNode.terminal = NaN;
EmptyNode.parent = NaN;
EmptyNode.feature = NaN;
EmptyNode.threshold = NaN;
EmptyNode.class = NaN;
EmptyNode.members = NaN;
EmptyNode.depth = NaN;
EmptyNode.NumPoints = NaN;
EmptyNode.Distribution = NaN;

for i = 1:length(TreeDataStructure)
    TreeDataStructure{i} = EmptyNode;
end

% Initialize Root Node
RootNode = TreeDataStructure{1}; % Checkout top node
RootNode.parent = Inf; % You have no parent you are the root node
RootNode.members = logical(ones(M1,1));
RootNode.terminal = 1; % The root node is currently the sole terminal node
[~,MajClass] = Impurity(TrainLabels(RootNode.members), UniqueClasses);
RootNode.class = MajClass;
RootNode.depth = 1; % Initial depth of the tree is 1
RootNode.NumPoints = length(RootNode.members);
% Distribution of the node based on class
xbins = min(UniqueClasses):max(UniqueClasses);
[RootNode.Distribution,~] = hist(TrainLabels(RootNode.members),xbins);
TreeDataStructure{1} = RootNode; % Put it back into the data structure

SplitCount = 0;
BestDecImpurityAmongTermNodes = Inf;
ResultsMatrix = []; % Initialize the Results Matrix as an empty matrix of all zeros.
% available. meaning that they do not exceed the max_depth and are not
% parents. 
NodesToCheck = 1;
Depth = 0; % Used the count the deepest the tree goes

while ((SplitCount < MaxSplits) && (BestDecImpurityAmongTermNodes >= StoppingCriteria))
    % Now that we have the terminal nodes let's find the best split
    % at each and then perform said split on the best performing node
    
    for i = 1:length(NodesToCheck) 
        % Here you check to see if the terminal node exceeds the MaxDepth
        % of the tree
        if (TreeDataStructure{NodesToCheck(i)}.depth < MaxDepth)
            
            if (TreeDataStructure{NodesToCheck(i)}.depth > Depth)
                Depth = TreeDataStructure{NodesToCheck(i)}.depth;
            end
            
            TerminalNode = TreeDataStructure{NodesToCheck(i)};
            TermNodeTrainFeatures = TrainFeatures(TerminalNode.members,:);
            TermNodeTrainLabels = TrainLabels(TerminalNode.members);
            [TermTrainPts, ~] = size(TermNodeTrainFeatures);

            % Next, calculate the subset of training points and features for 
            % each terminal node if we are above the max number allowed 
            % (because I want this code to run before the universe ends)

            if TermTrainPts > 210 %100 for MNIST % If there are a very large amount of training points
                TreeS = randperm(TermTrainPts);
                TreeS = TreeS(1:210); % Select random subset of training points
            else
                TreeS = 1:TermTrainPts; % Select all training points
            end
            if N > 40 % 20 for MNIST % If there are a very large amount of features
                TreeJ = randperm(N);
                TreeJ = TreeJ(1:40); % Select random subset of features
            else
                TreeJ = 1:N; % Select All Features 
            end
            
            % Reserve space for recording the best node 
            BestDecImpurityThisNode = 0;
            BestSplitPointS = NaN;
            BestSplitFeatureJ = NaN;
            BestSplitThreshold = NaN;
            BestParentImpurity = NaN;
            BestLeftImpurity = NaN;
            BestRightImpurity = NaN;
            
            NumTotal = length(TermNodeTrainLabels);
            if (NumTotal >= MinSplitSize) % This ensures we don't consider nodes that are too small
                for j = 1:length(TreeJ) % For each Feature
                    for s = 1:length(TreeS) % For each split point
                        DataThreshold = TermNodeTrainFeatures(TreeS(s), TreeJ(j));
                        % Data which goes to the right
                        DataR = TermNodeTrainFeatures(:,TreeJ(j))>=DataThreshold; 
                        DataL = ~DataR; % Data which goes to the left
                        DataClassesR = TermNodeTrainLabels(DataR);
                        DataClassesL = TermNodeTrainLabels(DataL);
                        % Calculate the decrease in impurity for this split
                        [ImpurityR, ~] = Impurity(DataClassesR, UniqueClasses);
                        [ImpurityL, ~] = Impurity(DataClassesL, UniqueClasses);
                        [ImpurityP, ~] = Impurity(TermNodeTrainLabels, UniqueClasses);
                        NumRight = length(DataClassesR);
                        NumLeft = length(DataClassesL);
                        DecInGiniCost = ImpurityP-NumRight/NumTotal*ImpurityR-NumLeft/NumTotal*ImpurityL;
                        % Log the split information as best if it is best
                        if ( (DecInGiniCost >= BestDecImpurityThisNode) )
                            BestDecImpurityThisNode = DecInGiniCost;
                            BestSplitPointS = TreeS(s);
                            BestSplitFeatureJ = TreeJ(j);
                            BestSplitThreshold = DataThreshold;
                            BestParentImpurity = ImpurityP;
                            BestLeftImpurity = ImpurityL;
                            BestRightImpurity = ImpurityR;
                        end
                    end
                end

                NewResults = NaN(1,8);
                % Calculates the change in impurity between the parent node
                % and best found split
                NewResults(1) = BestDecImpurityThisNode; % Best decrease in impurity
                NewResults(2) = BestSplitPointS;
                NewResults(3) = BestSplitFeatureJ;
                NewResults(4) = BestSplitThreshold;
                NewResults(5) = NodesToCheck(i); % Pointer to the parent node of the proposed split
                NewResults(6) = BestParentImpurity;
                NewResults(7) = BestLeftImpurity;
                NewResults(8) = BestRightImpurity;
                ResultsMatrix = [ResultsMatrix; NewResults]; %#ok<AGROW>
            end
        end
    end
    % Now we have the best split for the current terminal node of interest
    
    % In the case we have a fully split tree at the max depth, there won't
    % be anything in the ResultsMatrix so we have to break out of this loop
    % before it tries to access the ResultsMatrix
    if (isempty(ResultsMatrix))
        break;
    end
    
    % Find the Terminal Node which maximizes the change in impurity between
    % parent node and best split and then assign this one as the one to be
    % split.
    [~,BestNodeIndex] = max(ResultsMatrix,[],1);
    BestDecImpurityAmongTermNodes = ResultsMatrix(BestNodeIndex(1),1);
    Convergence = [Convergence BestDecImpurityAmongTermNodes]; %#ok<AGROW>
    ParentNodeIndex = ResultsMatrix(BestNodeIndex(1),5);
    ParentNode = TreeDataStructure{ParentNodeIndex};
    ParentNode.terminal = 0; % This node is not a terminal node anymore
    ParentNode.feature = ResultsMatrix(BestNodeIndex(1),3);
    ParentNode.threshold = ResultsMatrix(BestNodeIndex(1),4);
    
    % Find free nodes in the Tree Data Structure
    for i = 1:length(TreeDataStructure)
        if isnan(TreeDataStructure{i}.parent)
            ParentNode.right = i;
            ParentNode.left = i+1;
            break;
        end
    end
    
    % Update the Tree Data Structure with the information from the new
    % parent node
    TreeDataStructure{ResultsMatrix(BestNodeIndex(1),5)} = ParentNode;
    
    % Calculate the values for the right and left child nodes
    NodeR = TreeDataStructure{ParentNode.right};
    NodeL = TreeDataStructure{ParentNode.left};
    NodeR.terminal = 1;
    NodeL.terminal = 1;
    NodeR.parent = ResultsMatrix(BestNodeIndex(1),5);
    NodeL.parent = ResultsMatrix(BestNodeIndex(1),5);
    NodeL.depth = ParentNode.depth + 1;
    NodeR.depth = ParentNode.depth + 1;
    
    parentmembers = ParentNode.members;
    featureindex = ParentNode.feature;
    MembersRight = TrainFeatures(:,featureindex) >= ParentNode.threshold;
    MembersLeft = ~MembersRight;
    
    NodeL.members = MembersLeft & parentmembers;
    NodeR.members = MembersRight & parentmembers;
    SumL = sum(double(NodeL.members));
    SumR = sum(double(NodeR.members));
    NodeL.NumPoints = SumL;
    NodeR.NumPoints = SumR;
    [NodeL.Distribution,~] = hist(TrainLabels(NodeL.members),xbins);
    [NodeR.Distribution,~] = hist(TrainLabels(NodeR.members),xbins);
    
    [~,ClassR] = Impurity(TrainLabels(NodeR.members),UniqueClasses);
    [~,ClassL] = Impurity(TrainLabels(NodeL.members),UniqueClasses);
    
    NodeL.class = ClassL;
    NodeR.class = ClassR;
    
    % Put the new nodes (after the split) in the data structure
    TreeDataStructure{ParentNode.right} = NodeR;
    TreeDataStructure{ParentNode.left} = NodeL;
    
    NodesToCheck = [NodesToCheck, ParentNode.right, ParentNode.left];
    % Remove old nodes
    NodesToCheck = NodesToCheck(NodesToCheck~=ResultsMatrix(BestNodeIndex(1),5));
    
    ResultsMatrixRemoval = ResultsMatrix(:,5) ~= ResultsMatrix(BestNodeIndex(1),5);
    ResultsMatrix = ResultsMatrix(ResultsMatrixRemoval, :);
    
    % To prevent early terminations
    if BestDecImpurityAmongTermNodes < 0 && (SplitCount < 3)
        BestDecImpurityAmongTermNodes = abs(BestDecImpurityAmongTermNodes);
    end
    if ~isempty(ResultsMatrix)
        fprintf('-----------CURRENT SPLIT-----------\n');
        fprintf('Parent Node: %d | Left Node: %d | Right Node: %d\n',ParentNodeIndex,ParentNode.left,ParentNode.right);
        fprintf('Number on Left: %d | Number on Right: %d\n',SumL,SumR);
%         fprintf('Parent Distribution: %d | %d | %d | %d | %d | %d | %d | %d | %d | %d IMPURITY:%f\n',ParentNode.Distribution,ResultsMatrix(BestNodeIndex(1),6)); 
%         fprintf('Left Distribution:   %d | %d | %d | %d | %d | %d | %d | %d | %d | %d IMPURITY:%f\n',NodeL.Distribution,ResultsMatrix(BestNodeIndex(1),7)); 
%         fprintf('Right Distribution:  %d | %d | %d | %d | %d | %d | %d | %d | %d | %d IMPURITY:%f\n',NodeR.Distribution,ResultsMatrix(BestNodeIndex(1),8)); 
        fprintf('Impurity Decrease: %f\n',BestDecImpurityAmongTermNodes);
        fprintf('Current Depth: %d\n',ParentNode.depth); 
    end
    if isempty(NodesToCheck)
        break; % Break the loop no more good nodes to check
    end
    
    SplitCount = SplitCount + 1;
end

% Next, evaluate on the training data
TestPredictions = NaN(size(TestFeatures,1),1);
for i = 1:size(TestFeatures,1)
    DataPoint = TestFeatures(i,:);
    Node = TreeDataStructure{1}; % Start at the root
    while Node.terminal ~= 1
        featureindex = Node.feature;
        if (DataPoint(featureindex)>=Node.threshold) % Move to the right node
            Node = TreeDataStructure{Node.right};
        else
            Node = TreeDataStructure{Node.left};
        end
    end
    TestPredictions(i) = Node.class;
end
ScalingFactor = zeros(1,MaxDepth);
for i = 1:length(TreeDataStructure)
    if ~isnan(TreeDataStructure{i}.depth)
        ScalingFactor(TreeDataStructure{i}.depth)=ScalingFactor(TreeDataStructure{i}.depth)+1; 
    end
end
Convergence = zeros(1,MaxDepth);
for i = 1:length(TreeDataStructure)
    if ~isnan(TreeDataStructure{i}.depth)
        Convergence(TreeDataStructure{i}.depth) = Convergence(TreeDataStructure{i}.depth) + ...
            Impurity(TrainLabels(TreeDataStructure{i}.members),UniqueClasses)/...
            ScalingFactor(TreeDataStructure{i}.depth);
    end
end

fprintf('Number of Nodes in Tree: %d\n', SplitCount*2+1);
fprintf('Number of Splits in Tree: %d\n', SplitCount);
fprintf('Furthest depth achieved: %d\n', Depth);

end
