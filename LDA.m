function [ TestPredictions ] = LDA( TrainFeatures, TrainLabels, TestFeatures)
%LDA.m implements Linear Discriminant Analysis Classifier
%   Inputs: 
%   
%   TrainFeatures - A matrix of size M1xN where M1 is the number of training
%   data images and N is the number of training data features.
%   
%   TrainLabels - A matrix of size M1x1 where M1 is the number of training
%   images.
% 
%   TestFeatures - A matrix of size M2xN where M2 is the number of testing
%   data images and N is the number of training data features.
% 
%   Output:
%   
%   TestPredictions - Returns the predicted class of the Test Features

UniqueClasses = unique(TrainLabels); % Extract the unique class labels
[M1,N] = size(TrainFeatures); % N is the number of features
% Initialize an array of size (Number_Classes x Number_Features) to hold
% the calculated class means.
ClassMeans = Inf.*ones(length(UniqueClasses),N);
Prior = Inf.*ones(length(UniqueClasses),1);
PooledCovariance = zeros(N,N);
for i = 1:length(UniqueClasses)
    CurrentClass = UniqueClasses(i); % Extract the current label
    % Extract the indices of where this class occurs within the data
    CurrentLabelIndices = (TrainLabels==CurrentClass); 
    DataCurrentClass = TrainFeatures(CurrentLabelIndices,:);
    % Estimate the Prior probability of this class occurring
    Prior(i) = size(DataCurrentClass,1)/M1;
    % Calculate the means of each feature for each class
    ClassMeans(i,:) = mean(TrainFeatures(CurrentLabelIndices,:));
    % Next perform the X_i-mu_i step for each class
    X_i = DataCurrentClass-repmat(ClassMeans(i,:),size(DataCurrentClass,1),1);
    % Calculate the pooled covariance
    PooledCovariance = PooledCovariance + (X_i'*X_i);
end

PooledCovariance = PooledCovariance./(M1-N);

% Initialize an array of size (Number_Classes x Number_Features+1) to hold
% the calculated class discriminant function coefficients.
W = NaN(length(UniqueClasses),N+1);
for i = 1:length(UniqueClasses)
   W(i,1) = -1/2 * ClassMeans(i,:)*pinv(PooledCovariance)*ClassMeans(i,:)';
   W(i,2:end) = ClassMeans(i,:)*pinv(PooledCovariance);
end

L = NaN(size(TestFeatures,1),length(UniqueClasses));
for i = 1:size(TestFeatures,1) % For each test point
    for j = 1:length(UniqueClasses) % For each class
        L(i,j) = W(j,1)+log(Prior(j))+TestFeatures(i,:)*W(j,2:end)';
    end
end

% Find the most likely class to which each test point belongs to and assign
% a prediction based on this analysis
[~, maxscoreindex] = max(L,[],2);
TestPredictions = UniqueClasses(maxscoreindex);

end
