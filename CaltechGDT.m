%% CaltechGDT.m
% This script goes through the download and preparation of the Caltech10
% image set, as well as the cross-validation to find the optimal
% hyperparameters of the GDT algorithm. It also calculates the accuracy of
% the algorithm. 

clear all; clc;

%% Caltech Data Loading and Bag of Features creation
% This first section is from 
% https://www.mathworks.com/help/vision/examples/image-category-classification-using-bag-of-features.html

% Location of the compressed data set
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

rootFolder = fullfile(outputFolder, '101_ObjectCategories');

% Select classes of the 101 bigger Caltech data set
categories = {'ant','bass','butterfly','camera','chair','crab','dolphin','elephant','sunflower','yin_yang'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

% Split into 2 sets
[calTrainingSet, calValidationSet] = splitEachLabel(imds, 0.5, 'randomize');

% Create a bag of features 
numFeatures = 40; % Empirically determined by LDA/Caltech to be best
bag = bagOfFeatures(calTrainingSet,'StrongestFeatures',0.1,'VocabularySize',numFeatures);

% Encode Images into a feature vector
calTrainVector = zeros(length(calTrainingSet),numFeatures);
calTrainLabels = zeros(length(calTrainingSet),1);
calTestVector = zeros(length(calValidationSet),numFeatures);
calTestLabels = zeros(length(calValidationSet),1);

for i = 1:length(calTrainingSet.Files)
    train_img = readimage(calTrainingSet,i);
    calTrainVector(i,:) = encode(bag,train_img);
    current_label = char(calTrainingSet.Labels(i));
    calTrainLabels(i) = find(strcmp(categories,current_label));
    
    test_img = readimage(calValidationSet,i);
    calTestVector(i,:) = encode(bag,test_img);
    current_label = char(calValidationSet.Labels(i));
    calTestLabels(i) = find(strcmp(categories,current_label));
    
%    % Plot the histogram of visual word occurrences
%     figure(1)
%     bar(featureVector)
%     axis([0 numFeatures 0 1]);
%     title('Visual word occurrences')
%     xlabel('Visual word index')
%     ylabel('Frequency of occurrence')
%     pause(1)
end

% Mix up the test and train matrices
k = randperm(size(calTrainVector,1));
calTrainVector = calTrainVector(k,:);
calTrainLabels = calTrainLabels(k);

k = randperm(size(calTestVector,1));
calTestVector = calTestVector(k,:);
calTestLabels = calTestLabels(k);

%% Cross Validate the Decision Tree Classifier on the Caltech dataset to find the optimal parameters

MaxSplits = [10,20,30,40,50,60,70];
MaxDepths = [3,5,7,9,11,13,15,17,50];
StopCriterias = [0.01,0.1,0.2];
[MaxSplitsOpt, StoppingCriteriaOpt, MaxDepthOpt] = CrossValidateGDT(calTrainVector,calTrainLabels',...
    MaxSplits, StopCriterias, MaxDepths, 5);


MaxSplitsOpt, StoppingCriteriaOpt, MaxDepthOpt

tic
TestLabels = calTestLabels(1,:)';
TestPreds = GreedyDecisionTree2(calTrainVector, calTrainLabels',...
    calTestVector,MaxSplitsOpt,StoppingCriteriaOpt, MaxDepthOpt, 10);
TIMEFORTHIS = toc
Accuracy = sum(TestPreds==TestLabels)/size(TestPreds,1);
fprintf('The Accuracy of the Decision Tree classifier on the Caltech dataset is %f percent\n', Accuracy*100);

%% This is the calculation of training and testing accuracy of Caltech using the optimal parameters

TrainLabels = calTrainLabels';
TestLabels = calTestLabels';
MinLeafSize = 10;

TestPreds = GreedyDecisionTree2(calTrainVector, TrainLabels, calTestVector, MaxSplitsOpt, StoppingCriteriaOpt, MaxDepthOpt, MinLeafSize);
AccuracyTest = sum(TestPreds==TestLabels)/size(TestPreds,1);
fprintf('The Accuracy of the Decision Tree classifier on the Caltech10 testing dataset is %f percent\n', AccuracyTest*100);

TrainPreds = GreedyDecisionTree2(calTrainVector, TrainLabels, calTrainVector, MaxSplitsOpt, StoppingCriteriaOpt, MaxDepthOpt, MinLeafSize);
AccuracyTrain = sum(TrainPreds==TrainLabels)/size(TrainPreds,1);
fprintf('The Accuracy of the Decision Tree classifier on the Caltech10 training dataset is %f percent\n', AccuracyTrain*100);



