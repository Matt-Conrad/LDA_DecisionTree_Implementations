%% CaltechLDA.m
% This script goes through the download and preparation of the Caltech10
% image set, as well as the accuracy calculation for test and train sets. 
clear all; clc;

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
numFeatures = 40;
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
end

% Mix up the test and train matrices
k = randperm(size(calTrainVector,1));
calTrainVector = calTrainVector(k,:);
calTrainLabels = calTrainLabels(k);

k = randperm(size(calTestVector,1));
calTestVector = calTestVector(k,:);
calTestLabels = calTestLabels(k);

% Test the Caltech dataset on the LDA Classifer using the Training Dataset
tic
TestPreds = LDA(calTrainVector,calTrainLabels',calTrainVector);
LDA_time_to_train = toc;
TestLabels = calTrainLabels';
Accuracy = sum(TestPreds==TestLabels)/size(TestPreds,1);
fprintf('The Accuracy of the LDA classifier on the Caltech training dataset is %f percent\n', Accuracy*100);

% Test the Caltech dataset on the LDA Classifer using the Testing Dataset
tic
TestPreds = LDA(calTrainVector,calTrainLabels',calTestVector);
LDA_time_to_train_2 = toc;
TestLabels = calTestLabels';
Accuracy = sum(TestPreds==TestLabels)/size(TestPreds,1);
fprintf('The Accuracy of the LDA classifier on the Caltech testing dataset is %f percent\n', Accuracy*100);
