%% MNIST_LDA.m
% This script goes through the preparation of the MNIST image set and
% calculates the accuracy of LDA on the classification of the image set.

clc, clear all, close all

%% Load the datasets

MNIST_raw = csvread('./MNIST/mnist.csv');

%% Visualize a random image in the MNIST image set
i = randi(size(MNIST_raw,1)-1); % This gets a random row from the MNIST dataset
figure;
img = MNIST_raw(i,:);
img_label = img(1);
img = img(2:end); % drop the image label
img = reshape(img, [28,28]);
img = flipud(img); % These steps are just to make the image come out in the proper orientation.
img = rot90(img,3);
imshow(img, [0 255]);
str_title = ['Example of ', int2str(img_label)];
title(str_title);

%% Split the MNIST image set into 50% training and 50% testing sets
k = randperm(size(MNIST_raw,1));
MNIST_train = MNIST_raw(k(1:round(size(MNIST_raw,1)/2)), :);
MNIST_test = MNIST_raw(k((round(size(MNIST_raw,1)/2)+1):end), :);

%% Feature Extraction - MNIST. mnistVec_train and mnistVec_test have the HOG feature vectors
for i = 1:length(MNIST_train)
    image_train = MNIST_train(i,2:end);
    image_test = MNIST_test(i,2:end);
    
    m = 28; n = 28;

    num_cells_rows = 7; % 7x7 for LDA and 4x4 for Decision Tree
    num_cells_cols = 7;
    
    cell_size_m = floor(m/num_cells_rows);
    cell_size_n = floor(n/num_cells_cols);
    
    image_train = reshape(image_train, [28,28]);
    image_train = rot90(flipud(image_train),3); 
    image_test = reshape(image_test, [28,28]);
    image_test = rot90(flipud(image_test),3); % These steps are just to make the image come out in the proper orientation.
    
    if i == 1
        [tempFeatVec,hogVisTrain] = extractHOGFeatures(image_train,'CellSize',[cell_size_m,cell_size_n]);
        mnistVec_train = NaN(length(MNIST_train),length(tempFeatVec));
        mnistVec_train(i,:) = tempFeatVec;
        
        mnistVec_test = NaN(length(MNIST_test),length(tempFeatVec));
        [mnistVec_test(i,:),hogVisTest] = extractHOGFeatures(image_test,'CellSize',[cell_size_m,cell_size_n]);
    else
        [mnistVec_train(i,:),hogVisTrain] = extractHOGFeatures(image_train,'CellSize',[cell_size_m,cell_size_n]);
        [mnistVec_test(i,:),hogVisTest] = extractHOGFeatures(image_test,'CellSize',[cell_size_m,cell_size_n]);
    end
    
    if i == 1
        sideBySide = figure(1);
        subplot(1,2,1); imshow(image_train);
        hold on;
        subplot(1,2,2); plot(hogVisTrain);
        
        overlay = figure(2);
        imshow(image_train);
        hold on;
        plot(hogVisTrain);
%         waitforbuttonpress;
%         clf(sideBySide); clf(overlay);
    end
end

%% Test the MNIST dataset on the LDA Classifer using the Training Dataset
tic
TestPreds = LDA(mnistVec_train,MNIST_train(:,1),mnistVec_train);
LDA_time_to_train = toc;
TestLabels = MNIST_train(:,1);
Accuracy = sum(TestPreds==TestLabels)/size(TestPreds,1);
fprintf('The Accuracy of the LDA classifier on the MNIST training dataset is %f percent\n', Accuracy*100);

%% Test the MNIST dataset on the LDA Classifer using the Testing Dataset
tic
TestPreds = LDA(mnistVec_train,MNIST_train(:,1),mnistVec_test);
LDA_time_to_train_2 = toc;
TestLabels = MNIST_test(:,1);
Accuracy = sum(TestPreds==TestLabels)/size(TestPreds,1);
fprintf('The Accuracy of the LDA classifier on the MNIST testing dataset is %f percent\n', Accuracy*100);
