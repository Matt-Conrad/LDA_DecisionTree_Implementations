function [ impurity, MajorityClass ] = Impurity( DataClasses, UniqueClasses)
%Impurity.m - This function calculates the impurity, a measure of node
%impurity based on entropy, for a given node.
%
% Inputs: 
%
% DataClasses - The Nm x 1 vector of classes to which the training data in
% this node belong to.
% 
% UniqueClasses - The vector of unique classes in the training dataset
% 
% Output:
% 
% impurity - A measure of node impurity (zero is a pure node)
% 
% MajorityClass - Majority class of the node

N = length(DataClasses);
if (N == 0) % Make sure the node isn't a null set of training points
    impurity = 0;
    MajorityClass = NaN;
else
    impurity = 0;
    P_max = 0;
    MajorityClass = Inf;
    % Cycle through the classes and add to the impurity measure if the
    % class exists in the node's data set
    for i = 1:length(UniqueClasses)
        w_i = UniqueClasses(i);
        if (ismember(w_i,DataClasses))
            Ni = length( DataClasses( w_i == DataClasses ) );
            P = Ni/N;
            impurity = impurity - (P*log2(P));
            if P > P_max
                P_max = P;
                MajorityClass = w_i;
            end
        end
    end
end

end
