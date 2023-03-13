%% File to demonstrate the Matrix LVQ algorithm
clc; clear;
addpath(genpath('algorithms'));
addpath(genpath('tools'));
%% load the data
% [Wine,Labels] = wine_dataset;
load('data/wine_dataset.mat');
[~,c_X] = find(Labels'==1);
X = Wine';
rng(100); % for reproducability
CrossValIdx = cvpartition(c_X,'KFold',5);
projection_dimension = 2;
%% compute the global GMLVQ models
nb_repetitions = 5;
prepros = cell(1,CrossValIdx.NumTestSets);
GMLVQ_performance = array2table(nan(CrossValIdx.NumTestSets*nb_repetitions,4),'VariableNames',{'fold','rep','trainError','testError'});
allModels = cell(CrossValIdx.NumTestSets,nb_repetitions);
for fold=1:CrossValIdx.NumTestSets
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:)),'S',std(X(CrossValIdx.training(fold),:)));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=c_X(CrossValIdx.training(fold)); testLab=c_X(CrossValIdx.test(fold));
    for rep=1:nb_repetitions
        rng(rep); % for reproducability
        actModel = GMLVQ_train(trainX,trainLab,'dim',projection_dimension,'nb_reiterations',1,'PrototypesPerClass',1);
        allModels{fold,rep} = actModel;
        estTrainLabs = GMLVQ_classify(trainX,actModel);
        estTestLabs = GMLVQ_classify(testX,actModel);
        GMLVQ_performance((fold-1)*nb_repetitions+rep,:) = array2table([fold,rep,mean(estTrainLabs~=trainLab),mean(estTestLabs~=testLab)]);
    end
end
disp(GMLVQ_performance);
fprintf('GMLVQ AVG Training error: %f test error: %f\n',mean(table2array(GMLVQ_performance(:,[3,4]))));
%% plot the result of the last model
data = [trainX;testX];
c_data = [trainLab;testLab];
projectedX = GMLVQ_project(data, allModels{1,1}, 2);
usecolor = {'or','sg','vb'};
f1 = figure(1);clf(f1);set(f1, 'color', 'white');hold on;axis square;box on;
arrayfun(@(c) plot(projectedX(c_data==c,1),projectedX(c_data==c,2),usecolor{c}),unique(c_data));
