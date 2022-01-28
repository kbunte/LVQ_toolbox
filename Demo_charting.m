clc; clear;
addpath(genpath('algorithms'));
addpath(genpath('tools'));
%% load the data
[Wine,Labels] = wine_dataset;
[~,c_X] = find(Labels'==1);
X = Wine';
rng(10); % for reproducability
CrossValIdx = cvpartition(c_X,'KFold',5);
%% compute the local LiRaM models
nb_repetitions = 5;
dim = 2;
PrototypesPerClass = 1;
prepros = cell(1,CrossValIdx.NumTestSets);
LGMLVQ_performance = array2table(nan(CrossValIdx.NumTestSets*nb_repetitions,4),'VariableNames',{'fold','rep','trainError','testError'});
LGMLVQ_Models = cell(CrossValIdx.NumTestSets,nb_repetitions);
for fold=1:CrossValIdx.NumTestSets
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:)),'S',std(X(CrossValIdx.training(fold),:)));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=c_X(CrossValIdx.training(fold)); testLab=c_X(CrossValIdx.test(fold));
    for rep=1:nb_repetitions
        rng(rep); % for reproducability
        actModel = LGMLVQ_train(trainX,trainLab,'dim',dim,'PrototypesPerClass',PrototypesPerClass);
% this normalization is important for the scales!
actModel.psis = cellfun(@(omega) omega./trace(omega'*omega),actModel.psis,'uni',0);
        
        LGMLVQ_Models{fold,rep} = actModel;
        estTrainLabs = LGMLVQ_classify(trainX,actModel);
        estTestLabs = LGMLVQ_classify(testX,actModel);
        LGMLVQ_performance((fold-1)*nb_repetitions+rep,:) = array2table([fold,rep,mean(estTrainLabs~=trainLab),mean(estTestLabs~=testLab)]);
    end
end
disp(LGMLVQ_performance);
