clc; clear;
addpath(genpath('algorithms'));
addpath(genpath('tools'));
%% load the data
[Wine,Labels] = wine_dataset;
[~,c_X] = find(Labels'==1);
X = Wine';
rng(100); % for reproducability
CrossValIdx = cvpartition(c_X,'KFold',5);
%% compute the local LiRaM models
nb_repetitions = 5;
prepros = cell(1,CrossValIdx.NumTestSets);
LLiRaMLVQ_performance = array2table(nan(CrossValIdx.NumTestSets*nb_repetitions,4),'VariableNames',{'fold','rep','trainError','testError'});
allModels = cell(CrossValIdx.NumTestSets,nb_repetitions);
for fold=1:CrossValIdx.NumTestSets
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:)),'S',std(X(CrossValIdx.training(fold),:)));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=c_X(CrossValIdx.training(fold)); testLab=c_X(CrossValIdx.test(fold));
    for rep=1:nb_repetitions
        rng(rep); % for reproducability
        actModel = LLiRaMLVQ_train(trainX,trainLab,'dim',2,'optreps',1,'PrototypesPerClass',2,'classwise',1);
        allModels{fold,rep} = actModel;
        estTrainLabs = LLiRaMLVQ_classify(trainX,actModel);
        estTestLabs = LLiRaMLVQ_classify(testX,actModel);
        LLiRaMLVQ_performance((fold-1)*nb_repetitions+rep,:) = array2table([fold,rep,mean(estTrainLabs~=trainLab),mean(estTestLabs~=testLab)]);
    end
end
disp(LLiRaMLVQ_performance);
fprintf('LLiRaMLVQ AVG Training error: %f test error: %f\n',mean(table2array(LLiRaMLVQ_performance(:,[3,4]))));
%% plot an example visualization
fold=2;rep=1;
trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
trainLab=c_X(CrossValIdx.training(fold)); testLab=c_X(CrossValIdx.test(fold));

actModel = allModels{fold,rep};
plotModel([trainX;testX],[trainLab;testLab],actModel);
        
%% helper function
function plotModel(X,c_X,useModel)
    classes = unique(c_X);
    nb_classes = length(classes);
    newX = X*useModel.omega';
    newW = useModel.w*useModel.omega';
    markers = {'v','o','s'};
    colors = [0 0 0;0.4,0.4,0.4;0.8,0.8,0.8];
    figure;hold on;box on;
    arrayfun(@(i) plot(newX(c_X==i,1),newX(c_X==i,2),markers{i},'MarkerSize',3,'MarkerEdgeColor',colors(i,:),'MarkerFaceColor',colors(i,:)),1:nb_classes);
    arrayfun(@(i) plot(newW(useModel.c_w==i,1),newW(useModel.c_w==i,2),'o','MarkerSize',12,'MarkerEdgeColor','k','MarkerFaceColor','w'),1:nb_classes);
    arrayfun(@(i) text(newW(useModel.c_w==i,1),newW(useModel.c_w==i,2),num2str(find(useModel.c_w==i)),'color','k','VerticalAlignment','cap','HorizontalAlignment','center','FontWeight','bold','FontSize',10),1:nb_classes);

    minx = floor(min(newX(:,1)));  maxx = ceil(max(newX(:,1)));
    miny = floor(min(newX(:,2)));  maxy = ceil(max(newX(:,2)));
    nb_points = 500;
    sepx = (maxx-minx)/nb_points;
    sepy = (maxy-miny)/nb_points;
    [XX,YY] = meshgrid(minx:sepx:maxx,miny:sepy:maxy);
    xi = cell2mat(arrayfun(@(i) [XX(:,i),YY(:,i)],1:size(XX,2),'UniformOutput',false)');

    P = size(xi,1);
    dist = zeros(P,length(useModel.c_w));
    for i=1:length(useModel.c_w)
        if length(useModel.c_w)~=length(useModel.psis)
            matIdx = classes==useModel.c_w(i);
        else
            matIdx = i;
        end
        dist(1:P,i) = sum((bsxfun(@minus, xi, newW(i,:))*useModel.psis{matIdx}').^2, 2);
    end
    [~, index] = min(dist,[],2);
    c_xi = useModel.c_w(index);
    
    figure;hold on;box on;
    arrayfun(@(i) plot(xi(c_xi==i,1),xi(c_xi==i,2),'.','MarkerSize',1,'color',colors(i,:)),1:nb_classes);
    
    arrayfun(@(i) plot(newX(c_X==i,1),newX(c_X==i,2),markers{i},'MarkerSize',3,'MarkerEdgeColor','w','MarkerFaceColor',colors(i,:)),1:nb_classes);
    t=arrayfun(@(i) plot(newW(useModel.c_w==i,1),newW(useModel.c_w==i,2),'o','MarkerSize',12,'MarkerEdgeColor','k','MarkerFaceColor','w'),1:nb_classes);
    arrayfun(@(i) text(newW(useModel.c_w==i,1),newW(useModel.c_w==i,2),num2str(find(useModel.c_w==i)),'color','k','VerticalAlignment','middle','HorizontalAlignment','center','FontWeight','bold','FontSize',10),1:nb_classes);
    legend(t(1),'prototypes');
end
