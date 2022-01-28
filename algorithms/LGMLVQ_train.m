function [model, varargout] = LGMLVQ_train(trainSet, trainLab, varargin)
%LGMLVQ_trai.m - trains the Localized Generalized Matrix LVQ algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  LGMLVQ_model=LGMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = LGMLVQ_classify(trainSet, LGMLVQ_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
%  trainLab : vector with the labels of the training set
% optional parameters:
%  PrototypesPerClass: (default=1) the number of prototypes per class used. This could
%   be a number or a vector with the number for each class
%  initialPrototypes : (default=[]) a set of prototypes to start with. If not given initialization near the class means
%  initialMatrices   : the matrices psis{} to start with. If not given random initialization.
%  classwise         : (default=0) set this to 1 if you want only classwise local matrices
%  dim               : (default=nb of features for training) the maximum rank or projection dimension of each matrix. 
%   This can be one value <= the number of features or a vector containing
%   the projection dimension for each matrices psis
%  regularization    : (dealgorithmsfault=0) values usually between 0 and 1 treat with care. 
%   Regularizes the eigenvalue spectrum of psis{i}'*psis{i} to be more homogeneous.
%   This can be one value or a vector containing the regularization for each matrix psis{i}.
%  testSet           : (default=[]) an optional test set used to compute
%   the test error. The last column is expected to be a label vector
%  comparable        : (default=0) a flag which resets the random generator
%   to produce comparable results if set to 1
%  optimization      : (default=fminlbfgs) indicates which optimization is used: sgd or fminlbfgs
% parameter for the stochastic gradient descent sgd
%  nb_epochs             : (default=100) the number of epochs for sgd
%  learningRatePrototypes: (default=[]) the learning rate for the prototypes. 
%   Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  learningRateMatrix    : (default=[]) the learning rate for the matrix.
%   Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  MatrixStart           : (default=1) the epoch to start the matrix training
% parameter for the build-in function fminlbfgs
%  threshstop       : (default=0) the training error for early stopping
%  nb_reiterations  : (default=100) the number of optimization reiterations performed
%  useEarlyStopping : (default=1) use early stopping based on threshstop
%  Display          : (default=iter) the optimization output 'iter' or 'off'
%  GradObj          : (default=on) use the gradient information or not@(x)(~(x-floor(x)) || sum(~(x-floor(x)))==length(unique(trainLab))));
%  HessUpdate       : (default=lbfgs) the update can be 'lbfgs', 'bfgs' or 'steepdesc'
%  TolFun           : (default=1e-6) the tolerance
%  MaxIter          : (default=2500) the maximal number of iterations
%  MaxFunEvals      : (default=1000000) the maximal number of function evaluations
%  TolX             : (default=1e-10) tolerance
%  DiffMinChange    : (default=1e-10) minimal change
%
% output: the LGMLVQ model with prototypes w their labels c_w and the matrices psis 
%  optional output:
%  initialization : a struct containing the settings
%  trainError     : error in the training set
%  testError      : error in the training set (only computed if 'testSet' is given)
%  costs          : the output of the cost function
% 
% Citation information:
% Petra Schneider, Michael Biehl, Barbara Hammer: 
% Adaptive Relevance Matrices in Learning Vector Quantization. Neural Computation 21(12): 3532-3561 (2009)
% 
% K. Bunte, P. Schneider, B. Hammer, F.-M. Schleif, T. Villmann and M. Biehl, 
% Limited Rank Matrix Learning - Discriminative Dimension Reduction and Visualization, 
% Neural Networks, vol. 26, nb. 4, pp. 159-173, 2012.
% 
% P. Schneider, K. Bunte, B. Hammer and M. Biehl, Regularization in Matrix Relevance Learning, 
% IEEE Transactions on Neural Networks, vol. 21, nb. 5, pp. 831-840, 2010.
% 
% Kerstin Bunte
% uses the Fast Limited Memory Optimizer fminlbfgs.m written by Dirk-Jan Kroon available at the MATLAB central
% kerstin.bunte@googlemail.com
% Fri Nov 09 14:13:52 CEST 2012
%
% Conditions of GNU General Public License, version 2 and BSD License apply.
% See file 'license-gpl2.txt' and 'BSD_license.txt' enclosed in this package.
% Programs are not for use in critical applications!
%
nout = max(nargout,1)-1;
p = inputParser;   % Create an instance of the class.
p.addRequired('trainSet', @isfloat);
p.addRequired('trainLab', @(x) length(x)==size(trainSet,1) & isnumeric(x));

p.addParameter('PrototypesPerClass', ones(1,length(unique(trainLab))), @(x)(sum(~(x-floor(x)))/length(x)==1 && (length(x)==length(unique(trainLab)) || length(x)==1)));
p.addParameter('initialPrototypes',[], @(x)( size(x,2)-1==size(trainSet,2) && isfloat(x)) );
p.addParameter('initialMatrices',{}, @(x)(iscell(x)));
p.addOptional('classwise',0, @(x)(~(x-floor(x))));
p.addOptional('dim',size(trainSet,2), @(x)(sum(~(x-floor(x)))==length(x) || isvector(x)));
p.addOptional('regularization',0, @(x)(isfloat(x) && sum(x>=0)==length(x)));
p.addParameter('testSet', [], @(x)(size(x,2)-1)==size(trainSet,2) & isfloat(x));
p.addOptional('comparable', 0, @(x)(~(x-floor(x))));
p.addOptional('optreps',1,@(x)(~(x-floor(x))));
p.addOptional('optimization', 'fminlbfgs', @(x)any(strcmpi(x,{'sgd','fminlbfgs'})));
% parameter for the stochastic gradient descent
p.addOptional('nb_epochs', 100, @(x)(~(x-floor(x))));
p.addOptional('learningRatePrototypes', [], @(x)(isfloat(x) || isa(x,'function_handle'))); %  && (length(x)==2 || length(x)==p.Results.epochs))
p.addOptional('learningRateMatrix', [], @(x)(isfloat(x) || isa(x,'function_handle')));
p.addOptional('MatrixStart', 1, @(x)(~(x-floor(x))));
% parameter for the build-in function
p.addOptional('threshstop',0,@(x) isfloat(x));
p.addOptional('nb_reiterations',100,@(x)(~(x-floor(x))));
p.addOptional('useEarlyStopping',1,@(x)(~(x-floor(x))));
p.addOptional('Display', 'iter', @(x)any(strcmpi(x,{'iter','off'})));
p.addOptional('GradObj', 'on', @(x)any(strcmpi(x,{'on','off'})));
p.addOptional('HessUpdate', 'lbfgs', @(x)any(strcmpi(x,{'lbfgs', 'bfgs', 'steepdesc'})));
p.addOptional('TolFun',1e-6,@(x) isfloat(x));
p.addOptional('MaxIter', 2500, @(x)(~(x-floor(x))));
p.addOptional('MaxFunEvals', 1000000, @(x)(~(x-floor(x))));
p.addOptional('TolX',1e-10,@(x) isfloat(x));
p.addOptional('DiffMinChange',1e-10,@(x)isfloat(x));
p.CaseSensitive = true;
p.FunctionName = 'LGMLVQ';
% Parse and validate all input arguments.
p.parse(trainSet, trainLab, varargin{:});

%%% check if results should be comparable
if p.Results.comparable, rng('default');end
%%% set useful variables
[nb_samples,nb_features] = size(trainSet);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end

classes = unique(trainLab);
nb_classes = length(classes);
classwise = p.Results.classwise;
dim = p.Results.dim;
MatrixStart = p.Results.MatrixStart;
testSet = p.Results.testSet;
% global regularization;
regularization = p.Results.regularization;
if regularization, disp(['Regularize the eigenvalue spectrum of the matrices with ',num2str(regularization)]);end

initialization = rmfield(p.Results, 'trainSet');
initialization.trainSet = [num2str(nb_samples),'x',num2str(nb_features),' matrix'];
initialization = rmfield(initialization, 'trainLab');
initialization.trainLab = ['vector of length ',num2str(length(trainLab))];
if ~isempty(testSet)
    initialization = rmfield(initialization, 'testSet');
    initialization.testSet = [num2str(size(testSet,1)),'x',num2str(size(testSet,2)),' matrix'];
end
switch(p.Results.optimization)
case{'sgd'}
    initialization = rmfield(initialization, 'useEarlyStopping');
    initialization = rmfield(initialization, 'Display');
    initialization = rmfield(initialization, 'GradObj');
    initialization = rmfield(initialization, 'HessUpdate');
    initialization = rmfield(initialization, 'TolFun');
    initialization = rmfield(initialization, 'MaxIter');
    initialization = rmfield(initialization, 'MaxFunEvals');
    initialization = rmfield(initialization, 'TolX');
    initialization = rmfield(initialization, 'DiffMinChange');
    initialization = rmfield(initialization, 'nb_reiterations');
case{'fminlbfgs'}
    disp('The fminlbfgs optimization uses some global variables:');
    disp('threshstop earlystopped useEarlyStopping');
    initialization = rmfield(initialization, 'nb_epochs');
    initialization = rmfield(initialization, 'learningRatePrototypes');
    initialization = rmfield(initialization, 'learningRateMatrix');
    initialization = rmfield(initialization, 'MatrixStart');
    nb_reiterations = p.Results.nb_reiterations;
end
% Display all arguments.
disp 'Settings for LGMLVQ:'
disp(initialization);

%%% check the number of prototypes per class if one integer is given and turn
%%% it into a vector
nb_ppc = p.Results.PrototypesPerClass;
if length(nb_ppc)~=nb_classes
    nb_ppc = ones(1,nb_classes)*nb_ppc;
end

%%% initialize the prototypes
if isempty(p.Results.initialPrototypes)
    % initialize near the class centers
    w = zeros(sum(nb_ppc),nb_features);
    c_w = zeros(sum(nb_ppc),1);
    actPos = 1;
    for actClass=1:nb_classes
        nb_prot_c = nb_ppc(actClass);
        classMean = mean(trainSet(trainLab==classes(actClass),:));
        % set the prototypes to the class mean and add a random variation between -0.1 and 0.1
        w(actPos:actPos+nb_prot_c-1,:) = classMean(ones(nb_prot_c,1),:)+(rand(nb_prot_c,nb_features)*2-ones(nb_prot_c,nb_features))/10;
        c_w(actPos:actPos+nb_prot_c-1) = classes(actClass);
        actPos = actPos+nb_prot_c;
    end
else
    % initialize with given w
    w = p.Results.initialPrototypes(:,1:end-1);
    c_w = p.Results.initialPrototypes(:,end);
end
%%% initialize the matrices
initializeRandom = 0;
if ~isempty(p.Results.initialMatrices)
    if classwise
        if length(p.Results.initialMatrices)~=nb_classes
            disp(['Error: classwise requested, but ',num2str(length(p.Results.initialMatrix)),' given matrices for ',...
                   num2str(nb_classes),' classes -> use random initialization.']);
            initializeRandom = 1;
        else
            psis = p.Results.initialMatrices;
            dim = cellfun(@(x) size(x,1), psis);
            if sum(cellfun(@(x) size(x,2), psis)) ~= nb_features*length(psis)
                disp(['Error: each matrix should have ',num2str(nb_features),' columns -> use random initialization.']);
                initializeRandom = 1;
            end
        end
    else % not classwise
        if length(p.Results.initialMatrices)~=size(w,1)
            disp(['Error: ',num2str(length(p.Results.initialMatrix)),' given matrices for ',num2str(size(w,1)),...
                  ' prototypes -> use random initialization.']);
            initializeRandom = 1;
        else
            psis = p.Results.initialMatrices;
            dim = cellfun(@(x) size(x,1), psis);
            if sum(cellfun(@(x) size(x,2), psis)) ~= nb_features*length(psis)
                disp(['Error: each matrix should have ',num2str(nb_features),' columns -> use random initialization.']);
                initializeRandom = 1;
            end
        end
    end
else 
    initializeRandom = 1;
end
if initializeRandom
    if classwise
        if length(dim)==1
            dim = dim*ones(1,nb_classes);
        else
            if length(dim)>nb_classes
                dim = dim(1:nb_classes);
            else
                dim = [dim,nb_features*ones(1,nb_classes-length(dim))];
            end
        end
    else % not classwise
        if length(dim)==1
            dim = dim*ones(1,size(w,1));
        else
            if length(dim)>size(w,1)
                dim = dim(1:size(w,1));
            else
                dim = [dim,nb_features*ones(1,size(w,1)-length(dim))];
            end
        end        
    end
    psis = cell(1,length(dim));
    for i=1:length(dim)
        psis{i} = rand(dim(i),nb_features)*2-ones(dim(i),nb_features);
    end
end
if length(regularization)==1
    regularization = regularization*ones(1,length(psis));
else
    if classwise
        if length(regularization)>nb_classes
            regularization = regularization(1:nb_classes);
        end
        if length(regularization)<nb_classes
            regularization = [regularization,regularization(1)*ones(1,nb_classes-length(regularization))];
        end
    else
        if length(regularization)>size(w,1)
            regularization = regularization(1:size(w,1));
        end
        if length(regularization)<size(w,1)
            regularization = [regularization,regularization(1)*ones(1,size(w,1)-length(regularization))];
        end
    end
end
model = struct('w',w,'c_w',c_w,'psis',[]);
model.psis = psis;
initialization.dim = dim;
initialization.regularization = regularization;

clear w c_w psis;

switch(p.Results.optimization)
case{'sgd'}
    %%% gradient descent variables
    nb_epochs = p.Results.nb_epochs;
    % compute the vector of nb_epochs learning rates alpha for the prototype learning
    if isa(p.Results.learningRatePrototypes,'function_handle')
        % with a given function specified from the user
        alphas = arrayfun(p.Results.learningRatePrototypes, 1:nb_epochs);
    elseif length(p.Results.learningRatePrototypes)>2
        if length(p.Results.learningRatePrototypes)==nb_epochs
            alphas = p.Results.learningRatePrototypes;
        else
            disp('The learning rate vector for the prototypes does not fit the nb of epochs');
            return;
        end
    else
        % or use an decay with a start and a decay value
        if isempty(p.Results.learningRatePrototypes)
            initialization.learningRatePrototypes = [nb_features/100, nb_features/10000];
        end
        alpha_start = initialization.learningRatePrototypes(1);
        alpha_end = initialization.learningRatePrototypes(2);
        alphas = arrayfun(@(x) alpha_start * (alpha_end/alpha_start)^(x/nb_epochs), 1:nb_epochs);
    %     alphas = arrayfun(@(x) alpha_start / (1+(x-1)*alpha_end), 1:nb_epochs);
    end
    % compute the vector of nb_epochs learning rates epsilon for the Matrix learning
    epsilons = zeros(1,nb_epochs);
    if isa(p.Results.learningRateMatrix,'function_handle')
        % with a given function specified from the user
    % 	epsilons = arrayfun(p.Results.learningRateMatrix, 1:nb_epochs);
        epsilons(MatrixStart:nb_epochs) = arrayfun(p.Results.learningRateMatrix, MatrixStart:nb_epochs);
    elseif length(p.Results.learningRateMatrix)>2
        if length(p.Results.learningRateMatrix)==nb_epochs
            epsilons = p.Results.learningRateMatrix;
        else
            disp('The learning rate vector for the Matrix does not fit the nb of epochs');
            return;
        end
    else
        % or use an decay with a start and a decay value
        if isempty(p.Results.learningRateMatrix)
            initialization.learningRateMatrix = [nb_features/1000, nb_features/100000];
        end
        eps_start = initialization.learningRateMatrix(1);
        eps_end = initialization.learningRateMatrix(2);
    %     epsilons = arrayfun(@(x) eps_start * (eps_end/eps_start)^(x/nb_epochs), 1:nb_epochs);
        epsilons(MatrixStart:nb_epochs) = arrayfun(@(x) eps_start * (eps_end/eps_start)^((x-MatrixStart)/(nb_epochs-MatrixStart)), MatrixStart:nb_epochs);
    end
    
    %%% initialize requested outputs
    trainError = [];
    costs = [];
    testError = [];
    if nout>=2
        % train error requested
        trainError = ones(1,nb_epochs+1);
        estimatedLabels = LGMLVQ_classify(trainSet, model); % error after initialization
        trainError(1) = sum( trainLab ~= estimatedLabels )/nb_samples;
        if nout>=3
            % test error requested
            if isempty(testSet)
                testError = [];
                disp('The test error is requested, but no labeled test set given. Omitting the computation.');
            else
                testError = ones(1,nb_epochs+1);
                estimatedLabels = LGMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                testError(1) = sum( testSet(:,end) ~= estimatedLabels )/length(estimatedLabels);
            end        
            if nout>=4
                % costs requested
                disp('The computation of the costs is an expensive operation, do it only if you really need it!');
                costs = ones(1,nb_epochs+1);        
                costs(1) = LGMLVQ_costfun(trainSet, trainLab, model, regularization);
%                 costs(1) = sum(arrayfun(@(idx) GLVQ_costfun(min(dist(idx,model.c_w == trainLab(idx))),...
%                                                             min(dist(idx,model.c_w ~= trainLab(idx))))-regTerm, 1:size(dist,1)));
            end
        end
    end

    %%% optimize with stochastic gradient descent
    for epoch=1:nb_epochs
        if mod(epoch,100)==0, disp(epoch); end
        % generate order to sweep through the trainingset
        order = randperm(nb_samples);	
        % perform one sweep through trainingset
        regTerm = zeros(1,nb_samples);
        for i=1:nb_samples
            % select one training sample randomly
            xi = trainSet(order(i),:);
            c_xi = trainLab(order(i));
            dist = zeros(1,length(model.c_w));
            if classwise
                for j=1:length(model.c_w)
                    rightIdx = classes == model.c_w(j);
%                     dist(j) = (xi - model.w(j,:))*model.psis{rightIdx}'*(model.psis{rightIdx}*(xi - model.w(j,:))');
                    dist(j) = sum((bsxfun(@minus, xi, model.w(j,:))*model.psis{rightIdx}').^2, 2);
                end
            else
                for j=1:length(model.c_w)
%                     dist(j) = (xi - model.w(j,:))*model.psis{j}'*(model.psis{j}*(xi - model.w(j,:))');
                    dist(j) = sum((bsxfun(@minus, xi, model.w(j,:))*model.psis{j}').^2, 2);
                end
            end
            % determine the two winning prototypes
            % nearest prototype with the same class
            [sortDist,sortIdx] = sort(dist);
            count = 1;
            J = sortIdx(count);
            while model.c_w(sortIdx(count)) ~= c_xi
                count = count+1;
                J = sortIdx(count);
            end
            dJ = sortDist(count);
            count = 1;
            K = sortIdx(count);
            while model.c_w(sortIdx(count)) == c_xi
                count = count+1;
                K = sortIdx(count);
            end
            dK = sortDist(count);
    %         disp([J,K,dJ,dK]);
            wJ = model.w(J,:);
            wK = model.w(K,:);
            % prototype update
            norm_factor = (dJ + dK)^2;
            DJ = (xi-wJ);
            DK = (xi-wK);

            if classwise
                rightIdxJ = find(model.c_w(J)==classes);
                rightIdxK = find(model.c_w(K)==classes);
            else
                rightIdxJ = J;
                rightIdxK = K;
            end
			dwJ = (2*dK/norm_factor)*2*model.psis{rightIdxJ}'*model.psis{rightIdxJ}*DJ';
			dwK = (2*dJ/norm_factor)*2*model.psis{rightIdxK}'*model.psis{rightIdxK}*DK';
    
            model.w(J,:) = wJ + alphas(epoch) * dwJ';
            model.w(K,:) = wK - alphas(epoch) * dwK';
            % update matrices
            if epsilons(epoch)>0 % epoch >= MatrixStart
                % compute updates for matrices
                f1 =  (2*dK/norm_factor)*2*(model.psis{rightIdxJ}*DJ'*DJ);
                f2 = (-2*dJ/norm_factor)*2*(model.psis{rightIdxK}*DK'*DK);
                f3J = 0;
                f3K = 0;
                if regularization(rightIdxJ)>0
                    f3J = (pinv(model.psis{rightIdxJ}))';                
                    regTerm(i) = regTerm(i)-regularization(rightIdxJ)*log(det(model.psis{rightIdxJ}*model.psis{rightIdxJ}'));
                end
                if regularization(rightIdxK)>0
                    f3K = (pinv(model.psis{rightIdxK}))';
                    regTerm(i) = regTerm(i)-regularization(rightIdxK)*log(det(model.psis{rightIdxK}*model.psis{rightIdxK}'));
                end
                % update lambda & normalization
                model.psis{rightIdxJ} = model.psis{rightIdxJ}-epsilons(epoch) * (f1 - regularization(rightIdxJ) * f3J);
                model.psis{rightIdxK} = model.psis{rightIdxK}-epsilons(epoch) * (f2 - regularization(rightIdxK) * f3K);
            end
        end
        if nout>=2
            % train error requested
            estimatedLabels = LGMLVQ_classify(trainSet, model); % error after epoch
            trainError(epoch+1) = sum( trainLab ~= estimatedLabels )/nb_samples;
            if nout>=3
                % test error requested
                if ~isempty(testSet)
                    estimatedLabels = LGMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                    testError(epoch+1) = sum( testSet(:,end) ~= estimatedLabels )/length(estimatedLabels);
                end 
                if nout>=4
                    % costs requested
                    costs(epoch+1) = LGMLVQ_costfun(trainSet, trainLab, model, regularization);
%                     costs(epoch+1) = sum(arrayfun(@(idx) GLVQ_costfun(min(dist(idx,model.c_w == trainLab(idx))),...
%                                                                       min(dist(idx,model.c_w ~= trainLab(idx))))-regTerm, 1:size(dist,1)));                    
                end
            end
        end
    end
case{'fminlbfgs'}
    %%% optimization options
    options = struct( ...
      'Display',p.Results.Display, ...
      'GradObj',p.Results.GradObj, ...
      'GradConstr',false, ...
      'GoalsExactAchieve',0, ...
      'TolFun',p.Results.TolFun, ...
      'MaxIter',p.Results.MaxIter, ...
      'MaxFunEvals', p.Results.MaxFunEvals, ...
      'TolX',p.Results.TolX, ...
      'DiffMinChange',p.Results.DiffMinChange, ...%       'OutputFcn','LVQ_progresser', ...
      'HessUpdate',p.Results.HessUpdate ...
    );
%     clear('progresser'); % memory therein might need reset  
%     global threshstop earlystopped useEarlyStopping % for LVQ_progresser.m datval labval n_vec
%     useEarlyStopping = p.Results.useEarlyStopping; % use early stopping
% %     useEarlyStopping = []; % if empty, no validation set given/used for early stopping
%     earlystopped = false;
%     threshstop = p.Results.threshstop; % stop if classification below this threshold for early stopping
%     prototypeLabel = model.c_w;
    nb_prototypes = numel(model.c_w);
%     LabelEqualsPrototype = trainLab*ones(1,nb_prototypes) == (model.c_w*ones(1,nb_samples))';     
    LabelEqualsPrototype = bsxfun(@eq,trainLab,model.c_w');
% %     LabelEqualsPrototype = bsxfun(@eq, trainLab, prototypeLabel');
%     earlystopped = false; % don't change, assigned in progresser.m
%     clear('progresser'); % memory therein might need reset
    newfval = realmax('single');
    % fminlbfgs optimizer courtesy of Dirk-Jan Kroon:       
    % http://www.mathworks.de/matlabcentral/fileexchange/23245
    % early stopping to be implemented in progresser function    
    variables = zeros(nb_prototypes+sum(dim),size(trainSet,2));
    variables(1:nb_prototypes,:) = model.w;
    variables(nb_prototypes+1:end,:) = cell2mat(model.psis');
    LRprototypes = 1; % learn prototype locations
    LRrelevances = 1; % don't learn metric
    [variables,fval,ExitFlag] = fminlbfgs(@(variables) LGMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,dim,regularization),variables,options);
    for actrep = 1:p.Results.optreps
        [variables,fval,ExitFlag] = fminlbfgs(@(variables) LGMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,dim,regularization),variables,options);
    end 
%     [variables,fval] = fminlbfgs(@(variables) LGMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,dim,regularization),variables,options);
%     if not(isempty(fval))
%         newfval = fval;
%     end
%     LRprototypes = 0; % don't learn prototype locations
%     LRrelevances = 1; % learn metric
%     [variables,fval] = fminlbfgs(@(variables) LGMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,dim,regularization),variables,options);
%     if not(isempty(fval))
%         newfval = fval;
%     end
%     if not(isempty(fval))
%       clear('progresser'); % memory therein might need reset
%       LRprototypes = 1; 
%       for i = 1:nb_reiterations  % depending on data, re-iterations might further improve
% %         [variables,fval] = fminlbfgs(@LGMLVQ_optfun,variables,options);
%         [variables,fval] = fminlbfgs(@(variables) LGMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,dim,regularization),variables,options);
%         if not(isempty(fval))
%           if abs(fval - newfval) < 1e-3 
%             newfval = fval;
%             break
%           end
%           newfval = fval;
%         end
%         if isempty(fval) % || earlystopped
%           break
%         end
%       end
%     end
    model.w = variables(1:nb_prototypes,:);
    model.psis = mat2cell(variables(nb_prototypes+1:end,:),dim,nb_features)';
    if nout>=2
        % train error requested
        estimatedLabels = LGMLVQ_classify(trainSet, model); % error after initialization
        trainError = mean( trainLab ~= estimatedLabels );
        if nout>=3
            % test error requested
            if isempty(testSet)
                testError = [];
                disp('The test error is requested, but no labeled test set given. Omitting the computation.');
            else
                estimatedLabels = LGMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                testError = mean( testSet(:,end) ~= estimatedLabels );
            end        
            if nout>=4
                % costs requested
                costs = LGMLVQ_costfun(trainSet, trainLab, model, regularization);
            end
        end
    end
end
%%% output of the training
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {initialization};
		case(2)
			varargout(k) = {trainError};
		case(3)
			varargout(k) = {testError};
		case(4)
            varargout(k) = {costs};
	end
end