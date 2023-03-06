function [model, varargout] = GMLVQ_train(trainSet, trainLab, varargin)
%GMLVQ_trai.m - trains the Generalized Matrix LVQ algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  GMLVQ_model=GMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = GMLVQ_classify(trainSet, GMLVQ_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
%  trainLab : vector with the labels of the training set
% optional parameters:
%  PrototypesPerClass: (default=1) the number of prototypes per class used. This could
%  be a number or a vector with the number for each class
%  initialPrototypes : (default=[]) a set of prototypes to start with. If not given initialization near the class means
%  initialMatrix     : the matrix omega to start with. If not given random
%  initialization for rectangular matrices and Unity for squared omega
%  dim               : (default=nb of features for training) the maximum rank or projection dimension
%  regularization    : (default=0) values usually between 0 and 1 treat with care. 
%  Regularizes the eigenvalue spectrum of omega'*omega to be more homogeneous
%  testSet           : (default=[]) an optional test set used to compute
%  the test error. The last column is expected to be a label vector
%  comparable        : (default=0) a flag which resets the random generator
%  to produce comparable results if set to 1
%  optimization      : (default=fminlbfgs) indicates which optimization is used: sgd or fminlbfgs
% parameter for the stochastic gradient descent sgd
%  nb_epochs             : (default=100) the number of epochs for sgd
%  learningRatePrototypes: (default=[]) the learning rate for the prototypes. 
%  Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  learningRateMatrix    : (default=[]) the learning rate for the matrix.
%  Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  MatrixStart           : (default=1) the epoch to start the matrix training
% parameter for the build-in function fminlbfgs
%  threshstop       : (default=0) the training error for early stopping
%  nb_reiterations  : (default=100) the number of optimization reiterations performed
%  useEarlyStopping : (default=1) use early stopping based on threshstop
%  Display          : (default=iter) the optimization output 'iter' or 'off'
%  GradObj          : (default=on) use the gradient information or not
%  HessUpdate       : (default=lbfgs) the update can be 'lbfgs', 'bfgs' or 'steepdesc'
%  TolFun           : (default=1e-6) the tolerance
%  MaxIter          : (default=2500) the maximal number of iterations
%  MaxFunEvals      : (default=1000000) the maximal number of function evaluations
%  TolX             : (default=1e-10) tolerance
%  DiffMinChange    : (default=1e-10) minimal change
%
% output: the GMLVQ model with prototypes w their labels c_w and the matrix omega 
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
% Kerstin Bunte (modified based on the code of Marc Strickert http://www.mloss.org/software/view/323/ and Petra Schneider)
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
p.addParameter('initialPrototypes',[], @(x)(size(x,2)-1==size(trainSet,2) && isfloat(x)));
p.addParameter('initialMatrix',[], @(x)(size(x,2)==size(trainSet,2) && isfloat(x)));
p.addParameter('dim',size(trainSet,2), @(x)(~(x-floor(x)) && x<=size(trainSet,2) && x>0));
p.addParameter('regularization',0, @(x)(isfloat(x) && x>=0));
p.addOptional('testSet', [], @(x)(size(x,2)-1)==size(trainSet,2) & isfloat(x));
p.addOptional('comparable', 0, @(x)(~(x-floor(x))));
p.addOptional('optimization', 'fminlbfgs', @(x)any(strcmpi(x,{'sgd','fminlbfgs'})));
% parameter for the stochastic gradient descent
p.addOptional('nb_epochs', 100, @(x)(~(x-floor(x))));
p.addOptional('learningRatePrototypes', [], @(x)(isfloat(x) || isa(x,'function_handle'))); % && (length(x)==2 || length(x)==p.Results.epochs)
p.addOptional('learningRateMatrix', [], @(x)(isfloat(x)  || isa(x,'function_handle')));
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
p.FunctionName = 'GMLVQ';
% Parse and validate all input arguments.
p.parse(trainSet, trainLab, varargin{:});

%%% check if results should be comparable
if p.Results.comparable
    rng('default');
end
%%% set useful variables
nb_samples = size(trainSet,1);
nb_features = size(trainSet,2);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end

classes = unique(trainLab);
nb_classes = length(classes);
dim = p.Results.dim;
MatrixStart = p.Results.MatrixStart;
testSet = p.Results.testSet;
% global regularization;
regularization = p.Results.regularization;
if regularization, disp(['Regularize the eigenvalue spectrum of omega''*omega with ',num2str(regularization)]);end

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
%     disp('The fminlbfgs optimization uses some global variables:');
%     disp('threshstop earlystopped useEarlyStopping');
    initialization = rmfield(initialization, 'nb_epochs');
    initialization = rmfield(initialization, 'learningRatePrototypes');
    initialization = rmfield(initialization, 'learningRateMatrix');
    initialization = rmfield(initialization, 'MatrixStart');
    nb_reiterations = p.Results.nb_reiterations;
end
% Display all arguments.
disp 'Settings for GMLVQ:'
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
%%% initialize the matrix
if isempty(p.Results.initialMatrix)
    if(p.Results.dim==nb_features)
        omega = eye(nb_features);
    else % initialize with random numbers between -1 and 1
        omega = rand(dim,nb_features)*2-ones(dim,nb_features);
    end
else
    omega = p.Results.initialMatrix;
end
% normalize the matrix
omega = omega / sqrt(sum(diag(omega'*omega)));
model = struct('w',w,'c_w',c_w,'omega',omega);
clear w c_w omega;

% GMLVQ_algorithm = struct('update',@GMLVQ_update,'classify',@GMLVQ_classify,'costfun',@GLVQ_costfun);
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
        estimatedLabels = GMLVQ_classify(trainSet, model); % error after initialization
        trainError(1) = sum( trainLab ~= estimatedLabels )/nb_samples;
        if nout>=3
            % test error requested
            if isempty(testSet)
                testError = [];
                disp('The test error is requested, but no labeled test set given. Omitting the computation.');
            else
                testError = ones(1,nb_epochs+1);
                estimatedLabels = GMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                testError(1) = sum( testSet(:,end) ~= estimatedLabels )/length(estimatedLabels);
            end        
            if nout>=4
                % costs requested
%                 LabelEqPrototype = trainLab*ones(1,numel(model.c_w)) == (model.c_w*ones(1,nb_samples))';
                disp('The computation of the costs is an expensive operation, do it only if you really need it!');
                costs = ones(1,nb_epochs+1);
                costs(1) = GMLVQ_costfun(trainSet, trainLab, model, regularization);
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
        for i=1:nb_samples
            % select one training sample randomly
            xi = trainSet(order(i),:);
            c_xi = trainLab(order(i));

%             dist = ((xi(ones(size(model.w,1),1),:))-model.w)*model.omega'*(model.omega*((xi(ones(size(model.w,1),1),:))-model.w)');
%             dist = diag(dist);
            dist = sum((bsxfun(@minus, xi, model.w)*model.omega').^2, 2);
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

            oo = model.omega'*model.omega;

            dwJ = (2*dK/norm_factor)*2*oo*DJ';
            dwK = (2*dJ/norm_factor)*2*oo*DK';
    %         dwJ = (2*dK/norm_factor)*2*model.omega'*model.omega*DJ';
    %         dwK = (2*dJ/norm_factor)*2*model.omega'*model.omega*DK';
            model.w(J,:) = wJ + alphas(epoch) * dwJ';
            model.w(K,:) = wK - alphas(epoch) * dwK';
            % update matrices
            if epsilons(epoch)>0 % epoch >= MatrixStart
                f1 = (2*dK/norm_factor)*2*(model.omega*DJ')*DJ;
                f2 = (2*dJ/norm_factor)*2*(model.omega*DK')*DK;
                % update omega
                if regularization
                    f3 = (pinv(model.omega))';                
                else
                    f3 = 0;
                end
                model.omega = model.omega-epsilons(epoch) * (f1-f2  - regularization * f3);
                % normalization
                model.omega = model.omega / sqrt(sum(diag(oo)));
            end
        end
        if nout>=2
            % train error requested
            estimatedLabels = GMLVQ_classify(trainSet, model); % error after epoch
            trainError(epoch+1) = sum( trainLab ~= estimatedLabels )/nb_samples;
            if nout>=3
                % test error requested
                if ~isempty(testSet)
                    estimatedLabels = GMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                    testError(epoch+1) = sum( testSet(:,end) ~= estimatedLabels )/length(estimatedLabels);
                end 
                if nout>=4
                    % costs requested
                    costs(epoch+1) = GMLVQ_costfun(trainSet, trainLab, model, regularization);
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
      'DiffMinChange',p.Results.DiffMinChange, ...
      'HessUpdate',p.Results.HessUpdate ...
    );%       'OutputFcn','LVQ_progresser', ...
% clear('progresser'); % memory therein might need reset  
% global threshstop earlystopped useEarlyStopping training_data training_label prototypeLabel% for LVQ_progresser.m datval labval n_vec
% useEarlyStopping = p.Results.useEarlyStopping; % use early stopping
% earlystopped = false;
% threshstop = p.Results.threshstop; % stop if classification below this threshold for early stopping
% training_data = trainSet;
% training_label = trainLab;
% prototypeLabel = model.c_w;
%     earlystopped = false; % don't change, assigned in progresser.m
%     clear('progresser'); % memory therein might need reset
%     save('test','model','nb_ppc');
    LabelEqualsPrototype = bsxfun(@eq,trainLab,model.c_w');
%     newfval = realmax('single');
    % fminlbfgs optimizer courtesy of Dirk-Jan Kroon:       
    % http://www.mathworks.de/matlabcentral/fileexchange/23245
    % early stopping to be implemented in progresser function    
    variables = [model.w;model.omega];

    LRprototypes = 1; % learn prototype locations
    LRrelevances = 0; % don't learn metric
    [variables] = fminlbfgs(@(variables) GMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,regularization),variables,options); %,fval
%     if not(isempty(fval))
%         newfval = fval;
%     end
    LRprototypes = 0; % don't learn prototype locations
    LRrelevances = 1; % learn metric    %     [variables,fval] = fminlbfgs(@GMLVQ_optfun,variables,options);      
    [variables] = fminlbfgs(@(variables) GMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,regularization),variables,options); % ,fval
%     if not(isempty(fval))
%         newfval = fval;
%     end

%     if not(isempty(fval))
%       clear('progresser'); % memory therein might need reset
      LRprototypes = 1; 
      for i = 1:nb_reiterations  % depending on data, re-iterations might further improve
%         [variables,fval] = fminlbfgs(@GMLVQ_optfun,variables,options);
        [variables] = fminlbfgs(@(variables) GMLVQ_optfun(variables,trainSet,LabelEqualsPrototype,LRrelevances,LRprototypes,model.c_w,regularization),variables,options); % ,fval
%         if not(isempty(fval))
%           if abs(fval - newfval) < 1e-3 
%             newfval = fval;
%             break
%           end
%           newfval = fval;
%         end
%         if isempty(fval) || earlystopped
%           break
%         end
      end
%     end
    nb_prototypes = sum(nb_ppc);
    model.w = variables(1:nb_prototypes,:);
    model.omega = variables(nb_prototypes+1:end,:);
%     model.omega = model.omega / sqrt(sum(diag(model.omega'*model.omega)));
    if nout>=2
        % train error requested
        estimatedLabels = GMLVQ_classify(trainSet, model); % error after initialization
        trainError = mean( trainLab ~= estimatedLabels );
        if nout>=3            
            % test error requested
            if isempty(testSet)
                testError = [];
                disp('The test error is requested, but no labeled test set given. Omitting the computation.');
            else
                estimatedLabels = GMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                testError = mean( testSet(:,end) ~= estimatedLabels );
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
	end
end