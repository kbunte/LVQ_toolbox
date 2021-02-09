function [estimatedLabels, varargout] = LLiRaMLVQ_classify(Data, model)
%LGMLVQ_classify.m - classifies the given data with the given model
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  LLiRaM_model=LLiRaMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = LLiRaMLVQ_classify(trainSet, LLiRaM_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
%  model    : LLiRaMLVQ model with prototypes w their labels c_w and matrices omega and psi
% 
% output    : the estimated labels
%  
% Kerstin Bunte
% kerstin.bunte@googlemail.com
% Mon Nov 23 14:05:52 CEST 2019
%
% Conditions of GNU General Public License, version 2 apply.
% See file 'license-gpl2.txt' enclosed in this package.
% Programs are not for use in critical applications!
%
nout = max(nargout,1)-1;
nb_samples = size(Data,1);

dist = zeros(nb_samples,length(model.c_w));
% dist = computeDistance(Data, model.w, model);
if length(model.psis)~=size(model.w,1)
    classes = unique(model.c_w);
    for i = 1:size(model.w,1)
        matrixIdx = classes==model.c_w(i);
        dist(1:nb_samples,i) = sum((bsxfun(@minus, Data, model.w(i,:))*model.omega'*model.psis{matrixIdx}').^2, 2);
    end
else
    for i = 1:size(model.w,1)
        dist(1:nb_samples,i) = sum((bsxfun(@minus, Data, model.w(i,:))*model.omega'*model.psis{i}').^2, 2);
    end
end
[~, index] = min(dist,[],2);


estimatedLabels = model.c_w(index);
if nout>0
    if length(unique(model.c_w))>2
        error('more than 2 classes! The score return value is not meant for this!');
    else
        scores = 0.5*(1+(dist(:,2)-dist(:,1))./(dist(:,2)+dist(:,1)));
    end    
end

%%% additional output
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {scores};
        case(2)
			varargout(k) = {dist};
	end
end