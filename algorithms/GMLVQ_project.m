function projection = GMLVQ_project(data, model, dim)
%GMLVQ_project.m - projects the given data with the given model
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  GMLVQ_model=GMLVQ_train(trainSet,trainLab); % minimal parameters required
%  trainprojection = GMLVQ_project(trainSet, GMLVQ_model, 2);
%  gscatter(trainprojection(:,1),trainprojection(:,2),trainLab,'','o',4,'off','dim 1','dim 2');box on;title('2 dim projection of the training data');
%
% input: 
%  data  : matrix with training samples in its rows
%  model : GMLVQ model with prototypes w their labels c_w and the matrix omega
%  dim   : the target dimension for the projection
% 
% output : the procection of the data
%  
% Kerstin Bunte
% kerstin.bunte@googlemail.com
% Mon Nov 05 09:05:52 CEST 2012
%
% Conditions of GNU General Public License, version 2 apply.
% See file 'license-gpl2.txt' enclosed in this package.
% Programs are not for use in critical applications!
% 
% [U,S,~] = svd(model.omega'*model.omega);
% model.omega = sqrt(diag(S(1:dim,1:dim))).*U(:,1:dim)';
% projection = data*model.omega';

[V,D] = eigs(model.omega'*model.omega);
model.omega = sqrt(diag(D(1:dim,1:dim))).*V(:,1:dim)';
projection = data*model.omega';

% if size(model.omega,1)>dim
%     lambda = model.omega'*model.omega;
%     [U,~,~] = svd(lambda);
%     A = U(:,1:dim)';
% else
%     A = model.omega;
% end
% [U,D,V] = svd(LMNNMatrices(:,:,i,c,actK)'*LMNNMatrices(:,:,i,c,actK));
% [val,idx] = max(abs(U));
% for count=1:size(LMNNMatrices,1)
%     if U(idx(count),count)>0, U(:,count) =  U(:,count);
%     else 		 		 	U(:,count) = -U(:,count);end
% end
% LMNNCans(:,:,i,c,actK) = (U(:,1:size(LMNNMatrices,1))*sqrt(D(1:size(LMNNMatrices,1),1:size(LMNNMatrices,1))))';